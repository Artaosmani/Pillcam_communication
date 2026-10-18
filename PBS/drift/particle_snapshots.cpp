//* particle_snapshots.cpp
//
// Small reactive-receiver PBS run that saves particle positions every k
// steps, for the offline particle animation (visualizers/animate_particles.py).
// Never part of the timed loop.
//
// It uses the same engine as example_reactive.cpp / bbox_replay.cpp and
// only observes it through channel::setStepHook(). Particles that leave the
// engine's vector are classified exactly:
//   gone between phase 0 and 1 (propagation) -> DEGRADED in the medium
//   gone between phase 1 and 2 (RX.process)  -> INTERNALIZED
// Terminal particles keep their last position and are written in every
// later snapshot, so the animation can show them.
//
// A second, per-step check counts FREE particles that are still inside the
// sphere after RX.process. With working push-back this is always 0.
//
// Output CSV: particle_snapshots.csv
//   step,t_s,id,x_m,y_m,z_m,state      state: 0 free, 1 bound, 2 internalized, 3 degraded
// plus a one-line header file particle_snapshots_meta.csv:
//   r0_m,a_m,D,dt,T,N,kf,kb,ki,kd,every,free_inside_after_process
//
// Compile (from drift/):
//   g++ -O3 -fopenmp particle_snapshots.cpp \
//       src_drift/particle.cpp src_drift/channel.cpp -o particle_snapshots
//
// Run:
//   ./particle_snapshots                       # N 200, dt 5e-8, every 100 steps
//   ./particle_snapshots --N 200 --every 50 --r0 1e-6 --a 0.5e-6

#include "src_drift/channel.h"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <cmath>
#include <cstring>
#include <cstdlib>

using namespace molcompbs;
using namespace std;

static double getf(int argc, char** argv, const char* flag, double def) {
    for (int i = 1; i < argc - 1; ++i)
        if (strcmp(argv[i], flag) == 0) return atof(argv[i + 1]);
    return def;
}
static string gets(int argc, char** argv, const char* flag, const string& def) {
    for (int i = 1; i < argc - 1; ++i)
        if (strcmp(argv[i], flag) == 0) return argv[i + 1];
    return def;
}

struct Rec { double x, y, z; int state; };

int main(int argc, char** argv)
{
    // Table I reference (same as bbox_replay.cpp Settings)
    double D  = getf(argc, argv, "--D",  5e-9);
    double T  = getf(argc, argv, "--T",  1e-3);
    double kf = getf(argc, argv, "--kf", 1e-14);
    double kb = getf(argc, argv, "--kb", 2e3);
    double ki = getf(argc, argv, "--ki", 1e4);
    double kd = getf(argc, argv, "--kd", 2e3);
    double dt = getf(argc, argv, "--dt", 5e-8);
    int    N  = (int)getf(argc, argv, "--N", 200);
    double r0 = getf(argc, argv, "--r0", 1e-6);
    double a  = getf(argc, argv, "--a",  0.5e-6);
    int every = (int)getf(argc, argv, "--every", 100);
    const string out = gets(argc, argv, "--out", "particle_snapshots.csv");

    channel ch;
    ch.setD(D); ch.setdt(dt); ch.setT(T); ch.setN(N);
    ch.setBit({true});
    ch.setKd(kd);
    ch.setTX(0.0, 0.0, 0.0);
    ch.setRXReactive(r0, 0.0, 0.0, a, kf, kb, ki);

    ofstream f(out);
    if (!f.is_open()) { cerr << "Cannot open " << out << "\n"; return 1; }
    f << "step,t_s,id,x_m,y_m,z_m,state\n";
    f.precision(6);
    f << scientific;

    unordered_map<int, Rec> last;      // last known record of every id
    unordered_map<int, Rec> terminal;  // ids that reached a terminal state
    long long free_inside_after = 0;   // push-back check
    const double a2 = a * a;

    auto mark_missing = [&](const vector<particle>& ps, int new_state) {
        unordered_map<int, char> alive;
        alive.reserve(ps.size() * 2);
        for (const auto& p : ps) alive[p.id] = 1;
        for (auto& kv : last)
            if (!alive.count(kv.first) && !terminal.count(kv.first)) {
                Rec r = kv.second; r.state = new_state;
                terminal[kv.first] = r;
            }
    };
    auto remember = [&](const vector<particle>& ps) {
        for (const auto& p : ps)
            last[p.id] = { p.getX(), p.getY(), p.getZ(),
                           p.isBound() ? 1 : 0 };
    };

    ch.setStepHook([&](int n, double t, int phase, const vector<particle>& ps) {
        if (phase == 0) { remember(ps); return; }
        if (phase == 1) { mark_missing(ps, 3); remember(ps); return; }
        // phase 2: after the receiver
        mark_missing(ps, 2);
        remember(ps);
        for (const auto& p : ps) {
            if (!p.isFree()) continue;
            double dx = p.getX() - r0, dy = p.getY(), dz = p.getZ();
            if (dx*dx + dy*dy + dz*dz <= a2) ++free_inside_after;
        }
        if (n % every != 0) return;
        for (const auto& p : ps)
            f << n << "," << t << "," << p.id << "," << p.getX() << ","
              << p.getY() << "," << p.getZ() << "," << (p.isBound() ? 1 : 0) << "\n";
        for (const auto& kv : terminal)
            f << n << "," << t << "," << kv.first << "," << kv.second.x << ","
              << kv.second.y << "," << kv.second.z << "," << kv.second.state << "\n";
    });

    vector<SimStep> steps = ch.simulateDetailed();
    f.close();

    int n_int = 0, n_deg = 0;
    for (auto& kv : terminal) (kv.second.state == 2 ? n_int : n_deg)++;
    int engine_int = 0;
    for (auto& s : steps) engine_int += s.N_internalized;

    ofstream m(out.substr(0, out.rfind('.')) + "_meta.csv");
    m << "r0_m,a_m,D,dt,T,N,kf,kb,ki,kd,every,free_inside_after_process\n"
      << r0 << "," << a << "," << D << "," << dt << "," << T << "," << N << ","
      << kf << "," << kb << "," << ki << "," << kd << "," << every << ","
      << free_inside_after << "\n";

    cout << "Steps " << steps.size() << ", snapshots every " << every
         << "\nInternalized " << n_int << " (engine count " << engine_int << ")"
         << ", degraded " << n_deg
         << "\nFREE particles inside the sphere after RX.process: " << free_inside_after
         << (free_inside_after == 0 ? "  (push-back OK)" : "  (CHECK push-back)")
         << "\nSaved " << out << "\n";
    return 0;
}
