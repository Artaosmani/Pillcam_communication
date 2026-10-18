//* bbox_replay.cpp
//
// Replays detector bounding boxes through the reactive-receiver PBS, one box
// at a time, and times each box. This is the offline stand-in for the live
// link from the detector: later, boxes arrive over the network instead of
// from a CSV, and only the input loop in main() changes. process_box() and
// its timing stay the same.
//
// Input CSV (one row per box; a frame with two boxes is two rows):
//   frame_id,track_id,t_capture_ns,t_detect_ns,t_send_ns,
//   x_min,y_min,x_max,y_max,img_w,img_h,confidence
//   Absolute pixels, origin top-left, x right, y down.
//   data/boxes_fake.csv is made-up data with this layout; replace it with
//   the real detector output.
//
// Output CSV (one row per box): bbox_replay_out.csv
//   frame_id,track_id,r0_m,a_m,Pi_T,t_half_s,inject,dose_N,latency_ms,
//   mapping_ms,lookup_ms,decision_ms
//   latency_ms is the wall time of process_box() only. Reading the input
//   file and writing the output file are outside the timed region.
//   mapping_ms / lookup_ms / decision_ms split that time into the three
//   server stages for the latency-breakdown figure (visualizers/). lookup_ms
//   is the full PBS run for now; it becomes the table lookup later.
//
// Compile (from drift/):
//   g++ -O3 -fopenmp bbox_replay.cpp \
//       src_drift/particle.cpp src_drift/channel.cpp \
//       -o bbox_replay
//
// Run:
//   ./bbox_replay                              # all rows of data/boxes_fake.csv
//   ./bbox_replay --csv my_boxes.csv           # real detector output
//   ./bbox_replay --rows 1                     # only the first box
//   ./bbox_replay --N 1000 --dt 5e-8           # PBS settings per box
//   ./bbox_replay --conf 0.5 --need 100        # decision placeholders

#include "src_drift/channel.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <iomanip>

using namespace molcompbs;
using namespace std;

// -------------------------------------------------------------------------
// Minimal "--flag value" parser (same style as example_reactive.cpp)
// -------------------------------------------------------------------------
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

// -------------------------------------------------------------------------
// One detection as read from the CSV
// -------------------------------------------------------------------------
struct Box {
    long long frame_id, track_id;
    long long t_capture_ns, t_detect_ns, t_send_ns;
    double x_min, y_min, x_max, y_max;
    double img_w, img_h;
    double confidence;
};

struct Decision {
    double r0, a;       // geometry used by the PBS [m]
    double Pi_T;        // internalized fraction at the end of the window
    double t_half;      // time for Pi to reach half of Pi_T [s]
    bool   inject;
    double dose_N;      // molecules to release (placeholder rule)
    double ms_map, ms_lookup, ms_decide;   // per-stage wall time [ms]
};

struct Settings {
    // Table I reference rates (Khoshfekr Rudsari et al., IEEE Trans. Commun. 73(2):817-831, 2025)
    double D  = 5e-9;
    double T  = 1e-3;
    double kf = 1e-14;
    double kb = 2e3;
    double ki = 1e4;
    double kd = 2e3;
    // PBS fidelity per box
    double dt = 5e-8;
    int    N  = 1000;
    // Decision placeholders
    double conf_min = 0.5;   // below this confidence: no injection
    double need     = 100.0; // molecules that must be internalized
};

// -------------------------------------------------------------------------
// Box -> geometry. PLACEHOLDER mapping: the pixel-to-metre assumption is
// still open (working distance, field of view or reference object). This
// only keeps r0 and a in the range of the Table I reference (a = 0.5 um,
// r0 = 1 um) so the PBS runs on sensible values. Replace when agreed.
//
//   a  : grows with box size (sqrt of normalized area), 0.25 .. 1.0 um
//   r0 : injection point assumed at the image centre; distance from the
//        image centre to the box centre (normalized by the half-diagonal)
//        adds 0 .. 1 um on top of a + 0.25 um, so r0 > a always.
// -------------------------------------------------------------------------
static void box_to_geometry(const Box& b, double& r0, double& a)
{
    double w_n  = (b.x_max - b.x_min) / b.img_w;
    double h_n  = (b.y_max - b.y_min) / b.img_h;
    double cx_n = 0.5 * (b.x_min + b.x_max) / b.img_w - 0.5;
    double cy_n = 0.5 * (b.y_min + b.y_max) / b.img_h - 0.5;
    double dist_n = std::sqrt(cx_n * cx_n + cy_n * cy_n) / std::sqrt(0.5);

    a  = 0.25e-6 + 0.75e-6 * std::sqrt(std::max(0.0, w_n * h_n));
    r0 = a + 0.25e-6 + 1.0e-6 * std::min(1.0, dist_n);
}

// -------------------------------------------------------------------------
// The per-box work that is timed: geometry, PBS, decision.
// -------------------------------------------------------------------------
static double ms_since(chrono::steady_clock::time_point t0)
{
    return chrono::duration<double, milli>(chrono::steady_clock::now() - t0).count();
}

static Decision process_box(const Box& b, const Settings& s)
{
    Decision d{};
    auto t0 = chrono::steady_clock::now();
    box_to_geometry(b, d.r0, d.a);
    d.ms_map = ms_since(t0);

    // Low confidence: decide "no" without simulating.
    if (b.confidence < s.conf_min) {
        t0 = chrono::steady_clock::now();
        d.inject = false;
        d.ms_decide = ms_since(t0);
        return d;
    }

    t0 = chrono::steady_clock::now();
    channel ch;
    ch.setD(s.D);
    ch.setdt(s.dt);
    ch.setT(s.T);
    ch.setN(s.N);
    ch.setBit({true});
    ch.setKd(s.kd);
    ch.setTX(0.0, 0.0, 0.0);
    ch.setRXReactive(d.r0, 0.0, 0.0, d.a, s.kf, s.kb, s.ki);

    vector<SimStep> steps = ch.simulateDetailed();

    // Cumulative internalized fraction, and when it reaches half its final value
    vector<double> Pi(steps.size());
    int cum = 0;
    for (size_t n = 0; n < steps.size(); ++n) {
        cum += steps[n].N_internalized;
        Pi[n] = double(cum) / s.N;
    }
    d.Pi_T = Pi.empty() ? 0.0 : Pi.back();
    d.t_half = 0.0;
    for (size_t n = 0; n < steps.size(); ++n)
        if (Pi[n] >= 0.5 * d.Pi_T && d.Pi_T > 0.0) { d.t_half = steps[n].t; break; }
    d.ms_lookup = ms_since(t0);

    // PLACEHOLDER decision: inject if anything reaches the target, and release
    // enough molecules that N * Pi_T covers the required amount.
    t0 = chrono::steady_clock::now();
    d.inject = d.Pi_T > 0.0;
    d.dose_N = d.inject ? std::ceil(s.need / d.Pi_T) : 0.0;
    d.ms_decide = ms_since(t0);
    return d;
}

// -------------------------------------------------------------------------
static vector<Box> read_boxes(const string& path)
{
    vector<Box> boxes;
    ifstream f(path);
    if (!f.is_open()) {
        cerr << "Cannot open " << path << "\n";
        return boxes;
    }
    string line;
    getline(f, line);   // header
    while (getline(f, line)) {
        if (line.empty()) continue;
        for (char& c : line) if (c == ',') c = ' ';
        istringstream ss(line);
        Box b;
        if (ss >> b.frame_id >> b.track_id >> b.t_capture_ns >> b.t_detect_ns >> b.t_send_ns
               >> b.x_min >> b.y_min >> b.x_max >> b.y_max >> b.img_w >> b.img_h >> b.confidence)
            boxes.push_back(b);
        else
            cerr << "Skipping malformed row: " << line << "\n";
    }
    return boxes;
}

static double percentile(vector<double> v, double p)
{
    if (v.empty()) return 0.0;
    sort(v.begin(), v.end());
    size_t k = (size_t)std::ceil(p / 100.0 * v.size());
    return v[std::min(v.size() - 1, k == 0 ? 0 : k - 1)];
}

int main(int argc, char** argv)
{
    Settings s;
    s.dt       = getf(argc, argv, "--dt", s.dt);
    s.N        = (int)getf(argc, argv, "--N", s.N);
    s.kd       = getf(argc, argv, "--kd", s.kd);
    s.conf_min = getf(argc, argv, "--conf", s.conf_min);
    s.need     = getf(argc, argv, "--need", s.need);
    const string in_path  = gets(argc, argv, "--csv", "data/boxes_fake.csv");
    const string out_path = gets(argc, argv, "--out", "bbox_replay_out.csv");
    const int    rows     = (int)getf(argc, argv, "--rows", 0);   // 0 = all

    // Load everything before timing starts
    vector<Box> boxes = read_boxes(in_path);
    if (boxes.empty()) { cerr << "No boxes read from " << in_path << "\n"; return 1; }
    size_t n_run = (rows > 0) ? std::min((size_t)rows, boxes.size()) : boxes.size();

    cout << "Replaying " << n_run << " of " << boxes.size() << " boxes from " << in_path
         << "  (N = " << s.N << ", dt = " << s.dt << ")\n";

    vector<Decision> dec(n_run);
    vector<double>   lat_ms(n_run);
    for (size_t k = 0; k < n_run; ++k) {
        auto t0 = chrono::steady_clock::now();
        dec[k] = process_box(boxes[k], s);                      // the only thing timed
        auto t1 = chrono::steady_clock::now();
        lat_ms[k] = chrono::duration<double, milli>(t1 - t0).count();
    }

    // Write results after timing
    ofstream f(out_path);
    if (!f.is_open()) { cerr << "Cannot open " << out_path << "\n"; return 1; }
    f << "frame_id,track_id,r0_m,a_m,Pi_T,t_half_s,inject,dose_N,latency_ms,mapping_ms,lookup_ms,decision_ms\n";
    for (size_t k = 0; k < n_run; ++k) {
        const Box& b = boxes[k];
        const Decision& d = dec[k];
        f << b.frame_id << "," << b.track_id << ","
          << scientific << setprecision(4) << d.r0 << "," << d.a << ","
          << d.Pi_T << "," << d.t_half << ","
          << defaultfloat << (d.inject ? 1 : 0) << ","
          << fixed << setprecision(0) << d.dose_N << ","
          << setprecision(3) << lat_ms[k] << ","
          << setprecision(4) << d.ms_map << "," << d.ms_lookup << "," << d.ms_decide
          << defaultfloat << "\n";
    }
    f.close();

    double mean = 0.0;
    for (double x : lat_ms) mean += x;
    mean /= lat_ms.size();
    cout << fixed << setprecision(2)
         << "Per-box latency [ms]: mean " << mean
         << ", p50 " << percentile(lat_ms, 50)
         << ", p95 " << percentile(lat_ms, 95)
         << ", max " << *max_element(lat_ms.begin(), lat_ms.end()) << "\n"
         << "Saved " << out_path << "\n";
    return 0;
}
