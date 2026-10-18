//* example_reactive.cpp
//
// Full per-timestep trace of the reactive receiver, in the CSV format
// consumed by plot_PBS.m for direct overlay against the MATLAB analytical
// curves. Kinetic model: H. Khoshfekr Rudsari, M. Damrath, M. Zoofaghari,
// I. Balasingham, and M. Veletic, "An Intercellular Communication System for
// Intra-Body Communication Networks," IEEE Trans. Commun., vol. 73, no. 2,
// pp. 817-831, Feb. 2025. doi:10.1109/TCOMM.2024.3440877
//
// CSV columns (matches plot_PBS.m):
//   t     : time [s]
//   Pe    : P(FREE)       — fraction currently free (diffusing)
//   Pf    : P(BOUND)      — fraction currently bound (AR)
//   Pi    : cum. P(INT)   — cumulative internalized fraction
//   Pd    : cum. P(DEG)   — cumulative degraded fraction (medium + AR)
//   Psum  : Pe+Pf+Pi+Pd   — conservation check; should be 1.0 every row
//   Pabs  : cum. P(ABS)   — absorbing-Rx reference (upper bound for Pi)
//
// Compile (from repo root):
//   g++ -O3 -fopenmp example_reactive.cpp \
//       src_drift/particle.cpp src_drift/channel.cpp \
//       -o example_reactive
//
// Run:
//   ./example_reactive                 # Table I reference, dt=5e-9, N=50k
//   ./example_reactive --dt 5e-8       # faster, slightly lower fidelity
//   ./example_reactive --N  100000     # more particles = less MC noise
//   ./example_reactive --no-abs        # skip the absorbing-Rx reference run
//
// The output file (example_reactive.csv) is the input to plot_PBS.m — run
// that in MATLAB on top of the figure produced by your analytical script.

#include "src_drift/channel.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <iomanip>

using namespace molcompbs;
using namespace std;

// -------------------------------------------------------------------------
// Minimal "--flag value" parser (matches sbi_runner.cpp style)
// -------------------------------------------------------------------------
static double getf(int argc, char** argv, const char* flag, double def) {
    for (int i = 1; i < argc - 1; ++i)
        if (strcmp(argv[i], flag) == 0) return atof(argv[i + 1]);
    return def;
}
static bool has_flag(int argc, char** argv, const char* flag) {
    for (int i = 1; i < argc; ++i)
        if (strcmp(argv[i], flag) == 0) return true;
    return false;
}

int main(int argc, char** argv)
{
    // ---------------------------------------------------------------------
    // Table I reference parameters (Khoshfekr Rudsari et al., IEEE Trans. Commun. 73(2):817-831, 2025; doi:10.1109/TCOMM.2024.3440877)
    // ---------------------------------------------------------------------
    const double D  = 5e-9;     // diffusion coefficient [m^2/s]
    const double a  = 0.5e-6;   // receiver radius       [m]
    const double r0 = 1e-6;     // Tx-Rx distance        [m]
    const double T  = 1e-3;     // observation window    [s]
    const double kf = 1e-14;    // forward binding rate  [m^3 s^-1]
    const double kb = 2e3;      // unbinding rate        [s^-1]
    const double ki = 1e4;      // internalization rate  [s^-1]
    // Degradation rate — CLI-overridable for the kd=0 isolation test.
    const double kd = getf(argc, argv, "--kd", 2e3);

    // Tunables
    const double dt    = getf(argc, argv, "--dt", 5e-9);
    const int    N     = (int)getf(argc, argv, "--N", 50000.0);
    const bool   do_abs = !has_flag(argc, argv, "--no-abs");

    const string fname = "example_reactive.csv";

    cout << "Reactive receiver, Table I reference:\n"
         << "  D  = " << D  << "   a  = " << a  << "   r0 = " << r0 << "\n"
         << "  kf = " << kf << "   kb = " << kb << "   ki = " << ki << "   kd = " << kd << "\n"
         << "  dt = " << dt << "   T  = " << T  << "   N  = " << N  << "\n\n";

    // ---------------------------------------------------------------------
    // REACTIVE run
    // ---------------------------------------------------------------------
    channel ch;
    ch.setD(D);
    ch.setdt(dt);
    ch.setT(T);
    ch.setN(N);
    ch.setBit({true});
    ch.setTX(0.0, 0.0, 0.0);
    ch.setRXReactive(r0, 0.0, 0.0, a, kf, kb, ki);
    ch.setKd(kd);

    cout << "Simulating REACTIVE   ..." << flush;
    vector<SimStep> steps = ch.simulateDetailed();
    cout << " [DONE]\n";

    // ---------------------------------------------------------------------
    // ABSORBING reference (for Pabs upper bound)
    // ---------------------------------------------------------------------
    vector<SimStep> steps_abs;
    if (do_abs) {
        channel ch_abs;
        ch_abs.setD(D);
        ch_abs.setdt(dt);
        ch_abs.setT(T);
        ch_abs.setN(N);
        ch_abs.setBit({true});
        ch_abs.setTX(0.0, 0.0, 0.0);
        ch_abs.setRX(r0, 0.0, 0.0, a, /*transparent=*/false);
        // no kd for a clean absorbing reference

        cout << "Simulating ABSORBING  ..." << flush;
        steps_abs = ch_abs.simulateDetailed();
        cout << " [DONE]\n";

        if (steps.size() != steps_abs.size()) {
            cerr << "Step count mismatch between REACTIVE (" << steps.size()
                 << ") and ABSORBING (" << steps_abs.size() << ")\n";
            return 1;
        }
    }

    // ---------------------------------------------------------------------
    // Write per-timestep CSV
    // ---------------------------------------------------------------------
    ofstream f(fname);
    if (!f.is_open()) {
        cerr << "Cannot open " << fname << "\n";
        return 1;
    }
    f << "t,Pe,Pf,Pi,Pd,Psum,Pabs\n";
    f << scientific << setprecision(9);

    int cum_int     = 0;
    int cum_deg     = 0;
    int cum_int_abs = 0;

    for (size_t n = 0; n < steps.size(); ++n) {
        const auto& s = steps[n];
        cum_int += s.N_internalized;
        cum_deg += s.N_degraded_med + s.N_degraded_rx;

        // N_active = FREE + BOUND (INTERNALIZED and DEGRADED are erased)
        int n_free = s.N_active - s.N_bound;

        double Pe   = double(n_free)    / N;
        double Pf   = double(s.N_bound) / N;
        double Pi   = double(cum_int)   / N;
        double Pd   = double(cum_deg)   / N;
        double Psum = Pe + Pf + Pi + Pd;

        double Pabs = 0.0;
        if (do_abs) {
            cum_int_abs += steps_abs[n].N_internalized;
            Pabs         = double(cum_int_abs) / N;
        }

        f << s.t << "," << Pe << "," << Pf << ","
          << Pi   << "," << Pd << "," << Psum << "," << Pabs << "\n";
    }
    f.close();

    cout << "\nSaved " << fname << "  (" << steps.size() << " rows)\n"
         << "  Pi(tau)   = " << double(cum_int)     / N << "\n";
    if (do_abs)
        cout << "  Pabs(tau) = " << double(cum_int_abs) / N
             << "   (absorbing Rx upper bound)\n";
    cout << "  Psum(tau) = "
         << (double(steps.back().N_active) / N
             + double(cum_int) / N
             + double(cum_deg) / N)
         << "   (should be ~1.0)\n";

    return 0;
}
