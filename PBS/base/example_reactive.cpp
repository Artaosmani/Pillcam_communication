//* example_reactive.cpp
//
// Kinetic model / reference parameters: H. Khoshfekr Rudsari, M. Damrath,
// M. Zoofaghari, I. Balasingham, and M. Veletic, "An Intercellular Communication
// System for Intra-Body Communication Networks," IEEE Trans. Commun., vol. 73,
// no. 2, pp. 817-831, Feb. 2025. doi:10.1109/TCOMM.2024.3440877
// Tracks Pe, Pf, Pi, Pd and validates Pe + Pf + Pi + Pd = 1.
//
// Compile:
//   g++ -O3 -fopenmp example_reactive.cpp src/particle.cpp src/channel.cpp -o example_reactive
//
// Run:
//   ./example_reactive
//
#include "src/channel.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

using namespace molcompbs;
using namespace std;

int main()
{
    //* -----------------------------------------------------------------------
    //* Parameters  (Table I of paper)
    //* -----------------------------------------------------------------------
    const string fname = "example_reactive.csv";

    const double D   = 5e-9;       // diffusion coefficient [m^2/s]
    const double r0  = 1e-6;       // release distance [m]
    const double a   = 0.5e-6;     // receiver radius [m]
    const double kd  = 2e3;        // degradation rate [s^-1]
    const double kf  = 1e-14;   // forward binding rate [particle^-1 m^3 s^-1]
    const double kb  = 2e3;        // backward rate [s^-1]
    const double ki  = 1e4;        // internalization rate [s^-1]

// const double kf  = 3.14e-14 * 1e6;  // effectively infinite  →  3.14e-8
// const double kb  = 0.0;              // exactly zero, this one is safe
// const double ki  = 1e4 * 1e6;       // effectively infinite  →  1e10

    const double dt   = 5e-9;      // timestep [s]
    const int    Npbs = 500;      // particles per realisation
    const int    Rpbs = 100;      // Monte-Carlo realisations
    const double tau  = 1e-3;      // observation window [s]

    const int Nsteps = static_cast<int>(tau / dt) + 1;

    //* -----------------------------------------------------------------------
    //* Accumulators
    //* -----------------------------------------------------------------------
    // Pe[n]  = FREE anywhere        = (N_active - N_bound) / (R*N)
    // Pf[n]  = BOUND                = N_bound / (R*N)
    // Pi[n]  = cumulative INTERNALIZED / (R*N)
    // Pd[n]  = cumulative DEGRADED (rx + medium) / (R*N)
    // Pabs[n]= cumulative absorbed (perfectly absorbing Rx) / (R*N)

    vector<double> cnt_active(Nsteps, 0.0);
    vector<double> cnt_bound(Nsteps, 0.0);
    vector<double> cnt_int_cum(Nsteps, 0.0);
    vector<double> cnt_deg_cum(Nsteps, 0.0);   // rx + medium combined
    vector<double> cnt_abs_cum(Nsteps, 0.0);

    cout << "Running " << Rpbs << " realisations ("
         << Npbs << " particles each) ..." << flush;

    for (int r = 0; r < Rpbs; ++r)
    {
        // --- Reactive channel ---
        channel ch;
        ch.setD(D);  ch.setdt(dt);  ch.setT(tau);
        ch.setToff(0.0);  ch.setN(Npbs);  ch.setBit({true});
        ch.setTX(0.0, 0.0, 0.0);
        ch.setRXReactive(r0, 0.0, 0.0, a, kf, kb, ki);
        ch.setKd(kd);

        vector<SimStep> steps = ch.simulateDetailed();

        int cumInt = 0, cumDeg = 0;
        for (int n = 0; n < (int)steps.size() && n < Nsteps; ++n)
        {
            const SimStep &s = steps[n];
            cumInt += s.N_internalized;
            cumDeg += s.N_degraded_rx + s.N_degraded_med;  // both sources
            cnt_active[n]  += s.N_active;
            cnt_bound[n]   += s.N_bound;
            cnt_int_cum[n] += cumInt;
            cnt_deg_cum[n] += cumDeg;
        }

        // --- Absorbing channel (Pabs upper bound) ---
        channel ch_abs;
        ch_abs.setD(D);  ch_abs.setdt(dt);  ch_abs.setT(tau);
        ch_abs.setToff(0.0);  ch_abs.setN(Npbs);  ch_abs.setBit({true});
        ch_abs.setTX(0.0, 0.0, 0.0);
        ch_abs.setRX(r0, 0.0, 0.0, a, false);
        ch_abs.setKd(kd);

        vector<SimStep> steps_abs = ch_abs.simulateDetailed();

        int cumAbs = 0;
        for (int n = 0; n < (int)steps_abs.size() && n < Nsteps; ++n)
        {
            cumAbs += steps_abs[n].N_internalized;
            cnt_abs_cum[n] += cumAbs;
        }
    }
    cout << " [DONE]" << endl;

    //* -----------------------------------------------------------------------
    //* Normalise
    //* -----------------------------------------------------------------------
    double norm = static_cast<double>(Rpbs) * Npbs;

    vector<double> Pe(Nsteps), Pf(Nsteps), Pi(Nsteps), Pd(Nsteps);
    vector<double> Psum(Nsteps), Pabs(Nsteps);
    for (int n = 0; n < Nsteps; ++n)
    {
        double active = cnt_active[n] / norm;
        Pf[n]   = cnt_bound[n]   / norm;
        Pe[n]   = active - Pf[n];
        Pi[n]   = cnt_int_cum[n] / norm;
        Pd[n]   = cnt_deg_cum[n] / norm;
        Psum[n] = Pe[n] + Pf[n] + Pi[n] + Pd[n];
        Pabs[n] = cnt_abs_cum[n] / norm;
    }

    //* -----------------------------------------------------------------------
    //* Validation
    //* -----------------------------------------------------------------------
    cout << "\nValidation: Pe + Pf + Pi + Pd should equal 1.0" << endl;
    cout << "  t=0         sum = " << Psum[0]         << endl;
    cout << "  t=tau/4     sum = " << Psum[Nsteps/4]  << endl;
    cout << "  t=tau/2     sum = " << Psum[Nsteps/2]  << endl;
    cout << "  t=tau       sum = " << Psum[Nsteps-1]  << endl;
    cout << "  Pi(tau)="    << Pi[Nsteps-1]
         << "  Pabs(tau)=" << Pabs[Nsteps-1]
         << "  (Pi <= Pabs always)" << endl;

    //* -----------------------------------------------------------------------
    //* Save
    //* -----------------------------------------------------------------------
    cout << "\nSaving to " << fname << " ... " << flush;

    ofstream f(fname);
    if (!f.is_open()) {
        cerr << "Error: could not open " << fname << endl;
        return 1;
    }

    // Columns:
    //   t     : time [s]
    //   Pe    : P(diffusing)
    //   Pf    : P(bound)
    //   Pi    : P(internalized/detected)
    //   Pd    : P(degraded)            <- now includes free-medium degradation
    //   Psum  : Pe+Pf+Pi+Pd            <- should equal 1
    //   Pabs  : P(absorbed, perf. Rx)  <- upper bound for Pi
    f << "t,Pe,Pf,Pi,Pd,Psum,Pabs" << endl;
    for (int n = 0; n < Nsteps; ++n)
        f << (n * dt)  << ","
          << Pe[n]     << "," << Pf[n]   << ","
          << Pi[n]     << "," << Pd[n]   << ","
          << Psum[n]   << "," << Pabs[n] << "\n";

    f.close();
    cout << "[DONE]" << endl;

    return 0;
}
