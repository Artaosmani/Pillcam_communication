//* example_drift_cylinder_CIR.cpp
//
// Channel impulse response in a cylindrical blood-vessel environment
// with axial drift (Poiseuille-like plug flow approximation).
//
// Setup:
//   - Cylinder radius R_vessel along the x-axis
//   - Tx at origin, Rx (transparent sphere) at distance d along x
//   - Uniform drift velocity v along x (set v=0 for pure diffusion)
//   - Reflective cylindrical wall, open ends (particles exiting are removed)
//
// The CSV output contains columns:
//   t         : time [s]
//   y         : normalised in-sphere count (PBS)  (= N_free_inside / N)
//   Y         : raw in-sphere count (PBS)         (= N_free_inside)
//   N_contact : contacts with the sphere this step (== Y in TRANSPARENT mode)
//   N_active  : particles still alive (in vessel, not yet exited / degraded)
//   N_exited  : particles lost through open ends this step
//
// Note: no closed-form analytical CIR exists for the bounded cylinder case,
// so only PBS results are saved.  To verify correctness, set R_vessel very
// large (e.g. 1e-3 m) — the result should converge to the free-space CIR.
//
// Compile (from repo root):
//   g++ -O3 -fopenmp example_drift_cylinder_CIR.cpp \
//       src_drift/particle.cpp src_drift/channel.cpp \
//       -o example_drift_cylinder_CIR
//
// Run:
//   ./example_drift_cylinder_CIR
//
#include "src_drift/channel.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <string>

using namespace molcompbs;
using namespace std;

int main()
{
    //* -----------------------------------------------------------------------
    //* Parameters
    //* -----------------------------------------------------------------------
    const string fname = "example_drift_cylinder_CIR.csv";

    // Channel / medium
    const double D        = 1e-9;    // diffusion coefficient [m^2/s]
    const double dist     = 20e-6;   // Tx-Rx distance [m]  (20 µm)
    const double rad      = 5e-6;    // receiver radius [m]  ( 5 µm)
    const bool   transp   = true;    // transparent receiver

    // [DRIFT] Blood-flow drift along x
    const double v        = 500e-6;  // drift velocity [m/s]  (500 µm/s, capillary range)

    // [CYLINDER] Blood vessel
    const double R_vessel = 15e-6;   // vessel radius [m]  (15 µm — small capillary)
    const double x_min    = -5e-6;   // left open end  (allow small back-diffusion)
    const double x_max    = 60e-6;   // right open end

    // Transmission
    //
    // T must be long enough for the pulse to reach the receiver.
    // With v = 500 µm/s and d = 20 µm the drift-arrival time is ~40 ms,
    // and the diffusion-only time is d^2/(2D) ~ 200 ms, so a 100 ms
    // window captures the peak and a substantial part of the tail.
    //
    // dt = 1e-5 s gives stddev = sqrt(2 D dt) ≈ 141 nm, i.e. ~1 % of
    // R_vessel — safely inside the "small step vs. boundary" regime.
    const int    N  = 5000;          // particles per impulse
    const double T  = 1e-1;          // observation window [s]  (100 ms)
    const double dt = 1e-5;          // timestep [s]            (10 µs)

    //* -----------------------------------------------------------------------
    //* Sanity: print stddev vs R_vessel
    //* -----------------------------------------------------------------------
    double stddev = sqrt(2.0 * D * dt);
    cout << "stddev = " << stddev * 1e9 << " nm,  R_vessel = "
         << R_vessel * 1e6 << " µm,  ratio = "
         << stddev / R_vessel << endl;
    if (stddev > 0.1 * R_vessel)
        cout << "WARNING: consider reducing dt for accurate reflection.\n";

    //* -----------------------------------------------------------------------
    //* Build channel
    //* -----------------------------------------------------------------------
    channel ch;
    ch.setBit({true});
    ch.setD(D);
    ch.setdt(dt);
    ch.setN(N);
    ch.setT(T);
    ch.setTX(0., 0., 0.);
    ch.setRX(dist, 0., 0., rad, transp);

    ch.setV(v, 0., 0.);                          // [DRIFT]
    ch.setCylinder(R_vessel, x_min, x_max);      // [CYLINDER]

    //* -----------------------------------------------------------------------
    //* Simulate
    //* -----------------------------------------------------------------------
    cout << "Running simulation ..." << flush;
    vector<SimStep> steps = ch.simulateDetailed();
    cout << " [DONE]" << endl;

    //* -----------------------------------------------------------------------
    //* Save
    //* -----------------------------------------------------------------------
    cout << "Saving to " << fname << " ..." << flush;
    ofstream f(fname);
    if (!f.is_open()) {
        cerr << "Error opening output file!" << endl;
        return 1;
    }

    f << "t,y,Y,N_contact,N_active,N_exited\n";
    for (const auto &s : steps)
    {
        double y_norm = static_cast<double>(s.N_free_inside) / N;
        f << s.t            << ","
          << y_norm          << ","
          << s.N_free_inside << ","
          << s.N_contact     << ","
          << s.N_active      << ","
          << s.N_exited      << "\n";
    }
    f.close();
    cout << " [DONE]" << endl;

    //* -----------------------------------------------------------------------
    //* Quick summary
    //*
    //* In TRANSPARENT mode the receiver does not remove particles, so the
    //* meaningful per-run conservation is
    //*
    //*     N  =  N_active_final  +  N_exited_total  +  N_degraded_medium_total
    //*
    //* (N_degraded_medium_total == 0 here, since kd_medium is not set.)
    //*
    //* A useful signal metric is the peak instantaneous in-sphere count, i.e.
    //* the peak of Y; summing Y over time would double-count every particle
    //* for every step it spends inside the sphere and is NOT a "hit count".
    //* -----------------------------------------------------------------------
    int    peak_hit     = 0;
    double peak_time    = 0.0;
    int    total_exited = 0;
    int    total_deg    = 0;
    for (const auto &s : steps) {
        if (s.N_free_inside > peak_hit) {
            peak_hit  = s.N_free_inside;
            peak_time = s.t;
        }
        total_exited += s.N_exited;
        total_deg    += s.N_degraded_med;
    }
    int remaining  = steps.empty() ? 0 : steps.back().N_active;
    int accounted  = remaining + total_exited + total_deg;

    cout << "\nSummary (" << N << " particles released):\n"
         << "  Peak in-sphere : " << peak_hit
         << "  (" << 100.0 * peak_hit    / N << " %)"
         << "  at t = " << peak_time << " s\n"
         << "  Exited vessel  : " << total_exited
         << "  (" << 100.0 * total_exited / N << " %)\n"
         << "  Degraded (med) : " << total_deg
         << "  (" << 100.0 * total_deg    / N << " %)\n"
         << "  Still active   : " << remaining
         << "  (" << 100.0 * remaining   / N << " %)\n"
         << "  Accounted for  : " << 100.0 * accounted / N << " %"
         << "   (expected 100 %)\n";

    return 0;
}
