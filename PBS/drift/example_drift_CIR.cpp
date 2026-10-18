//* example_drift_CIR.cpp
//
// Channel impulse response with drift.
// Compares the PBS simulation against the analytical closed-form solution
// for a 3-D diffusion channel with uniform drift velocity v along the x-axis.
//
// The analytical CIR for a TRANSPARENT receiver of radius a, Tx-Rx distance d,
// diffusion coefficient D, and drift velocity v is (Gaussian-kernel convolution
// of the point-source Green's function with the sphere volume):
//
//   h(t) = (4/3 * pi * a^3) / (4*pi*D*t)^(3/2)
//          * exp( -(d - v*t)^2 / (4*D*t) )
//
// i.e., the peak shifts from d/sqrt(... ) toward shorter times as v increases.
//
// Compile (from the repo root):
//   g++ -O3 -fopenmp example_drift_CIR.cpp src_drift/particle.cpp src_drift/channel.cpp -o example_drift_CIR
//
// Run:
//   ./example_drift_CIR
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
    //* Scenario parameters
    //* -----------------------------------------------------------------------
    const string fname = "example_drift_CIR.csv";

    // channel
    const double D    = 1e-11;   // diffusion coefficient [m^2/s]
    const double dist = 3e-6;    // Tx-Rx centre distance  [m]
    const double rad  = 1e-6;    // receiver radius         [m]
    const bool   transp = true;  // transparent receiver

    // [DRIFT] drift velocity along x-axis [m/s]
    // Try v = 0 to reproduce the original example_CIR result exactly.
    const double v = 2e-6;       // e.g. 2 µm/s  →  noticeable peak shift

    // transmission
    const int    N  = 1e4;       // particles per bit-1
    const double T  = 1e0;       // symbol duration [s]
    const double dt = 1e-3;      // timestep [s]
    const int    K  = 1;         // bits (CIR: single impulse)

    //* -----------------------------------------------------------------------
    //* Create and configure channel
    //* -----------------------------------------------------------------------
    channel ch;
    ch.setBit({true});           // single impulse
    ch.setD(D);
    ch.setdt(dt);
    ch.setN(N);
    ch.setT(T);
    ch.setTX(0., 0., 0.);
    ch.setRX(dist, 0., 0., rad, transp);

    // [DRIFT] set drift — remove or set to (0,0,0) for pure diffusion
    ch.setV(v, 0., 0.);

    //* -----------------------------------------------------------------------
    //* Simulate
    //* -----------------------------------------------------------------------
    cout << "Running simulation (drift v = " << v << " m/s) ... " << flush;
    vector<int>    Y  = ch.simulate();
    vector<double> y;
    for (int n : Y)
        y.emplace_back(1.0 * n / N);   // normalised hit probability
    cout << "[DONE]" << endl;

    //* -----------------------------------------------------------------------
    //* Analytical CIR with drift (transparent sphere, Gaussian Green's function)
    //*
    //* NOTE: after channel::simulateDetailed()'s time-axis fix, Y[i] is the
    //* PBS count at physical time (i+1)*dt. We therefore build the analytical
    //* time grid starting at t = dt (matching Y[0]) so the two curves line up
    //* in the output CSV without an off-by-one shift.
    //* -----------------------------------------------------------------------
    vector<double> tvec;
    for (size_t i = 0; i < Y.size(); ++i)
        tvec.emplace_back((i + 1) * dt);

    vector<double> cir;
    for (double t : tvec)
    {
        if (t <= 0.0) {
            cir.emplace_back(0.0);
        } else {
            // Shifted distance: effective separation decreases as drift
            // carries particles toward the receiver.
            double shifted = dist - v * t;
            double val = (4.0/3.0 * M_PI * pow(rad, 3))
                       / pow(4.0 * M_PI * D * t, 1.5)
                       * exp(-shifted * shifted / (4.0 * D * t));
            cir.emplace_back(val);
        }
    }

    //* -----------------------------------------------------------------------
    //* Save results
    //* -----------------------------------------------------------------------
    cout << "Saving to " << fname << " ... " << flush;
    ofstream f(fname);
    if (!f.is_open()) {
        cerr << "Error opening output file!" << endl;
        return 1;
    }
    f << "t,y,Y,h" << endl;
    for (size_t i = 0; i < y.size(); ++i)
        f << tvec[i] << "," << y[i] << "," << Y[i] << "," << cir[i] << "\n";
    f.close();
    cout << "[DONE]" << endl;

    return 0;
}
