//* test.cpp
// testscenario for creating DBMC channel and calculation of channel impulse response
//TODO create more examples, including Monte Carlo simulation, multicoreprocessing, bitseq transmission,...
//TODO better fileformat for saving
#include "src/channel.h"
#include <iostream>
#include <fstream>

using namespace molcompbs;
using namespace std;

int main() {
    //* scenario parameters
    // result file
    string fname = "example_CIR.csv";
    // channel parameters
    double diffc = 1e-11;           // diffusion coefficient in m^2/s
    double dist = 3e-6;             // distance between Tx and center of Rx in m
    double rad  = 1e-6;             // receiver radius in m
    bool transp = true;             // transparent receiver
    // transmission parameters
    int N = 1e4;                    // number of release particles for bit-1
    double T = 1e0;                 // symbol duration in s
    double dt = 1e-3;               // temporal step size in s
    int K = 1;                      // number of bits
    
    //* create scenario
    channel dbmc;
    // create bit sequence
    vector<bool> bits = {1};        // channel impulse response
    // set parameters
    dbmc.setBit(bits);
    dbmc.setD(diffc);
    dbmc.setdt(dt);
    dbmc.setN(N);
    dbmc.setT(T);
    dbmc.setTX(0., 0., 0.);
    dbmc.setRX(dist, 0., 0., rad, transp);

    //* simulate scenario
    cout << "Running simulation ... ";
    vector<int> Y = dbmc.simulate(); // received number of molecules
    vector<double> y;
    // calculate probability function
    for (int n : Y)
        transp ? y.emplace_back(1.*n / N) : y.emplace_back(1.*n / N / dt);
    cout << "[DONE]" << endl;

    //* theoretical result
    // create time vector
    vector<double> tvec;
    double ttmp = 0;
    while (ttmp <= K*T)
    {
        tvec.emplace_back(ttmp);
        ttmp += dt;
    }
    // create theoretical channel impulse response
    vector<double> cir;
    for (double t : tvec)
    {
        if (t <= 0)
            cir.emplace_back(0.);
        else
            transp ? cir.emplace_back(4./3*M_PI*pow(rad, 3)/pow(4*M_PI*diffc*t, 3./2) * exp(-pow(dist, 2.)/(4*diffc*t)))
                   : cir.emplace_back(rad/dist*(dist-rad)/sqrt(4*M_PI*diffc*pow(t, 3)) * exp(-pow(dist-rad, 2.)/(4*diffc*t)));
    }

    //* save results
    cout << "Saving results ... ";
    // openfile
    ofstream resfile(fname);
    if (!resfile.is_open())
    {
        cerr << "Error opening file for writing results!" << endl;
        return 1;
    }
    // write header
    resfile << "t,y,Y,h" << endl;
    // write data
    for (int idx=0; idx<y.size(); idx++)
        resfile << tvec[idx] << "," << y[idx] << "," << Y[idx] << "," << cir[idx] << endl;
    // close file
    resfile.close();
    cout << "[DONE]" << endl;

    return 0;
}