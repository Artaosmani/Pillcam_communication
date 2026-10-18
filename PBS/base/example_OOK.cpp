//* test.cpp
// testscenario for creating DBMC channel and calculation simulate OOK transmission
//TODO better fileformat for saving
#include "src/channel.h"
#include <iostream>
#include <fstream>
#include <omp.h>

using namespace molcompbs;
using namespace std;

int main() {
    //* scenario parameters
    // result file
    string fname = "example_OOK.csv";
    // channel parameters
    double diffc = 1e-11;           // diffusion coefficient in m^2/s
    double dist = 3e-6;             // distance between Tx and center of Rx in m
    double rad  = 1e-6;             // receiver radius in m
    bool transp = true;             // transparent receiver
    // transmission parameters
    int N = 5e3;                    // number of release particles for bit-1
    int R = 1e6;                    // Monte-Carlo runs
    double T = 5e-1;                // symbol duration in s
    double dt = T;                  // temporal step size in s
    double Toff = 0.15;             // offset time in s
    int K = 1e2;                    // number of bits
    

    //* simulate scenario
    cout << "Running simulation ... ";
    
    // create time vector
    vector<double> tvec;
    double ttmp = Toff;
    while (ttmp <= K*T+Toff)
    {
        tvec.emplace_back(ttmp);
        ttmp += dt;
    }
    
    // openfile
    ofstream resfile(fname);
    if (!resfile.is_open())
    {
        cerr << "Error opening file for writing results!" << endl;
        return 1;
    }
    
    // write time
    resfile << "t"; // header
    for (int idx=0; idx<tvec.size(); idx++)
        resfile << "," << tvec[idx];
    resfile << endl;

    // Monte-Carlo loop
    cout << omp_get_max_threads();
    //! parallel processing, comment the following two lines to avoid parallel processing, or set number of threads individually
    omp_set_num_threads(omp_get_max_threads());
    #pragma omp parallel for shared(resfile, K, diffc, dt, N, T, Toff, dist, rad, transp)
    for (int r = 0; r < R; r++)
    {
        // create channel
        channel dbmc;
        // set parameters
        dbmc.setD(diffc);
        dbmc.setdt(dt);
        dbmc.setN(N);
        dbmc.setT(T);
        dbmc.setToff(Toff);
        dbmc.setTX(0., 0., 0.);
        dbmc.setRX(dist, 0., 0., rad, transp);

        // create bit sequence
        vector<bool> bits = dbmc.random_bits(K);
        dbmc.setBit(bits);

        // run PBS
        vector<int> Y = dbmc.simulate(); // received number of molecules

        #pragma omp critical
        {
            // write bits
            resfile << "b"; // header
            for (int idx = 0; idx < K; idx++)
                resfile << "," << bits[idx];
            resfile << endl;

            // write molecule counts
            resfile << "y"; // header
            for (int idx = 0; idx < Y.size(); idx++)
                resfile << "," << Y[idx];
            resfile << endl;
        }
    }

    // close file
    resfile.close();
    cout << "[DONE]" << endl;

    return 0;
}