//* channel.cpp
#include "channel.h"
#include <algorithm>

using namespace molcompbs;

channel::channel()
    : D(0), dt(1), T(0), Toff(0), stddev(0), stddevoff(0),
      N(0), kd_medium(0), n_degraded_medium(0)
{}

// -------------------------------------------------------------------------
// propagation()
// Diffuse all FREE particles, then attempt degradation for FREE particles
// outside the receiver sphere. Count and remove degraded particles.
// -------------------------------------------------------------------------
void channel::propagation()
{
    for (auto &p : particles)
    {
        p.step(stddev);

        // Degradation applies uniformly everywhere in the medium (Veletic
        // paper Eq. 5 — kd acts on all free particles regardless of position).
        if (p.isFree() && kd_medium > 0.0)
            p.tryDegradation(kd_medium, dt);
    }

    // Count free-medium degradations before erasing
    n_degraded_medium = 0;
    for (const auto &p : particles)
        if (p.state == ParticleState::DEGRADED) n_degraded_medium++;

    particles.erase(
        std::remove_if(particles.begin(), particles.end(),
            [](const particle &p){ return p.state == ParticleState::DEGRADED; }),
        particles.end());
}

// -------------------------------------------------------------------------
// random_bits()
// -------------------------------------------------------------------------
std::vector<bool> channel::random_bits(int K)
{
    std::random_device rd;
    std::mt19937 gen;
    #pragma omp critical
    {
        if (rd.entropy() > 0.)
            gen.seed(rd());
        else {
            auto ref  = std::chrono::high_resolution_clock::now().time_since_epoch().count();
            auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
            while (ref == seed)
                seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
            gen.seed(static_cast<unsigned>(seed));
        }
    }
    std::uniform_int_distribution<> dis(0, 1);
    std::vector<bool> bits;
    for (int i = 0; i < K; ++i)
        bits.emplace_back(static_cast<bool>(dis(gen)));
    return bits;
}

// -------------------------------------------------------------------------
// simulateDetailed()
// -------------------------------------------------------------------------
std::vector<SimStep> channel::simulateDetailed()
{
    int Nbit   = static_cast<int>(bitseq.size());
    int Nsteps = static_cast<int>(1.0 * Nbit * T / dt) + 1;

    std::vector<SimStep> result;
    result.reserve(Nsteps);

    std::vector<int> releaseSteps;
    for (int n = 0; n < Nbit; ++n)
        if (bitseq[n])
            releaseSteps.push_back(static_cast<int>(1.0 * n * T / dt));
    int relIdx = 0;

    RX.setKd(kd_medium);

    for (int n = 0; n < Nsteps; ++n)
    {
        if (relIdx < (int)releaseSteps.size() && n == releaseSteps[relIdx])
        {
            ++relIdx;
            if (stddevoff > 1e-20)
                TX.release(particles, N, stddevoff);
            else
                TX.release(particles, N);
        }

        propagation();
        RX.process(particles, D, dt);

        SimStep s;
        s.t               = n * dt;
        s.N_internalized  = RX.getNumInternalized();
        s.N_free_inside   = RX.getNumFreeInside();
        s.N_bound         = RX.getNumBound();
        s.N_degraded_rx   = RX.getNumDegraded();       // inside receiver
        s.N_degraded_med  = n_degraded_medium;          // free medium
        s.N_active        = static_cast<int>(particles.size());
        result.push_back(s);
    }

    return result;
}

// -------------------------------------------------------------------------
// simulate()  — legacy interface
// -------------------------------------------------------------------------
std::vector<int> channel::simulate()
{
    auto detailed = simulateDetailed();
    std::vector<int> Y;
    Y.reserve(detailed.size());
    ReceiverMode mode = RX.getMode();
    for (const auto &s : detailed)
    {
        if (mode == ReceiverMode::TRANSPARENT)
            Y.push_back(s.N_free_inside);
        else
            Y.push_back(s.N_internalized);
    }
    return Y;
}
