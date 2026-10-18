//* channel.cpp  [src_drift]
//
// Extends the original channel.cpp with:
//   - Uniform drift (Euler-Maruyama)                // [DRIFT]
//   - Reflective cylindrical boundary               // [CYLINDER]
//
// Modified / new lines are tagged accordingly.
//
#include "channel.h"
#include <algorithm>
#include <chrono>

using namespace molcompbs;

// -------------------------------------------------------------------------
// Constructor
// -------------------------------------------------------------------------
channel::channel()
    : D(0), dt(1), T(0), Toff(0), stddev(0), stddevoff(0),
      kd_medium(0), N(0), n_degraded_medium(0), n_exited(0),
      vx(0), vy(0), vz(0)
{}

// -------------------------------------------------------------------------
// propagation()
//
// Per-timestep order:
//   1. Reset cylinder step counter and n_degraded_medium        [CYLINDER]
//   2. Move each FREE particle (drift+diffusion or diffusion)   [DRIFT]
//   3. Apply cylinder boundary (reflect / mark axial exits)     [CYLINDER]
//   4. Chemical degradation for FREE particles (counted directly)
//   5. Erase all DEGRADED particles
// -------------------------------------------------------------------------
void channel::propagation()
{
    bool has_drift = (vx != 0.0 || vy != 0.0 || vz != 0.0);  // [DRIFT]

    cyl.resetStep();                // [CYLINDER]
    n_degraded_medium = 0;          // reset per-step chemical degradation count

    for (auto &p : particles)
    {
        // 2 — move
        if (has_drift)
            p.stepDrift(vx, vy, vz, stddev, dt);   // [DRIFT]
        else
            p.step(stddev);

        // 3 — boundary: axial exits are marked DEGRADED and counted inside cyl
        cyl.apply(p);   // [CYLINDER]

        // 4 — chemical degradation (only FREE particles not already exited).
        //     Count degradations directly; no reliance on later arithmetic.
        if (p.isFree() && kd_medium > 0.0) {
            if (p.tryDegradation(kd_medium, dt))
                ++n_degraded_medium;
        }
    }

    // Cylinder exit count is exact and independent of the erase step below.
    n_exited = cyl.getExited();   // [CYLINDER]

    // 5 — erase DEGRADED particles
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
    RX.resetAll();               // start each simulation with fresh counters

    for (int n = 0; n < Nsteps; ++n)
    {
        // Release particles for this bit at the start of the timestep
        if (relIdx < (int)releaseSteps.size() && n == releaseSteps[relIdx])
        {
            ++relIdx;
            if (stddevoff > 1e-20)
                TX.release(particles, N, stddevoff);
            else
                TX.release(particles, N);
            for (auto &p : particles)                     // [VIZ]
                if (p.id < 0) p.id = next_id++;
        }

        if (stepHook) stepHook(n, (n + 1) * dt, 0, particles);   // [VIZ]

        // Propagate and detect: the state below reflects the system AFTER
        // one propagation step, so the physical time is (n+1)*dt.
        propagation();
        if (stepHook) stepHook(n, (n + 1) * dt, 1, particles);   // [VIZ]
        RX.process(particles, D, dt);
        if (stepHook) stepHook(n, (n + 1) * dt, 2, particles);   // [VIZ]

        // Record step — timestamp is the post-step physical time, matching
        // the state of particles / counters that we store.
        SimStep s;
        s.t              = (n + 1) * dt;
        s.N_internalized = RX.getNumInternalized();
        s.N_free_inside  = RX.getNumFreeInside();
        s.N_contact      = RX.getNumContact();
        s.N_bound        = RX.getNumBound();
        s.N_degraded_rx  = RX.getNumDegraded();
        s.N_degraded_med = n_degraded_medium;
        s.N_active       = static_cast<int>(particles.size());
        s.N_exited       = n_exited;                // [CYLINDER]
        result.push_back(s);
    }

    return result;
}

// -------------------------------------------------------------------------
// simulate() — legacy interface
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
