//* particle.cpp
#include "particle.h"

using namespace molcompbs;

// -------------------------------------------------------------------------
// RNG seed
// -------------------------------------------------------------------------
void particle::setRandomSeed()
{
    std::random_device rd;
    #pragma omp critical
    {
        if (rd.entropy() > 0.0) {
            gen.seed(rd());
        } else {
            // Clock-racing fallback: spin until the clock advances
            auto ref  = std::chrono::high_resolution_clock::now().time_since_epoch().count();
            auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
            while (ref == seed)
                seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
            gen.seed(static_cast<unsigned>(seed));
        }
    }
    gauss = std::normal_distribution<double>(0.0, 1.0);
    uni   = std::uniform_real_distribution<double>(0.0, 1.0);
}

// -------------------------------------------------------------------------
// Constructors
// -------------------------------------------------------------------------
particle::particle()
    : x(0), y(0), z(0), state(ParticleState::FREE)
{ setRandomSeed(); }

particle::particle(double x0, double y0, double z0)
    : x(x0), y(y0), z(z0), state(ParticleState::FREE)
{ setRandomSeed(); }

// Copy constructor: copy position, state, and RNG state
particle::particle(const particle &par)
    : x(par.x), y(par.y), z(par.z),
      gen(par.gen), gauss(par.gauss), uni(par.uni),
      state(par.state)
{}

// -------------------------------------------------------------------------
// Position
// -------------------------------------------------------------------------
void particle::setX(double xin) { setVar(x, xin); }
void particle::setY(double yin) { setVar(y, yin); }
void particle::setZ(double zin) { setVar(z, zin); }

void particle::setPos(double xin, double yin, double zin) {
    setX(xin); setY(yin); setZ(zin);
}

double particle::getX() const { return x; }
double particle::getY() const { return y; }
double particle::getZ() const { return z; }

void particle::printPos() {
    std::cout << "Particle [" << x << ", " << y << ", " << z << "]"
              << "  state=";
    switch (state) {
        case ParticleState::FREE:         std::cout << "FREE";         break;
        case ParticleState::BOUND:        std::cout << "BOUND";        break;
        case ParticleState::INTERNALIZED: std::cout << "INTERNALIZED"; break;
        case ParticleState::DEGRADED:     std::cout << "DEGRADED";     break;
    }
    std::cout << std::endl;
}

// -------------------------------------------------------------------------
// Random walk  (only moves FREE particles)
// -------------------------------------------------------------------------
void particle::step(double s) { step(s, s, s); }

void particle::step(double sx, double sy, double sz)
{
    if (state != ParticleState::FREE) return;
    x += sx * gauss(gen);
    y += sy * gauss(gen);
    z += sz * gauss(gen);
}

// -------------------------------------------------------------------------
// Stochastic reactions
// -------------------------------------------------------------------------

// Degradation in free medium (only for FREE particles).
// Uses exact Poisson-process probability: P(degrade) = 1 - exp(-kd*dt).
bool particle::tryDegradation(double kd, double dt)
{
    if (state != ParticleState::FREE) return false;
    if (kd <= 0.0) return false;

    double p_degrade = 1.0 - std::exp(-kd * dt);
    if (uni(gen) < p_degrade) {
        state = ParticleState::DEGRADED;
        return true;
    }
    return false;
}

// Binding attempt for a FREE particle inside the Rx sphere.
//
// Competing Poisson processes: binding (kf_eff) vs. degradation (kd).
//
//   Total rate  k_tot = kf_eff + kd
//   P(any event in dt) = 1 - exp(-k_tot * dt)
//   Given an event fires, P(bind) = kf_eff / k_tot
//                          P(degrade) = kd / k_tot
//
// kf_eff is the *effective first-order* binding rate, which the Receiver
// computes from the second-order kf and the surface geometry (see
// Receiver::computeKfEff).
ParticleState particle::tryBinding(double kf_eff, double kd, double dt)
{
    if (state != ParticleState::FREE) return state;

    double k_tot = kf_eff + kd;
    if (k_tot <= 0.0) return state;

    double p_event = 1.0 - std::exp(-k_tot * dt);
    if (uni(gen) < p_event) {
        // An event fires — decide which one
        if (uni(gen) < kf_eff / k_tot) {
            state = ParticleState::BOUND;
        } else {
            state = ParticleState::DEGRADED;
        }
    }
    return state;
}

// Transition for a BOUND particle.
//
// Competing Poisson processes: unbinding (kb), internalization (ki),
// degradation (kd).
//
//   k_tot = kb + ki + kd
//   P(any event in dt) = 1 - exp(-k_tot * dt)
//   Given event: P(unbind)       = kb / k_tot
//                P(internalize)  = ki / k_tot
//                P(degrade)      = kd / k_tot
ParticleState particle::tryBoundTransition(double kb, double ki, double kd, double dt)
{
    if (state != ParticleState::BOUND) return state;

    double k_tot = kb + ki + kd;
    if (k_tot <= 0.0) return state;

    double p_event = 1.0 - std::exp(-k_tot * dt);
    if (uni(gen) < p_event) {
        double u = uni(gen);
        if (u < kb / k_tot) {
            state = ParticleState::FREE;       // unbound, back to diffusion
        } else if (u < (kb + ki) / k_tot) {
            state = ParticleState::INTERNALIZED; // detected
        } else {
            state = ParticleState::DEGRADED;
        }
    }
    return state;
}
