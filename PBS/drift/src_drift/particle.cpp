//* particle.cpp  [src_drift]
//
// Identical to the original src/particle.cpp with one addition:
//   stepDrift()  — Euler-Maruyama drift + diffusion step  // [DRIFT]
//
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

particle::particle(const particle &par)
    : x(par.x), y(par.y), z(par.z),
      gen(par.gen), gauss(par.gauss), uni(par.uni),
      state(par.state), id(par.id)
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

// [DRIFT] -----------------------------------------------------------------------
// Euler-Maruyama drift + diffusion step.
//
//   x(t+dt) = x(t) + vx*dt + sqrt(2*D*dt)*N(0,1)
//
// where s = sqrt(2*D*dt) is passed in (precomputed by channel for efficiency).
// No-op if particle is not FREE.
// -------------------------------------------------------------------------------
void particle::stepDrift(double vx, double vy, double vz, double s, double dt)
{
    if (state != ParticleState::FREE) return;
    x += vx * dt + s * gauss(gen);
    y += vy * dt + s * gauss(gen);
    z += vz * dt + s * gauss(gen);
}

// -------------------------------------------------------------------------
// Stochastic reactions
// -------------------------------------------------------------------------

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

ParticleState particle::tryBinding(double kf_eff, double kd, double dt)
{
    if (state != ParticleState::FREE) return state;
    double k_tot = kf_eff + kd;
    if (k_tot <= 0.0) return state;
    double p_event = 1.0 - std::exp(-k_tot * dt);
    if (uni(gen) < p_event) {
        if (uni(gen) < kf_eff / k_tot)
            state = ParticleState::BOUND;
        else
            state = ParticleState::DEGRADED;
    }
    return state;
}

ParticleState particle::tryBoundTransition(double kb, double ki, double kd, double dt)
{
    if (state != ParticleState::BOUND) return state;
    double k_tot = kb + ki + kd;
    if (k_tot <= 0.0) return state;
    double p_event = 1.0 - std::exp(-k_tot * dt);
    if (uni(gen) < p_event) {
        double u = uni(gen);
        if (u < kb / k_tot)
            state = ParticleState::FREE;
        else if (u < (kb + ki) / k_tot)
            state = ParticleState::INTERNALIZED;
        else
            state = ParticleState::DEGRADED;
    }
    return state;
}
