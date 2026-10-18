//* particle.h
#pragma once

#include <random>
#include <iostream>
#include <chrono>
#include <cmath>

namespace molcompbs
{
  // -------------------------------------------------------------------------
  // Particle state machine
  //   FREE        : diffusing freely in the medium
  //   BOUND       : reversibly bound to a receptor on the Rx surface
  //   INTERNALIZED: irreversibly detected by the Rx  (terminal)
  //   DEGRADED    : chemically degraded in the medium (terminal)
  // -------------------------------------------------------------------------
  enum class ParticleState { FREE, BOUND, INTERNALIZED, DEGRADED };

  class particle
  {
    private:
      // position
      double x, y, z;
      // RNG
      std::mt19937 gen;
      std::normal_distribution<double> gauss; // N(0,1)
      std::uniform_real_distribution<double> uni; // U(0,1)
      // helpers
      void setVar(double &var, double in) { var = in; }
      double getVar(double &var) { return var; }
      void setRandomSeed();

    public:
      // state (public so Receiver can update it without a maze of friends)
      ParticleState state;

      // constructors / destructor
      particle();
      particle(double x0, double y0, double z0);
      particle(const particle &par);
      ~particle() {}

      // position
      void setX(double xin);
      void setY(double yin);
      void setZ(double zin);
      void setPos(double xin, double yin, double zin);
      double getX() const;
      double getY() const;
      double getZ() const;
      void printPos();

      // random walk  (no-op if not FREE)
      void step(double s);
      void step(double sx, double sy, double sz);

      // -----------------------------------------------------------------------
      // Stochastic reactions (called once per timestep per particle)
      // -----------------------------------------------------------------------

      // For a FREE particle anywhere in the medium:
      //   Competing reactions: degradation (kd) vs. nothing else outside Rx.
      //   Returns true if the particle degraded this step.
      bool tryDegradation(double kd, double dt);

      // For a FREE particle that is geometrically inside the Rx sphere:
      //   Competing reactions: binding (kf_eff, first-order effective rate)
      //   vs. degradation (kd).
      //   Updates state to BOUND or DEGRADED if event fires.
      //   Returns new state.
      ParticleState tryBinding(double kf_eff, double kd, double dt);

      // For a BOUND particle:
      //   Competing reactions: unbinding (kb), internalization (ki),
      //   degradation (kd).
      //   Updates state accordingly.
      //   Returns new state.
      ParticleState tryBoundTransition(double kb, double ki, double kd, double dt);

      // draw a single U(0,1) sample (used by Receiver for binding decisions)
      double drawUniform() { return uni(gen); }

      // After a failed binding attempt or unbinding, place the particle just
      // outside the sphere surface using the one-sided exit distribution
      // (Al-Zu'bi & Mohan 2018, Eq. 46-47). This prevents immediate re-entry
      // on the next diffusion step, which would cause over-counting of Pi.
      //
      //   displacement = sqrt(2*D*dt) * (0.571825*a - 0.552246*a^2)
      //                              / (1 - 1.53908*a + 0.546424*a^2)
      // where a ~ U(0,1).  Result is placed radially outward from (cx,cy,cz).
      void stepOutside(double D, double dt,
                       double cx, double cy, double cz, double r)
      {
          double am = uni(gen);
          double stdd = std::sqrt(2.0 * D * dt);
          double denom = 1.0 - 1.53908*am + 0.546424*am*am;
          double d_out = (std::abs(denom) > 1e-30)
              ? stdd * (0.571825*am - 0.552246*am*am) / denom
              : 0.0;
          if (d_out < 0.0) d_out = 0.0;

          double nx = x - cx, ny = y - cy, nz = z - cz;
          double nm = std::sqrt(nx*nx + ny*ny + nz*nz);
          if (nm < 1e-30) { nx=1.0; ny=0.0; nz=0.0; nm=1.0; }
          nx /= nm; ny /= nm; nz /= nm;

          x = cx + (r + d_out) * nx;
          y = cy + (r + d_out) * ny;
          z = cz + (r + d_out) * nz;
      }

      // convenience
      bool isActive() const { return state == ParticleState::FREE || state == ParticleState::BOUND; }
      bool isFree()   const { return state == ParticleState::FREE; }
      bool isBound()  const { return state == ParticleState::BOUND; }
  };

} // namespace molcompbs
