//* cylinder.h  [src_drift]
//
// Reflective cylindrical boundary aligned with the x-axis.
//
// Geometry:
//   Axis   : x-axis
//   Radius : R_cyl  (radial wall — fully reflective)
//   Extent : [x_min, x_max] along x (open ends — particles outside are removed)
//
// Reflection algorithm (radial):
//   After a diffusion/drift step the radial distance rho = sqrt(y^2+z^2) is
//   checked. If rho > R_cyl the distance is folded back iteratively:
//       rho_new = 2*R_cyl - rho
//   and (y,z) are rescaled to lie at rho_new from the axis.
//   One fold covers the vast majority of physical cases; the loop guards
//   against pathologically large steps.
//
// Axial open ends:
//   A FREE particle with x < x_min or x > x_max is marked DEGRADED so that
//   channel::propagation() erases it. The cylinder tracks these exits in an
//   internal per-step counter (n_exited) that is reset at the start of each
//   propagation call and read back via getExited().
//
// Usage:
//   ch.setCylinder(R, x_min, x_max);   // enable
//   ch.setCylinder(R);                 // enable with auto x range
//   ch.setCylinder(0);                 // disable (default: no boundary)
//
#pragma once

#include <cmath>
#include <iostream>
#include "particle.h"

namespace molcompbs
{
  class Cylinder
  {
    private:
      double R;        // cylinder radius [m]
      double x_min;    // left  open end  [m]
      double x_max;    // right open end  [m]
      bool   active;   // false -> apply() is a no-op
      int    n_exited; // axial exits counted in current timestep

    public:
      // ------------------------------------------------------------------
      // Construction
      // ------------------------------------------------------------------
      Cylinder()
          : R(0.0), x_min(0.0), x_max(0.0), active(false), n_exited(0)
      {}

      Cylinder(double Rin, double x_min_in, double x_max_in)
          : R(Rin), x_min(x_min_in), x_max(x_max_in),
            active(Rin > 0.0), n_exited(0)
      {}

      // ------------------------------------------------------------------
      // Configuration
      // ------------------------------------------------------------------
      void set(double Rin, double x_min_in, double x_max_in)
      {
          R        = Rin;
          x_min    = x_min_in;
          x_max    = x_max_in;
          active   = (Rin > 0.0);
          n_exited = 0;
      }

      void   disable()        { active = false; }
      bool   isActive() const { return active; }
      double getR()     const { return R; }
      double getXmin()  const { return x_min; }
      double getXmax()  const { return x_max; }

      // ------------------------------------------------------------------
      // warnIfStepLarge()
      //   Prints a warning when stddev = sqrt(2*D*dt) > 10% of R.
      // ------------------------------------------------------------------
      void warnIfStepLarge(double stddev) const
      {
          if (!active) return;
          if (stddev > 0.1 * R)
              std::cerr << "[Cylinder] WARNING: diffusion stddev (" << stddev
                        << " m) > 0.1 * R (" << 0.1 * R
                        << " m). Consider reducing dt.\n";
      }

      // ------------------------------------------------------------------
      // resetStep() — call once at the start of each propagation()
      // getExited() — axial exits since last resetStep()
      // ------------------------------------------------------------------
      void resetStep()        { n_exited = 0; }
      int  getExited() const  { return n_exited; }

      // ------------------------------------------------------------------
      // apply() — call once per FREE particle AFTER its move step.
      //
      //   Axial exit  : marks particle DEGRADED, increments n_exited.
      //   Radial cross: reflects particle back inside cylinder.
      //   Returns true if the particle was axially removed.
      // ------------------------------------------------------------------
      bool apply(particle &p)
      {
          if (!active)     return false;
          if (!p.isFree()) return false;   // BOUND particles stay at Rx surface

          // --- Axial: open ends ---
          if (p.getX() < x_min || p.getX() > x_max)
          {
              p.state = ParticleState::DEGRADED;
              ++n_exited;
              return true;
          }

          // --- Radial: reflective wall ---
          double y   = p.getY();
          double z   = p.getZ();
          double rho = std::sqrt(y * y + z * z);

          if (rho > R)
          {
              double rho_orig = rho;

              int iter = 0;
              while (rho > R && iter < 20)
              {
                  rho = 2.0 * R - rho;
                  ++iter;
              }
              if (rho < 0.0) rho = 0.0;

              if (rho_orig > 1e-30)
              {
                  double scale = rho / rho_orig;
                  p.setY(y * scale);
                  p.setZ(z * scale);
              }
          }

          return false;
      }
  };

} // namespace molcompbs
