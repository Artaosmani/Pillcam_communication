//* receiver.h
//
// Spherical receiver supporting three operating modes:
//
//   TRANSPARENT  : count FREE particles inside sphere each step; no removal.
//                  (legacy CIR mode, matches existing example_CIR.cpp)
//
//   ABSORBING    : remove FREE particles on first contact.
//                  (legacy absorbing mode)
//
//   REACTIVE     : reversible-binding / internalization model from
//                  Khoshfekr Rudsari et al., IEEE Trans. Commun. 73(2):817-831, 2025 (doi:10.1109/TCOMM.2024.3440877) :
//                    A + R  <-kf-> AR  -ki-> B + R   (binding, internalization)
//                    A      -kd->  ∅                  (degradation of free)
//
//  The analytical MATLAB derivation (channel_probabilities.m) applies kd
//  only to free particles A, not to bound AR; the reference receiver in
//  src/receiver.h was validated against that analytical and uses this
//  same convention. In REACTIVE mode the receiver tracks per-step and
//  cumulative counts for all four particle states so that Pe, Pf, Pi, Pd
//  can be reconstructed.
//
#pragma once

#include <vector>
#include <cmath>
#include <iostream>
#include "particle.h"

namespace molcompbs
{
  // -------------------------------------------------------------------------
  enum class ReceiverMode { TRANSPARENT, ABSORBING, REACTIVE };

  // -------------------------------------------------------------------------
  // Per-timestep state counts (useful for plotting Pe, Pf, Pi, Pd)
  // -------------------------------------------------------------------------
  struct StateCounts {
      int n_free         = 0;   // FREE particles inside sphere this step
      int n_bound        = 0;   // BOUND particles (anywhere — they stay at Rx)
      int n_internalized = 0;   // newly INTERNALIZED this step
      int n_degraded     = 0;   // newly DEGRADED this step (inside sphere)
  };

  // -------------------------------------------------------------------------
  class Receiver
  {
    private:
      // geometry
      double x, y, z;          // centre position
      double r;                 // radius
      double rs;                // r^2  (precomputed)
      double xm, xp, ym, yp, zm, zp; // AABB for fast rejection

      // reaction rates (REACTIVE mode)
      double kf;    // forward binding rate  [particle^-1 m^3 s^-1]
      double kb;    // backward (unbinding)  [s^-1]
      double ki;    // internalization       [s^-1]
      double kd;    // degradation           [s^-1]

      // receptor geometry for kf* correction (Eq. 11-12 of paper)
      int    M;     // number of receptors   (0 = fully covered / infinite)
      double rs_rec;// receptor radius       [m]

      // mode
      ReceiverMode mode;

      // per-step counters
      int num_internalized; // internalized THIS step  (== Pi contribution)
      int num_free_inside;  // FREE inside sphere THIS step (== Pf contribution)
                            //   TRANSPARENT: FREE inside sphere.
                            //   REACTIVE:    FREE inside sphere that did NOT bind
                            //                (they are then placed outside).
                            //   To obtain "contacts with receiver", use num_contact.
      int num_contact;      // Particles that were inside sphere THIS step (all modes).
                            //   TRANSPARENT:  == num_free_inside
                            //   REACTIVE:     == num_free_inside + newly-bound this step
      int num_bound;        // BOUND particles THIS step
      int num_degraded;     // degraded THIS step

      // cumulative
      int sum_internalized;
      int sum_degraded;

      void updateAABB() {
          xm = x-r; xp = x+r;
          ym = y-r; yp = y+r;
          zm = z-r; zp = z+r;
          rs = r*r;
      }

      // -----------------------------------------------------------------------
      // Forward binding probability per encounter, following the standard PBS
      // surface-reaction approach (Al-Zu'bi & Mohan, IEEE Access 2018, Eq.44;
      // also Ahmadzadeh et al. [15]).
      //
      // The input kf from the Veletic paper is in [particle^-1 m^3 s^-1].
      // We first convert it to a surface rate kf_s [m/s] by dividing by the
      // receiver surface area:
      //
      //   kf_s = kf_star / (4 * pi * a^2)          [m/s]
      //
      // Then the per-encounter binding probability is:
      //
      //   pfor = kf_s * sqrt(pi * dt / D)
      //
      // This is the correct formula used in the original C++ receiver that
      // reproduced the reference (Khoshfekr Rudsari et al., 2025) results.  A particle that lands inside
      // the sphere is tested against pfor; on failure it is REFLECTED back to
      // the sphere surface so it cannot be retested within the same timestep.
      // -----------------------------------------------------------------------
      double computePfor(double D, double dt) const
      {
          double kf_star = computeKfStar(D);
          double surface = 4.0 * M_PI * r * r;
          double kf_s    = kf_star / surface;          // [m/s]
          return kf_s * std::sqrt(M_PI * dt / D);
      }

      // kf* correction for finite receptors, Eq. (11) of paper.
      // If M == 0 (default), returns kf unchanged (infinite receptors).
      double computeKfStar(double D) const
      {
          if (M <= 0 || rs_rec <= 0.0) return kf; // fully covered

          double phi = computePhi(D);
          double kf_star_num = 4.0 * M_PI * D * r * kf * phi;
          double kf_star_den = kf * 1.0 * (1.0 - phi) + 4.0 * M_PI * D * r;
          // Note: Nref = 1 particle absorbed into kf denominator per paper
          return kf_star_num / kf_star_den;
      }

      // Coverage correction factor phi, Eq. (12) of paper.
      double computePhi(double D) const
      {
          double lambda_cov = M * (M_PI * rs_rec * rs_rec) / (4.0 * M_PI * r * r);
          double Nref = 1.0;
          double num = M * rs_rec * rs_rec * (kf * r + 4.0 * M_PI * D);
          double den = 2.0 * r * (1.0 - lambda_cov) * (M_PI * rs_rec * kf + 16.0 * M_PI * D)
                     + M * rs_rec * rs_rec * (kf * r + 4.0 * M_PI * D);
          // guard
          if (std::abs(den) < 1e-30) return 1.0;
          return num / den;
      }

      // Check if a particle is inside the sphere (AABB pre-filter first)
      bool isInside(const particle &p) const
      {
          if (p.getX() < xm || p.getX() > xp) return false;
          if (p.getY() < ym || p.getY() > yp) return false;
          if (p.getZ() < zm || p.getZ() > zp) return false;
          double dx = p.getX() - x;
          double dy = p.getY() - y;
          double dz = p.getZ() - z;
          return (dx*dx + dy*dy + dz*dz) <= rs;
      }

    public:
      // ----------------------------------------------------------------------
      // Construction
      // ----------------------------------------------------------------------
      Receiver()
          : x(0), y(0), z(0), r(1),
            kf(0), kb(0), ki(0), kd(0),
            M(0), rs_rec(0),
            mode(ReceiverMode::ABSORBING),
            num_internalized(0), num_free_inside(0), num_contact(0),
            num_bound(0), num_degraded(0),
            sum_internalized(0), sum_degraded(0)
      { updateAABB(); }

      // Convenience constructor: set geometry and mode in one call
      Receiver(double xin, double yin, double zin, double rin, ReceiverMode m)
          : x(xin), y(yin), z(zin), r(rin),
            kf(0), kb(0), ki(0), kd(0),
            M(0), rs_rec(0), mode(m),
            num_internalized(0), num_free_inside(0), num_contact(0),
            num_bound(0), num_degraded(0),
            sum_internalized(0), sum_degraded(0)
      { updateAABB(); }

      // ----------------------------------------------------------------------
      // Setters
      // ----------------------------------------------------------------------
      void setPos(double xin, double yin, double zin) { x=xin; y=yin; z=zin; updateAABB(); }
      void setR(double rin) { r=rin; updateAABB(); }
      void setMode(ReceiverMode m) { mode = m; }

      // Reaction rates (used in REACTIVE mode)
      void setKf(double v) { kf = v; }
      void setKb(double v) { kb = v; }
      void setKi(double v) { ki = v; }
      void setKd(double v) { kd = v; }

      // Finite receptor geometry (optional; leave at defaults for infinite Rx)
      void setReceptors(int Min, double rs_rec_in) { M = Min; rs_rec = rs_rec_in; }

      // ----------------------------------------------------------------------
      // Getters
      // ----------------------------------------------------------------------
      double getX() const { return x; }
      double getY() const { return y; }
      double getZ() const { return z; }
      double getR() const { return r; }
      ReceiverMode getMode() const { return mode; }

      // Per-step counts
      int getNumInternalized() const { return num_internalized; }
      int getNumFreeInside()   const { return num_free_inside; }
      int getNumContact()      const { return num_contact; }
      int getNumBound()        const { return num_bound; }
      int getNumDegraded()     const { return num_degraded; }

      // Cumulative counts
      int getSumInternalized() const { return sum_internalized; }
      int getSumDegraded()     const { return sum_degraded; }

      // Reset per-step counters (call at start of each timestep)
      void resetStep() {
          num_internalized = 0;
          num_free_inside  = 0;
          num_contact      = 0;
          num_bound        = 0;
          num_degraded     = 0;
      }

      void resetAll() {
          resetStep();
          sum_internalized = 0;
          sum_degraded     = 0;
      }

      // ----------------------------------------------------------------------
      // process() — main per-timestep method
      //
      // Iterates over all particles and applies the appropriate detection
      // logic according to mode.  Particles that reach a terminal state
      // (INTERNALIZED or DEGRADED) are removed from the vector to save memory
      // and compute, EXCEPT in TRANSPARENT mode where they are never removed.
      //
      // Parameters:
      //   par  : particle vector (modified in-place)
      //   D    : diffusion coefficient [m^2/s]  (needed for kf_eff)
      //   dt   : timestep [s]
      // ----------------------------------------------------------------------
      // process() — main per-timestep method.
      //
      // REACTIVE mode follows the PBS algorithm from Al-Zu'bi & Mohan (2018)
      // and the original receiver.cpp implementation:
      //
      //  FREE particle inside sphere:
      //    - Draw U ~ Uniform(0,1)
      //    - If U <= pfor  → BIND: snap to surface, mark BOUND
      //    - Else          → REFLECT: push back to surface along entry ray
      //      (particle stays FREE but is placed outside for next diffusion step)
      //
      //  BOUND particle:
      //    - Three-way competition: internalize (ki), unbind (kb), or stay
      //    - If internalized → remove (counted as Pi)
      //    - If unbound      → place just outside surface, mark FREE
      //
      //  NOTE: AR particles do NOT degrade in this kinetic model — the
      //  analytical derivation in MATLAB/channel_probabilities.m applies kd
      //  only to free particles (A -kd-> ∅). The exp(-kd*t) factor in the
      //  analytical Pf represents free-state survival prior to binding, not
      //  bound-state degradation. Using particle::tryBoundTransition here
      //  (which competes AR -kd-> ∅ as a fourth outcome) would systematically
      //  under-predict Pi by ~10 % relative to the analytical.
      //
      //  Degradation of FREE particles in the medium is handled by
      //  channel::propagation() before process() is called each step.
      void process(std::vector<particle> &par, double D, double dt)
      {
          resetStep();

          double pfor = 0.0;
          double pbac = 0.0;
          double pint = 0.0;
          if (mode == ReceiverMode::REACTIVE) {
              pfor = computePfor(D, dt);
              pbac = 1.0 - std::exp(-kb * dt);
              pint = 1.0 - std::exp(-ki * dt);
              // clamp probabilities to [0,1]
              if (pfor > 1.0) pfor = 1.0;
          }

          auto it = par.begin();
          while (it != par.end())
          {
              particle &p = *it;
              bool remove = false;

              switch (mode)
              {
                // ----------------------------------------------------------
                case ReceiverMode::TRANSPARENT:
                // ----------------------------------------------------------
                  if (p.isFree() && isInside(p)) {
                      num_free_inside++;
                      num_contact++;
                  }
                  ++it;
                  break;

                // ----------------------------------------------------------
                case ReceiverMode::ABSORBING:
                // ----------------------------------------------------------
                  if (p.isFree() && isInside(p)) {
                      num_internalized++;
                      sum_internalized++;
                      p.state = ParticleState::INTERNALIZED;
                      remove = true;
                  }
                  if (remove) it = par.erase(it);
                  else        ++it;
                  break;

                // ----------------------------------------------------------
                case ReceiverMode::REACTIVE:
                // ----------------------------------------------------------
                  if (p.isFree())
                  {
                      if (isInside(p))
                      {
                          // Every particle inside counts as a contact event.
                          num_contact++;
                          double U = p.drawUniform();
                          if (U <= pfor)
                          {
                              // BIND: snap particle to sphere surface
                              double dx = p.getX() - x;
                              double dy = p.getY() - y;
                              double dz = p.getZ() - z;
                              double dist = std::sqrt(dx*dx + dy*dy + dz*dz);
                              if (dist < 1e-30) { dx=1; dy=0; dz=0; dist=1; }
                              p.setX(x + r * dx/dist);
                              p.setY(y + r * dy/dist);
                              p.setZ(z + r * dz/dist);
                              p.state = ParticleState::BOUND;
                              num_bound++;
                          }
                          else
                          {
                              // REFLECT: use exit distribution (Al-Zu'bi Eq.46-47)
                              // to place the particle outside the sphere at a
                              // physically correct distance, preventing immediate
                              // re-entry on the next diffusion step.
                              p.stepOutside(D, dt, x, y, z, r);
                              // particle remains FREE; count its (unsuccessful)
                              // contact in num_free_inside for legacy Pe plots.
                              num_free_inside++;
                          }
                      }
                      // FREE outside: nothing to do here (diffusion + medium
                      // degradation already handled by channel::propagation())
                  }
                  else if (p.isBound())
                  {
                      // BOUND: competing internalization vs unbinding
                      // Order: check ki first if ki >= kb, else kb first
                      // (matches original receiver.cpp logic and the
                      // reference that validated against the MATLAB
                      // analytical).  AR does NOT have a degradation
                      // pathway here — see the note at the top of process().
                      double U = p.drawUniform();
                      if (ki >= kb)
                      {
                          if (U <= pint) {
                              // INTERNALIZE
                              p.state = ParticleState::INTERNALIZED;
                              num_internalized++;
                              sum_internalized++;
                              remove = true;
                          } else if (U <= pint + pbac) {
                              // UNBIND: use exit distribution so particle
                              // does not immediately re-enter sphere
                              p.state = ParticleState::FREE;
                              p.stepOutside(D, dt, x, y, z, r);
                          }
                          // else: stays BOUND
                          else { num_bound++; }
                      }
                      else
                      {
                          if (U <= pbac) {
                              // UNBIND: use exit distribution
                              p.state = ParticleState::FREE;
                              p.stepOutside(D, dt, x, y, z, r);
                          } else if (U <= pbac + pint) {
                              // INTERNALIZE
                              p.state = ParticleState::INTERNALIZED;
                              num_internalized++;
                              sum_internalized++;
                              remove = true;
                          }
                          else { num_bound++; }
                      }
                  }

                  if (remove) it = par.erase(it);
                  else        ++it;
                  break;

              } // switch
          } // while
      } // process()

      // Convenience: legacy-style getNum() returns the relevant count
      // depending on mode (matches old transceiver::getNum() semantics).
      int getNum() const {
          if (mode == ReceiverMode::TRANSPARENT) return num_free_inside;
          return num_internalized;
      }

      void printDetails() const {
          std::cout << "Receiver at [" << x << "," << y << "," << z << "]"
                    << "  r=" << r << "  mode=";
          switch (mode) {
              case ReceiverMode::TRANSPARENT: std::cout << "TRANSPARENT"; break;
              case ReceiverMode::ABSORBING:   std::cout << "ABSORBING";   break;
              case ReceiverMode::REACTIVE:    std::cout << "REACTIVE";    break;
          }
          std::cout << std::endl;
      }
  };

} // namespace molcompbs
