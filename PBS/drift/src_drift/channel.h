//* channel.h  [src_drift]
//
// Extends the original channel with:
//   - Uniform drift velocity (vx, vy, vz)           // [DRIFT]
//   - Optional reflective cylindrical boundary       // [CYLINDER]
//
// All original API is preserved unchanged.
// New additions are marked with their tag.
//
#pragma once

#include <vector>
#include <cmath>
#include <random>
#include <chrono>
#include <functional>
#include "transmitter.h"
#include "receiver.h"
#include "particle.h"
#include "cylinder.h"   // [CYLINDER]

namespace molcompbs
{
  // -------------------------------------------------------------------------
  // Per-timestep simulation output
  // -------------------------------------------------------------------------
  struct SimStep {
      double t;
      int    N_internalized;
      int    N_free_inside;   // TRANSPARENT: FREE particles inside sphere this step.
                              // REACTIVE:    FREE particles inside sphere this step
                              //              that did NOT bind (= contacts − bindings).
                              // Kept for backward compatibility; prefer N_contact.
      int    N_contact;       // Particles that were inside the sphere this step
                              //   (all modes). TRANSPARENT: == N_free_inside.
                              //   REACTIVE:    = N_bound(new) + N_free_inside.
      int    N_bound;
      int    N_degraded_rx;
      int    N_degraded_med;
      int    N_active;
      int    N_exited;        // [CYLINDER] particles lost through open ends this step
  };

  // -------------------------------------------------------------------------
  class channel
  {
    private:
      double D, dt, T, Toff, stddev, stddevoff;
      double kd_medium;
      int N;
      std::vector<bool> bitseq;
      Transmitter TX;
      Receiver    RX;
      std::vector<particle> particles;
      int n_degraded_medium;
      int n_exited;           // [CYLINDER] axial exits in last propagation()

      // [DRIFT]
      double vx, vy, vz;

      // [CYLINDER]
      Cylinder cyl;

      void updateStddev() {
          stddev = std::sqrt(2.0 * D * dt);
          cyl.warnIfStepLarge(stddev);
      }
      void updateStddevoff() {
          stddevoff = (Toff > 0.0) ? std::sqrt(2.0 * D * Toff) : 0.0;
      }
      void propagation();

      // [VIZ] optional per-step observer (see setStepHook)
      std::function<void(int, double, int, const std::vector<particle>&)> stepHook;
      int next_id = 0;

    public:
      channel();
      ~channel() {}

      // ----------------------------------------------------------------------
      // Original setters / getters — unchanged
      // ----------------------------------------------------------------------
      void setD(double Din)            { D    = Din;  updateStddev(); }
      void setdt(double dtin)          { dt   = dtin; updateStddev(); }
      void setT(double Tin)            { T    = Tin; }
      void setToff(double Toffin)      { Toff = Toffin; updateStddevoff(); }
      void setN(int Nin)               { N    = Nin; }
      void setBit(std::vector<bool> b) { bitseq = b; }

      void setKf(double v) { RX.setKf(v); }
      void setKb(double v) { RX.setKb(v); }
      void setKi(double v) { RX.setKi(v); }
      void setKd(double v) { kd_medium = v; }

      double getD()      const { return D; }
      double getdt()     const { return dt; }
      double getT()      const { return T; }
      int    getN()      const { return N; }
      double getstddev() const { return stddev; }
      std::vector<bool> getBit() const { return bitseq; }

      // ----------------------------------------------------------------------
      // [DRIFT] Set / get drift velocity
      // ----------------------------------------------------------------------
      void setV (double vxin, double vyin, double vzin) { vx=vxin; vy=vyin; vz=vzin; }
      void setVx(double vxin) { vx = vxin; }
      void setVy(double vyin) { vy = vyin; }
      void setVz(double vzin) { vz = vzin; }
      double getVx() const { return vx; }
      double getVy() const { return vy; }
      double getVz() const { return vz; }

      // ----------------------------------------------------------------------
      // [CYLINDER] Enable / disable cylindrical boundary
      //
      //   setCylinder(R, x_min, x_max)
      //     R      : vessel radius [m]
      //     x_min  : left  open end [m]  (particles exiting here are removed)
      //     x_max  : right open end [m]  (particles exiting here are removed)
      //
      //   setCylinder(R) — convenience: auto-range [0, 2 * Rx_x].
      //                    Call AFTER setRX() so Rx position is known.
      //   setCylinder(0) — disables the boundary.
      // ----------------------------------------------------------------------
      void setCylinder(double Rin, double x_min, double x_max)
      { cyl.set(Rin, x_min, x_max); cyl.warnIfStepLarge(stddev); }

      void setCylinder(double Rin)
      {
          if (Rin <= 0.0) { cyl.disable(); return; }
          double rx_x = RX.getX();
          cyl.set(Rin, 0.0, (rx_x > 0.0 ? 2.0 * rx_x : 1e-3));
          cyl.warnIfStepLarge(stddev);
      }

      void disableCylinder()       { cyl.disable(); }
      bool cylinderActive() const  { return cyl.isActive(); }

      // ----------------------------------------------------------------------
      // Transmitter / receiver setup — unchanged
      // ----------------------------------------------------------------------
      void setTX(double xin, double yin, double zin)
      { TX.setPos(xin, yin, zin); }

      void setRX(double xin, double yin, double zin, double rin, bool transparentin)
      {
          RX.setPos(xin, yin, zin);
          RX.setR(rin);
          RX.setMode(transparentin ? ReceiverMode::TRANSPARENT
                                   : ReceiverMode::ABSORBING);
      }

      void setRXReactive(double xin, double yin, double zin, double rin,
                         double kf, double kb, double ki)
      {
          RX.setPos(xin, yin, zin);
          RX.setR(rin);
          RX.setMode(ReceiverMode::REACTIVE);
          RX.setKf(kf);
          RX.setKb(kb);
          RX.setKi(ki);
          RX.setKd(kd_medium);
      }

      void setReceptors(int M, double rs_rec) { RX.setReceptors(M, rs_rec); }

      // ----------------------------------------------------------------------
      // Simulation
      // ----------------------------------------------------------------------
      std::vector<bool>    random_bits(int K);
      std::vector<int>     simulate();
      std::vector<SimStep> simulateDetailed();

      // ----------------------------------------------------------------------
      // [VIZ] Optional observer, called three times per step with the live
      // particle vector: phase 0 after release / before propagation, phase 1
      // after propagation (medium-degraded particles already erased),
      // phase 2 after RX.process (internalized particles already erased).
      // Used offline by particle_snapshots.cpp; leave unset in timed runs.
      //   hook(step n, post-step time t, phase, particles)
      // ----------------------------------------------------------------------
      void setStepHook(std::function<void(int, double, int, const std::vector<particle>&)> h)
      { stepHook = std::move(h); }
  };

} // namespace molcompbs
