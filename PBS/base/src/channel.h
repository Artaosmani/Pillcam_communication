//* channel.h
#pragma once

#include <vector>
#include <cmath>
#include <random>
#include <chrono>
#include "transmitter.h"
#include "receiver.h"
#include "particle.h"

namespace molcompbs
{
  // -------------------------------------------------------------------------
  // Per-timestep simulation output
  // -------------------------------------------------------------------------
  struct SimStep {
      double t;               // time [s]
      int    N_internalized;  // particles internalized this step         (Pi)
      int    N_free_inside;   // FREE particles inside Rx this step       (Pe proxy)
      int    N_bound;         // BOUND particles this step                (Pf)
      int    N_degraded_rx;   // degraded inside receiver this step       (Pd partial)
      int    N_degraded_med;  // degraded in free medium this step        (Pd partial)
      int    N_active;        // FREE + BOUND particles still alive
  };

  // -------------------------------------------------------------------------
  class channel
  {
    private:
      double D, dt, T, Toff, stddev, stddevoff;
      int N;
      std::vector<bool> bitseq;
      Transmitter TX;
      Receiver    RX;
      std::vector<particle> particles;
      int n_degraded_medium;   // free-medium degradations in last propagation()

      void updateStddev()    { stddev    = std::sqrt(2.0 * D * dt); }
      void updateStddevoff() { stddevoff = (Toff > 0.0)
                                           ? std::sqrt(2.0 * D * Toff) : 0.0; }
      void propagation();

    public:
      channel();
      ~channel() {}

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

      double getD()   const { return D; }
      double getdt()  const { return dt; }
      double getT()   const { return T; }
      int    getN()   const { return N; }
      double getstddev() const { return stddev; }
      std::vector<bool> getBit() const { return bitseq; }

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

      std::vector<bool> random_bits(int K);
      std::vector<int>  simulate();
      std::vector<SimStep> simulateDetailed();

    private:
      double kd_medium = 0.0;
  };

} // namespace molcompbs
