//* transmitter.h
// Point-source transmitter: releases N particles at a fixed position,
// either as an impulse or with a Gaussian spatial offset.
#pragma once

#include <vector>
#include "particle.h"

namespace molcompbs
{
  class Transmitter
  {
    private:
      double x, y, z;

    public:
      Transmitter() : x(0), y(0), z(0) {}
      Transmitter(double xin, double yin, double zin) : x(xin), y(yin), z(zin) {}

      void setPos(double xin, double yin, double zin) { x=xin; y=yin; z=zin; }
      double getX() const { return x; }
      double getY() const { return y; }
      double getZ() const { return z; }

      // Impulse release: all N particles placed exactly at (x,y,z)
      void release(std::vector<particle> &par, int N) const
      {
          par.reserve(par.size() + N);
          for (int n = 0; n < N; ++n)
              par.emplace_back(x, y, z);
      }

      // Release with Gaussian spatial offset (models finite-size Tx or
      // pre-diffusion over time Toff: stddev = sqrt(2*D*Toff))
      void release(std::vector<particle> &par, int N, double stddev) const
      {
          par.reserve(par.size() + N);
          for (int n = 0; n < N; ++n) {
              particle p(x, y, z);
              p.step(stddev);          // single Gaussian displacement
              par.emplace_back(p);
          }
      }
  };

} // namespace molcompbs
