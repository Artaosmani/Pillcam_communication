# PBS package — Particle-Based Simulation of the reactive molecular-communication receiver

This package collects the **PBS (particle-based simulation)** pipeline used to
validate the analytical reactive-receiver model against stochastic Monte-Carlo
particle simulations, together with the data and plotting/analysis scripts needed
to reproduce the PBS overlays. It is assembled from two projects — the base SBI
project (`base/`) and the drift-channel project (`drift/`) — and is self-contained
apart from the large raw traces noted under *Regenerating the heavy data*.

"PBS" here means the **P**article-**B**ased **S**imulation: a 3-D random-walk
Monte-Carlo of molecules diffusing (and, in the drift case, advecting) to a
spherical reactive receiver, where they reversibly bind, internalize, and degrade.
The simulation reports the time-resolved **state probabilities** P(free),
P(bound), P(internalized), P(degraded), which are overlaid as markers on the
analytical curves (`plot_PBS.m`).

---

## 1. Layout

```
PBS_package/
├── README.md                     ← this file
├── base/                         (from the base project; engine in src/)
│   ├── src/                      particle + channel engine (channel.*, particle.*, receiver.h, transmitter.h)
│   ├── example_reactive.cpp      reactive-receiver PBS  → example_reactive.csv
│   ├── example_CIR.cpp           channel impulse response PBS
│   ├── example_OOK.cpp           OOK signalling PBS
│   ├── plot_PBS.m                overlay PBS markers on analytical curves
│   └── data/
│       └── example_reactive.csv  canonical PBS trace (dt=5e-9)
└── drift/                        (from the drift project; engine in src_drift/)
    ├── src_drift/                particle + channel engine (+ cylinder.h for the cylindrical geometry)
    ├── example_reactive.cpp      reactive-receiver PBS (drift build)
    ├── example_drift_CIR.cpp     advection–diffusion CIR PBS (point receiver)
    ├── example_drift_cylinder_CIR.cpp   advection–diffusion CIR PBS (cylindrical receiver)
    ├── plot_PBS.m                overlay script
    ├── diagnose_pbs_analytical.pdf / .png   PBS-vs-analytical diagnostic figure
    ├── data/
    │   ├── example_reactive.csv          canonical reactive PBS trace (dt=5e-9)
    │   ├── example_reactive_kd0.csv      degradation switched off (kd=0 isolation test)
    │   ├── example_reactive_dt5e-8.csv   coarser dt (dt-convergence sample)
    │   ├── example_drift_CIR.csv         drift CIR PBS (point Rx)
    │   ├── example_drift_cylinder_CIR.csv  drift CIR PBS (cylindrical Rx)
    │   └── expb_pbs/                     PBS overlays used in the paper's Experiment-B figure
    │       ├── D_pbs.dat  kd_pbs.dat  kf_pbs.dat
    └── matlab_probability/       PBS pieces of the MATLAB probability-analysis workspace
        ├── PBS_forward_probability.m     forward-binding probability derivation/check
        ├── kf_PBS.dat  kb_PBS.dat  kd_PBS.dat  ki_PBS.dat   rate-sweep PBS results
        └── SystemDynamics_PBS.dat        full PBS state-dynamics trace
```

**Excluded on purpose** (see §5): compiled binaries (rebuild from source) and the
multi-hundred-MB raw traces (`example_reactive_dt1e-9.csv` ≈107 MB and the full
`dt5e-9` dump ≈22 MB is included as the canonical one; the 107 MB fine-dt trace is
regenerable). Python virtualenvs and `.it` Monte-Carlo dumps are not part of PBS.

---

## 2. Reference parameters

**Reactive receiver** (`example_reactive.cpp`, Table I of Khoshfekr Rudsari *et al.*, 2025 — see §7):

| symbol | value | meaning |
|---|---|---|
| `D`  | `5e-9`  m²/s   | diffusion coefficient |
| `a`  | `0.5e-6` m     | receiver radius |
| `r0` | `1e-6`  m      | Tx–Rx distance |
| `T`  | `1e-3`  s      | observation window |
| `kf` | `1e-14` m³/s   | forward binding rate |
| `kb` | `2e3`   1/s    | unbinding rate |
| `ki` | `1e4`   1/s    | internalization rate |
| `kd` | `2e3`   1/s    | degradation rate (CLI `--kd`) |
| `dt` | `5e-9`  s      | time step (CLI `--dt`) |
| `N`  | `50000`        | particles (CLI `--N`) |

**Drift CIR** (`example_drift_CIR.cpp`): `D=1e-11` m²/s, `dist=3e-6` m, `rad=1e-6` m,
`v=2e-6` m/s, `N=1e4` particles/bit, `T=1` s, `dt=1e-3` s.

---

## 3. Build

Requires a C++ compiler with OpenMP (`g++` ≥ 7, or `clang++` with `-Xpreprocessor
-fopenmp`). No external libraries.

**Base:**
```bash
cd base
g++ -O3 -fopenmp example_reactive.cpp src/particle.cpp src/channel.cpp -o example_reactive
g++ -O3 -fopenmp example_CIR.cpp      src/particle.cpp src/channel.cpp -o example_CIR
g++ -O3 -fopenmp example_OOK.cpp      src/particle.cpp src/channel.cpp -o example_OOK
```

**Drift:**
```bash
cd drift
g++ -O3 -fopenmp example_reactive.cpp            src_drift/particle.cpp src_drift/channel.cpp -o example_reactive
g++ -O3 -fopenmp example_drift_CIR.cpp           src_drift/particle.cpp src_drift/channel.cpp -o example_drift_CIR
g++ -O3 -fopenmp example_drift_cylinder_CIR.cpp  src_drift/particle.cpp src_drift/channel.cpp -o example_drift_cylinder_CIR
```

macOS with Apple-clang: `brew install libomp` and compile with
`-Xpreprocessor -fopenmp -lomp`.

---

## 4. Run and plot

Each simulator writes its CSV to the **current directory**. `plot_PBS.m` reads
`example_reactive.csv` from the current directory, so run the simulator in the same
folder as `plot_PBS.m`, or copy `data/example_reactive.csv` next to it first.

```bash
cd drift
./example_reactive                 # Table I reference, dt=5e-9, N=50k → example_reactive.csv
./example_reactive --dt 5e-8       # faster, slightly lower fidelity
./example_reactive --kd 0          # degradation off (kd=0 isolation test)
./example_reactive --N 100000      # more particles → less Monte-Carlo noise
./example_reactive --no-abs        # skip the absorbing-Rx reference run
```

Then in MATLAB (or Octave), with an analytical figure open (or none, for a
markers-only plot):
```matlab
plot_PBS      % overlays PBS markers on the current figure
```

**CSV column formats**
- Reactive PBS (`example_reactive.csv`): `t, Pe, Pf, Pi, Pd, Psum, Pabs`
  = time [s], P(free/diffusing), P(bound), P(internalized/detected), P(degraded),
  conservation check `Pe+Pf+Pi+Pd≈1`, and P(absorbed) for a perfectly-absorbing Rx
  (upper bound on `Pi`).
- Drift CIR PBS (`example_drift_CIR.csv`): `t, y, Y, h` = time, per-bit hits,
  cumulative hits, and impulse-response estimate.

---

## 5. MATLAB probability-analysis PBS files

`matlab_probability/` holds the PBS pieces of the larger probability-analysis
workspace:

- `PBS_forward_probability.m` — derivation/verification of the single-step forward
  binding probability used to convert `kf` [m³/s] into the per-step hit
  probability of the particle walk (compares the simple boundary-condition form
  with the precise overlap integral).
- `kf_PBS.dat`, `kb_PBS.dat`, `kd_PBS.dat`, `ki_PBS.dat` — PBS internalized-probability
  curves under rate sweeps. Columns: `t, Pi1, Pi2, Pi4, Pi10, Pi20, Pi40`, i.e.
  P(internalized) vs time with the corresponding rate scaled by
  ×1, ×2, ×4, ×10, ×20, ×40 of its nominal value.
- `SystemDynamics_PBS.dat` — full PBS state-dynamics trace for the reference point.
- `expb_pbs/{D,kf,kd}_pbs.dat` (under `drift/data/`) — PBS overlays consumed by the
  paper's Experiment-B pgfplots figure; two columns `x y`.

---

## 6. Regenerating the heavy data (excluded from the package)

The multi-hundred-MB raw traces are not shipped; regenerate them from source:
```bash
cd drift
./example_reactive --dt 1e-9       # fine-dt reference trace (~107 MB)
./example_reactive                 # default dt=5e-9 (~22 MB; = data/example_reactive.csv)
```
Larger `--N` and finer `--dt` reduce Monte-Carlo noise at the cost of size/runtime.

---

## 7. Provenance

Model and reference parameters are from:

> H. Khoshfekr Rudsari, M. Damrath, M. Zoofaghari, I. Balasingham, and M.
> Veletić, "An Intercellular Communication System for Intra-Body Communication
> Networks," *IEEE Transactions on Communications*, vol. 73, no. 2, pp. 817–831,
> Feb. 2025. doi: 10.1109/TCOMM.2024.3440877

The `base/` and `drift/` engines are the same particle/channel core specialized to
no-drift and advection-diffusion geometries respectively. Questions on the kinetic
parameterization or the analytical curves the PBS overlays validate: contact the
sender.

---

## 8. Bounding-box replay (`drift/bbox_replay.cpp`)

Feeds detector boxes into the reactive PBS one box at a time and times each box.
It reads a CSV once, then for every row runs `process_box()` (box → geometry →
PBS → decision) and times only that call. A frame with two boxes is two rows,
timed separately. In the live system only the input loop changes (network
instead of file); `process_box()` and the timing stay the same.

Input (`data/boxes_fake.csv` is made-up data; replace with real detector output):
`frame_id,track_id,t_capture_ns,t_detect_ns,t_send_ns,x_min,y_min,x_max,y_max,img_w,img_h,confidence`
(absolute pixels, origin top-left).

Output (`bbox_replay_out.csv`):
`frame_id,track_id,r0_m,a_m,Pi_T,t_half_s,inject,dose_N,latency_ms`

```bash
cd drift
g++ -O3 -fopenmp bbox_replay.cpp src_drift/particle.cpp src_drift/channel.cpp -o bbox_replay
./bbox_replay                        # all rows of data/boxes_fake.csv
./bbox_replay --csv my_boxes.csv     # real detector output
./bbox_replay --rows 1               # only the first box
./bbox_replay --N 1000 --dt 5e-8     # PBS fidelity per box
./bbox_replay --conf 0.5 --need 100  # decision placeholders
```

Placeholders to replace once agreed: the pixel → metre mapping in
`box_to_geometry()` (currently keeps r0 and a near the Table I values) and the
decision rule at the end of `process_box()` (confidence gate, dose = need / Pi_T).
