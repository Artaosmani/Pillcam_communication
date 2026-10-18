# Visualizers

Offline figures for the thesis. Nothing here runs inside the timed loop.

| Script | Figure | Reads |
|---|---|---|
| `plot_latency_breakdown.py` | Stacked bar per detected box: capture, detection, network, mapping, lookup, decision, plus physical delivery time t½ on top; side panel with mean / p95 per stage | detector log (`boxes_fake.csv` layout) + `bbox_replay_out.csv` |
| `animate_particles.py` | Particle animation (MP4/GIF) + four stills: receiver sphere as a circle, particles coloured free / bound / internalized / degraded | `particle_snapshots.csv` from `particle_snapshots.cpp` |
| `demo_overlay.py` (optional) | Frame with box, Pi_T, t½ and inject decision drawn on top | detector log + `bbox_replay_out.csv` (+ frame images) |

`vizstyle.py` holds the shared colours. Needs Python 3 with numpy and matplotlib; ffmpeg for MP4 (use `--gif` without it).

## Fake data, and how it disappears

- **Detection times** come from `t_detect_ns - t_capture_ns` in the detector log. They are flagged fake while the file name contains "fake" (pass `--real-boxes` to override).
- **Capture** and **network** have no log yet, so the script draws FAKE values (constants `FAKE` at the top: capture 8 ± 1.5 ms, network 3 ± 1 ms). They switch to real values automatically when the detector log has a `capture_ms` column and the server log has `t_recv_ns`.
- Fake segments are hatched and marked "(fake)" in the legend; `latency_stages.csv` lists them in `fake_stages`.
- **lookup** is currently the full PBS run per box (~0.8–1 s here), which is why it dwarfs the 100 ms budget. It becomes the table lookup after the speed-up.
- Demo overlay draws a grey PLACEHOLDER frame until real frames are given with `--frames <dir>` (files named `<frame_id>.png/.jpg`).

## Simulator changes (in `PBS_package_visualizers.zip`, drift/)

- `bbox_replay.cpp`: output gets three extra columns `mapping_ms,lookup_ms,decision_ms` (wall time of each server stage; `latency_ms` unchanged and still the whole of `process_box()`).
- `particle_snapshots.cpp` (new): small run (default N = 200, dt 5e-8, every 100 steps) that writes positions and states. It also counts FREE particles left inside the sphere after the receiver step; 0 means the push-back after a failed binding works. Example run: 0.
- Engine: `particle` gets an `id` field and `channel` an optional `setStepHook()` observer (tagged `[VIZ]`). Physics unchanged; when the hook is unset the cost is one empty check per step.

## Run it

```bash
cd PBS_package/drift
g++ -O3 -fopenmp bbox_replay.cpp        src_drift/particle.cpp src_drift/channel.cpp -o bbox_replay
g++ -O3 -fopenmp particle_snapshots.cpp src_drift/particle.cpp src_drift/channel.cpp -o particle_snapshots
./bbox_replay && ./particle_snapshots

cd <visualizers>
python plot_latency_breakdown.py --boxes <drift>/data/boxes_fake.csv --server <drift>/bbox_replay_out.csv --out-dir out
python animate_particles.py      --csv <drift>/particle_snapshots.csv --out-dir out
python demo_overlay.py           --boxes <drift>/data/boxes_fake.csv --server <drift>/bbox_replay_out.csv --out-dir out/overlay
```

`example_output/` has the result of exactly these commands on the fake boxes.
