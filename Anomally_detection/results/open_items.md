## What the detection total does and doesn't include

The 100 ms target runs from "frame received on the board" to "boxes out on the PC". In this run the board is a PC
stand-in, so the measured total is **PC preprocessing + PC TFLite classifier + YOLO26n**, with USB = 0. The pieces
still missing, and the best current estimate for each (not measured):

| Missing piece | Estimate | Source |
|---|---|---|
| Board JPEG decode (1/4 scale) + LUT on the 128 MHz M33 | unknown, likely 5–30 ms | none yet: first thing to time on the DK |
| Axon inference, MobileNetV1 0.25 | ≈ 14 ms at 96×96, a bit less at 84×84 | Nordic's published figure for this architecture |
| USB board → PC, 40 KB frame on "yes" | < 1 ms high speed (480 Mbit/s); ≥ 27 ms full speed (12 Mbit/s) | wire rate only |

Even with the slow end of every estimate (30 + 14 + 27 ms) added to the measured 18.6 ms p95, the total stays
under 100 ms. That is an estimate until the board numbers exist.

## Still open

1. **Board numbers.** `Board/` is written but not compiled (no nRF Connect SDK here). Flash it, run
   `Board/receiver.py`, then `pipeline.py --clf ...`: stages 2 and 3 then become **measured**. Check the Axon input
   layout (CHW) first, see `Board/README.md`.
2. **Stage 1 (frame → board)** stays a placeholder: over USB in the lab, the pill's radio link in the real system.
3. **Capsule domain shift.** Both classifiers find **none** of the 55 Kvasir-Capsule polyp frames (ROC-AUC 0.66–0.74),
   although they reject capsule normal mucosa almost perfectly. All positives in training are colonoscopy
   frames. A pill system needs capsule polyp frames in training. Kvasir-Capsule has only these 55, all from one video,
   so another capsule polyp source is needed. Until then, the colonoscopy test numbers above do not carry over to
   capsule frames.
4. **Classifier INT8 at 96×96** misses the 95 % recall target on test (0.941, threshold set on val). The 84×84 model
   meets it (0.961) and is now the default for the board and the pipeline.
5. **YOLO26n format.** INT8 ONNX costs ~5 mAP50-95 points and ~1 IoU point, for no speed gain over OpenVINO FP16
   here. The pipeline uses **OpenVINO FP16 at 4 threads**: same accuracy as FP32, 11.7 ms p95. This is an Apple M3
   (ARM). Re-run `YOLO/bench.py` on the PC that will host the detector, since x86 has fast INT8 kernels and the
   ranking may change.
6. **Simulation per box ≈ 785 ms p50** (full PBS run, N = 1000) is far above the budget. That is the lookup-table
   speed-up (Melwin), outside detection.
7. **Placeholders kept on purpose.** Pixel → mm mapping in `box_to_geometry()`; *capture* and *network* in the
   latency figure are still the visualizer's fake values.
8. **YOLO26n-P2** is training overnight; its results are added to this report when it finishes.
9. **Small test sets.** 100 Kvasir-SEG test polyps for the boxes, 153 polyp frames for the classifier (53 of them are
   Kvasir v1 copies of Kvasir-SEG test images in the uncropped format), 55 capsule polyps from one video. Read
   differences of a point or two as noise.
