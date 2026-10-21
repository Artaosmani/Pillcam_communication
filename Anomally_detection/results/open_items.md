## What the detection total does and doesn't include

The 100 ms target runs from "frame received on the board" to "boxes out on the PC". Board preprocessing, Axon
inference and board → PC are now **measured on the nRF54LM20 DK** (section above), YOLO26n on the PC CPU. The frame's
own transfer PC → board (stage 1) is measured too but not part of the total. The latency table above is from the
**raw 84×84** board run (`runs/pipeline_board_raw`).

| Input sent to the board | Board compute p50 (preprocessing + Axon) | Detection total p50 / p95 | Within 100 ms? |
|---|---|---|---|
| 336×336 JPEG, decoded on the board | 207.7 ms (193 + 14.9) | 258 / 283 ms | no |
| 84×84 raw RGB, resized on the PC | 31.7 ms (17.0 + 14.7) | **75.7 / 89.4 ms** | **yes** |

## Still open

1. **PC → board transfer is slow: ≈ 1.6 ms per 512-byte USB packet** (≈ 68 ms per 20 KB frame at 480 Mbit/s).
   Not in the detection total, but it limits the frame rate. Logging and the PC write pattern are ruled out; see
   "Known issue" in `Board/README.md` for what is checked and the next step.
2. **Raw-mode preprocessing (17 ms) can drop to ≈ 1 ms.** At 84×84 `to_input()` still runs its general float
   area-resize loop although nothing is resized; a direct LUT copy into CHW is enough. Not done yet, so the numbers
   above match the firmware in the repo.
3. **What the pill sends.** Raw 84×84 removes the board's JPEG decode (193 ms → 17 ms) but moves the decode + resize
   to the sender (≈ 10 ms on the M3 PC here) and needs the full frame for YOLO26n from somewhere else. In the lab
   pipeline YOLO26n reads the full frame from disk.
4. **Stage 1 (frame → board)** in the real system is the pill's radio link; here it is USB from the PC.
5. **Capsule domain shift.** Both classifiers find **none** of the 55 Kvasir-Capsule polyp frames (ROC-AUC 0.66–0.81),
   although they reject capsule normal mucosa almost perfectly. All positives in training are colonoscopy
   frames. A pill system needs capsule polyp frames in training. Kvasir-Capsule has only these 55, all from one video,
   so another capsule polyp source is needed. Until then, the colonoscopy test numbers above do not carry over to
   capsule frames.
6. **Classifier INT8 at 96×96** misses the 95 % recall target on test (0.941, threshold set on val). The 84×84 model
   meets it (0.961) and is now the default for the board and the pipeline.
7. **YOLO26n format.** INT8 ONNX costs ~5 mAP50-95 points and ~1 IoU point, for no speed gain over OpenVINO FP16
   here. The pipeline uses **OpenVINO FP16 at 4 threads**: same accuracy as FP32, 11.7 ms p95. This is an Apple M3
   (ARM). Re-run `YOLO/bench.py` on the PC that will host the detector, since x86 has fast INT8 kernels and the
   ranking may change.
8. **Simulation per box ≈ 785 ms p50** (full PBS run, N = 1000) is far above the budget. That is the lookup-table
   speed-up (Melwin), outside detection.
9. **Placeholders kept on purpose.** Pixel → mm mapping in `box_to_geometry()`; *capture* and *network* in the
   latency figure are still the visualizer's fake values.
10. **YOLO26n-P2** (extra stride-4 head) finds more polyps (mAP50 0.943 vs 0.930, 99 vs 95 of 104 matched) but its
   boxes are looser (mAP50-95 0.69 vs 0.73, mean IoU 0.873 vs 0.894, ~11 vs ~9 px error, ~4 px too narrow), it raises
   more false alarms (11 vs 7 of 648) and is ~30 % slower (15.2 vs 11.7 ms p95). Box tightness is the priority for
   the simulator, so **YOLO26n stays the default**. Part of P2's recall gain comes from its lower deployment
   confidence (0.25 vs 0.39). Kvasir-SEG has few small polyps, so the case P2 is designed for is barely tested here.
11. **Small test sets.** 100 Kvasir-SEG test polyps for the boxes, 153 polyp frames for the classifier (53 of them are
   Kvasir v1 copies of Kvasir-SEG test images in the uncropped format), 55 capsule polyps from one video. Read
   differences of a point or two as noise.
