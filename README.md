# Polyp detection on Kvasir-SEG

Two-stage pipeline for capsule endoscopy:

1. The camera pill sends a 336×336 frame to the **nRF54LM20 DK**.
2. The board runs the **polyp classifier** (`Classifier/`, INT8 on the Axon NPU) on every frame: polyp yes/no.
3. Over USB the board sends one result line per frame, plus the frame itself when the answer is yes.
4. The PC writes a CSV of every frame. **YOLO26n** (`YOLO/`, config `Configs/default.yml`) runs on frames with
   `detected=1` and outputs boxes for the drug-delivery simulator.

## Polyp classifier

MobileNetV1 α=0.25 (ImageNet weights) → average pool → dropout → 1×1 conv → **one logit**. There is no sigmoid in
the model; the board compares the raw int8 output with a threshold. Everything is in `Configs/classifier.yml`.

| File | What it does |
|---|---|
| `Configs/classifier.yml` | paths, negatives sources, input size (96 or 84), augmentation, two training stages, quantisation, threshold |
| `Classifier/data.py` | positives from `Data/kvasir-seg-split` (same train/val/test as YOLO26n), negatives from `negatives.sources`, de-duplication, video-level split, optional hard-negative crops |
| `Classifier/train.py` | training, val threshold, test eval float + INT8, INT8 export (QAT if needed), size/MACs/memory report |
| `Classifier/run_csv.py` | INT8 model over a folder of frames → one CSV row per frame |

### Setup

```bash
conda activate anomaly_detection
pip install -r requirements-classifier.txt
```

### Data and the shared split

| Folder | What | Used as |
|---|---|---|
| `Data/kvasir-seg-split/` | Kvasir-SEG, 800 / 100 / 100 with YOLO labels | positives (both models) |
| `Data/kvasir/kvasir-dataset/polyps` | Kvasir v1 polyps: 492 of 500 are Kvasir-SEG images with the border cropped | extra positives, each copy in its Kvasir-SEG image's split |
| `Data/negatives/hyperkvasir/` | HyperKvasir labelled, lower GI: cecum, ileum, bbps-0-1, bbps-2-3 | negatives |
| `Data/negatives/kvasir-capsule/` | Kvasir-Capsule "Normal clean mucosa" (34,338 frames, 37 videos), one frame per 15, whole videos per split | negatives + capsule eval |
| `Data/kvasir/kvasir-dataset/{ulcerative-colitis,dyed-resection-margins}` | Kvasir v1 | negatives |
| `Data/kvasir-capsule/polyp/` | Kvasir-Capsule polyps, 55 frames, all from video `131368cc17e44240` | capsule eval only |

Download: `hyper-kvasir-labeled-images.zip` (3.9 GB) and `kvasir-capsule-labeled-images.zip` (0.55 GB) from
datasets.simula.no; unpack only the classes above. Upper-GI classes (z-line, pylorus) are not used.

`Classifier/data.py` writes **`Data/splits.json`**, the one train/val/test split both models read. Duplicates are
handled like this:
- Kvasir-SEG images are cropped Kvasir frames, so an image hash misses the copies. `Classifier/twins.py` matches
  every Kvasir v1 polyp against Kvasir-SEG (masked pixel difference) → `labels/seg_twins.json`. Copies stay in their
  Kvasir-SEG image's split, so the same polyp never sits in train and test.
- Kvasir v1 `normal-cecum` is not used: HyperKvasir cecum contains it, cropped as well.
- The capsule polyp video is pinned to test, so none of its normal frames are trained on.
- The old `Data/test` (50 + 50 images moved out of Kvasir v1) was not held out: 37 of its 47 checked polyps were Kvasir-SEG
  images, 28 of them in Kvasir-SEG train. Those images are back in their folders; `Data/test` is now built by
  `pipeline.py` (see below).

### Train and export

```bash
python Classifier/train.py --dry-run                 # resolved config + split sizes
python Classifier/train.py                           # train, evaluate, export -> runs/classifier/
python Classifier/train.py --set imgsz=84            # 84×84 = 1/4-scale JPEG decode of the 336×336 frame
python Classifier/train.py --set hard_negatives=true stage2.lr=5e-5
python Classifier/train.py --weights runs/classifier/best.weights.h5   # re-quantise/evaluate/export only
```

Outputs in `runs/<name>/` (ignored by git):

- `best.weights.h5`: float weights, best val AUC
- `threshold.json`: picked on val for ≥95 % recall, then lowered as far as possible without adding a val false
  positive; stored as a float logit/probability and as a raw int8 value
- `metrics_test.json`: float and INT8 recall, precision, F1, specificity, ROC-AUC, confusion matrix and threshold,
  plus params, MACs, `.tflite` size, peak activation memory, op list and PC interpreter timing
- `metrics_capsule.json`: the same on Kvasir-Capsule frames, if they exist (domain-shift experiment)
- `model_int8.tflite`, `model_int8.h` (C array, quantisation constants, `POLYP_THRESHOLD_INT8`, a 256-entry input LUT)
- `preprocess.md`: exact input scale/zero-point, resize method, and how to apply the threshold

The PC timing in `metrics_test.json` comes from the TFLite interpreter on the PC. **It is not the board latency.**

### Checking and compiling for the Axon NPU

The Nordic tools come from the nRF Connect SDK **Edge AI add-on** (not installed here). With the add-on installed:

1. Check that every op maps to Axon: run `axons_tflite_model_scan.py` from the add-on on
   `runs/classifier/model_int8.tflite`. `train.py` already lists the ops and warns about any outside
   Conv2D / depthwise conv / fully connected / pooling / ReLU. A normal export contains only `CONV_2D`,
   `DEPTHWISE_CONV_2D` and `AVERAGE_POOL_2D`.
2. Compile the same `.tflite` with the add-on's Axon TFLite compiler to get the NPU command stream for the firmware.

Exact commands, the compiler YAML, the firmware and the PC receiver: [`Board/README.md`](Board/README.md).

### CSV for the PC side

```bash
python Classifier/run_csv.py --frames path/to/frames --model runs/classifier/model_int8.tflite
```

One row per frame, in frame order:

```
frame_id,t_rx_board_ns,t_infer_start_ns,t_infer_end_ns,t_sent_ns,score,detected
```

`frame_id` is the last number in the filename (`frame_001001.jpg` → 1001), the same id as in `boxes_fake.csv`, so
YOLO26n results join on it. On the PC the timestamps come from the host clock; the board will send the same columns.
`score` is the sigmoid of the dequantised logit, and `detected` uses the int8 threshold from `threshold.json`.

### Held-out test set

`Data/test/frames/frame_NNNNNN.jpg` are the 801 test-split images (153 polyp, 648 no-polyp) as 336×336 JPEG
replay frames, built by `pipeline.py` on first run. `Data/test/labels.csv` has `frame_id,file,polyp,source`, and
`Data/test/gt_boxes.csv` has the Kvasir-SEG boxes at 336 px.

```bash
python Classifier/run_csv.py --frames Data/test/frames --labels Data/test/labels.csv --model runs/classifier_96/model_int8.tflite
```

### Design choices for the NPU

- Keras' own MobileNet pads with `ZeroPadding2D` (a `PAD` op), and `GlobalAveragePooling2D` exports as `MEAN`.
  Neither is an Axon op, so the network is rebuilt with `padding='same'` and an `AveragePooling2D` over the whole
  feature map, with the ImageNet weights copied over by layer name.
- The head is a 1×1 conv, not Flatten + Dense, so there is no `RESHAPE` op either.
- Input and output are int8. The input is calibrated to the −1..1 range, so a uint8 pixel maps to int8 through
  `polyp_input_lut` in `model_int8.h`.
- The size is about 297 KB (218,801 params), above the ~250 KB target. Per-channel quantisation parameters and
  flatbuffer metadata add to the 214 KB of int8 weights; this is the same size class as the MLPerf Tiny VWW model,
  and the board has 2 MB of flash.
- QAT runs when INT8 val recall is more than 2 points below float. By default QAT gives the average pool its own
  output quantiser, which exports as an extra int8 → int8 `QUANTIZE` op (not an Axon op). `train.py` turns that
  quantiser off, so the QAT export uses the same three ops. `train.py` still warns if any other op appears.

## YOLO26n

`Configs/default.yml`; `YOLO/src.py` builds `Data/yolo-split/` from Kvasir-SEG plus the shared-split negatives as
background images (empty label files): train/val get 0.5 × the polyp count, test gets every test negative.

```bash
python YOLO/train.py --name yolo26n                              # 150 epochs, patience 30, imgsz 352
python YOLO/train.py --model Configs/yolo26.yml --name yolo26n_p2   # P2 head for small polyps
python YOLO/export.py   --weights runs/yolo26n/weights/best.pt    # ONNX FP32, ONNX INT8 (head kept float), OpenVINO
python YOLO/evaluate.py --weights runs/yolo26n/weights/best.pt    # P/R/mAP, box tightness, false alarms, deploy.json
python YOLO/bench.py    --weights runs/yolo26n/weights/best.pt    # CPU latency, 1/2/4 threads -> results/
python YOLO/test_boxes.py                                         # self-check: matching + tracker
```

## Pipeline and results

```bash
python pipeline.py               # frames -> classifier CSV -> YOLO boxes -> bbox_replay -> latency figure + summary
python pipeline.py --clf runs/pipeline_board/board.csv --out runs/pipeline_board   # with the board (Board/README.md)
python report.py                 # results/RESULTS.md + figures + the JSON/CSV behind it
sh run_all.sh                    # all of the above for every trained model, in order (add "train" to retrain first)
```

`YOLO/run_boxes.py` writes the boxes in the `PBS/drift/data/boxes_fake.csv` layout. `bbox_replay` is compiled with
OpenMP (Homebrew libomp) and is not modified. The board app and the PC receiver are in `../Board/`.
