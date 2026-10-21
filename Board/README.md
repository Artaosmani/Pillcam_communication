# Board: polyp classifier on the nRF54LM20 DK (Axon NPU, USB CDC-ACM)

Zephyr / nRF Connect SDK app. Per frame: receive a frame over USB (a 336×336 JPEG, or 84×84 raw RGB), turn it into
the 84×84 int8 model input, run MobileNetV1 0.25 INT8 on the Axon NPU, send one result line back, plus the frame when
a polyp is detected. `receiver.py` is the PC side and writes the classifier CSV that
`Anomally_detection/pipeline.py --clf` reads.

Built and run on an nRF54LM20 DK (nRF54LM20**B**) with nRF Connect SDK **v3.4.1**, Zephyr v4.4.2 and the
**Edge AI add-on v3.0.0** (Axon NN compiler 2.0.1). Measured results: `Anomally_detection/results/RESULTS.md`.

## Files

| File | |
|---|---|
| `src/main.c` | USB CDC-ACM I/O, JPEG decode (TJpgDec) or raw input, input LUT, Axon inference, timing, protocol |
| `src/model_int8.h` | copied from `Anomally_detection/runs/classifier_84/`: quantisation constants, threshold, input LUT |
| `src/generated/nrf_axon_model_polyp_.h` | the model compiled for Axon (step 3) |
| `src/tjpgd.c`, `tjpgd.h`, `tjpgdcnf.h` | TJpgDec R0.03 from elm-chan.org, unmodified apart from `tjpgdcnf.h` |
| `prj.conf`, `app.overlay`, `Kconfig`, `CMakeLists.txt` | USB device stack "next" + one CDC-ACM UART on `zephyr_udc0`, Axon driver |
| `axon/polyp_compile.yaml` | input for the Axon compiler |
| `receiver.py` | PC: sends frames, reads results + echoed frames, writes the CSV with board and USB timings |
| `test_receiver.py` | self-check for `receiver.py` (both modes) against a fake board |

## Protocol

PC → board, one frame at a time, all integers little-endian:

| Message | Payload | Board preprocessing |
|---|---|---|
| `"FRM0"`, u32 frame_id, u32 length | 336×336 baseline JPEG | TJpgDec at 1/4 scale → 84×84 RGB, LUT, HWC → CHW |
| `"RAW0"`, u32 frame_id, u32 length = 21168 | 84×84 RGB888, row-major | LUT, HWC → CHW only |

`receiver.py --raw` sends `RAW0`: it decodes the JPEG on the PC and area-resizes it to 84×84 (exact 4×4 block mean,
the resize used in training). Board → PC, one line per frame:

```
R,<frame_id>,<t_first_us>,<t_rx_us>,<t_infer_start_us>,<t_infer_end_us>,<t_sent_us>,<raw_int8>,<score>,<detected>,<n>,
  <rx_reads>,<rx_gap_max_us>,<rx_pauses>,<rx_busy_pct>\n
```

followed by `n` bytes (the same payload) when `detected=1`. The last four fields are receive diagnostics for the
frame (see "Known issue"); `receiver.py` ignores them.

## How each time is measured

Board times use `k_cycle_get_64()` on the 1 MHz system clock (1 µs resolution): `t_first` = 12-byte header received,
`t_rx` = last payload byte received, `t_infer_start/end` = around `nrf_axon_nn_model_infer_sync()` only,
`t_sent` = just before the reply is written. `receiver.py` turns them into:

| CSV column / stage | Definition | Clock |
|---|---|---|
| `usb_down_ns` (PC → board) | `t_rx − t_first`: payload transfer after the header | board |
| board preprocessing | `t_infer_start − t_rx` | board |
| Axon inference | `t_infer_end − t_infer_start` | board |
| `usb_ns` (board → PC) | host round trip − (`t_sent − t_first`): reply + echoed frame + USB scheduling both ways, so an upper bound | host |

The board and host clocks are never compared directly; board times are only used as differences, anchored at the
host send time.

## Steps

All commands from a shell with the nRF Connect SDK toolchain on `PATH`. If `west build` can't find the Zephyr SDK,
set `ZEPHYR_SDK_INSTALL_DIR=<toolchain>/opt/zephyr-sdk` and `ZEPHYR_TOOLCHAIN_VARIANT=zephyr` (the nRF Connect
terminal / VS Code extension does this for you).

1. **Train and export** (`Anomally_detection/Classifier/train.py --set imgsz=84 --name classifier_84`), then
   `cp Anomally_detection/runs/classifier_84/model_int8.h Board/src/`. Only its constants and `polyp_input_lut`
   are used.
2. **Check the ops** with the add-on's scanner, from `edge-ai/tools/axon/compiler/scripts` in a Python 3.11 env with
   `requirements.txt` installed (TensorFlow 2.19; keep it separate from the training env):
   ```bash
   python axons_tflite_model_scan.py <repo>/Anomally_detection/runs/classifier_84/model_int8.tflite   # -> PASS
   ```
3. **Compile for Axon.** Put the absolute `.tflite` path in `axon/polyp_compile.yaml` (keys are lowercase and nested
   under a test name, see the add-on's `compiler_input_yaml_template.yaml`), then from the same directory:
   ```bash
   python axons_ml_nn_compiler_executor.py <repo>/Board/axon/polyp_compile.yaml
   cp outputs/nrf_axon_model_polyp_.h <repo>/Board/src/generated/
   ```
   The symbol is `model_<model_name>` (`model_polyp`). The compiler prints the interlayer buffer it needs (35,952 B
   for this model); `CONFIG_NRF_AXON_INTERLAYER_BUFFER_SIZE` must be at least that (a `static_assert` checks it).
4. **JPEG decoder.** Use the original TJpgDec R0.03 from <http://elm-chan.org/fsw/tjpgd/>, with `JD_FORMAT 0`
   (RGB888), `JD_USE_SCALE 1`, `JD_FASTDECODE 1` in `tjpgdcnf.h`. **Not** the copy that ships with LVGL: it writes
   B,G,R and has no descaling, so the board sees scrambled BGR images and calls every polyp negative.
5. **Build and flash** through the DK's debugger port (labelled J4 on our DK; it enumerates as SEGGER J-Link):
   ```bash
   west build -b nrf54lm20dk/nrf54lm20b/cpuapp Board      # B variant: the A variant has no Axon
   west flash --runner jlink                              # works without nrfutil
   ```
   Logs (115200 baud) come out on the debugger's second VCOM port. Before the nRF USB port is connected the app
   prints `VBUS is not ready, block udc_enable()` and waits.
6. **Run.** Move the cable to the nRF54's own USB-C port (the DK has room for one cable at a time on our setup). It
   enumerates as "Zephyr Project", 480 Mbit/s, e.g. `/dev/tty.usbmodem1401` on macOS.
   ```bash
   python Board/receiver.py --port /dev/tty.usbmodem1401 --frames Anomally_detection/Data/test/frames \
       --out Anomally_detection/runs/pipeline_board/board.csv            # JPEG
   python Board/receiver.py ... --out Anomally_detection/runs/pipeline_board_raw/board.csv --raw   # raw 84×84
   python Anomally_detection/pipeline.py --clf Anomally_detection/runs/pipeline_board/board.csv \
       --out Anomally_detection/runs/pipeline_board
   python Anomally_detection/report.py      # adds the board section to results/RESULTS.md
   ```

## Checks we did, and what they found

- **Toolchain + NPU:** Nordic's `edge-ai/samples/axon/hello_axon` built, flashed and printed predictions matching
  its reference values before our app was tried.
- **Input layout:** the compiled header gives the input as 84×84×3 with row stride 84, i.e. channel-planar (CHW),
  which is what `to_input()` writes. Output: one int8 value, zero-point 24, same as the `.tflite`.
- **Board vs PC:** run the same frames through `Anomally_detection/Classifier/run_csv.py` (PC TFLite) and compare.
  On the 801 test frames the board takes the same decision on 800 (JPEG); raw outputs differ by 1–4 int8 steps
  because TJpgDec's block averaging is not bit-identical to TensorFlow's decode + area resize.
- **Echo integrity:** `receiver.py` checks every echoed frame is byte-identical to what it sent.
- **Full speed:** `maximum-speed` is not a property of this controller (`snps,dwc2`). Force full speed with
  `CONFIG_UDC_DRIVER_HIGH_SPEED_SUPPORT_ENABLED=n` and `CONFIG_USBD_MAX_SPEED_FULL=y`. High speed works; full speed
  was only needed while debugging.

## Known issue: PC → board transfer is slow

The payload arrives at **≈ 1.6 ms per 512-byte USB packet** (≈ 67 ms for a 20 KB JPEG, ≈ 68 ms for a raw frame),
far below the 480 Mbit/s link rate; time is exactly proportional to size. The host's `write()` blocks for the same
time, so the board is accepting packets slowly. The reply line carries RX diagnostics for each frame
(`rx_reads`, `rx_gap_max_us`, `rx_pauses`, `rx_busy_pct`). On 40 frames per mode they show:

| | JPEG | raw |
|---|---|---|
| receive time (median) | 68.2 ms | 69.8 ms |
| `uart_fifo_read` calls with data per frame | 322 | 330 |
| bytes per read | ≈ 64 | ≈ 64 |
| longest gap between reads | 0.39 ms | 0.39 ms |
| RX pauses (ring full) | 0 | 0 |
| **CPU busy while receiving** | **100 %** | **100 %** |

So the board CPU is saturated for the whole transfer, ≈ 0.2 ms per 64-byte read: the host and the wire are not the
limit, and the back-pressure in `uart_cb` never triggers. Also ruled out: the PC write pattern (whole frame vs
512 B chunks) and the UDC driver's INF logging (`CONFIG_UDC_DRIVER_LOG_LEVEL_WRN` changed nothing). The HS
descriptors declare 512-byte bulk packets, so why the data arrives in 64-byte pieces is open. Next step: per-thread
CPU time (main, system work queue, DWC2 driver thread, USBD thread) over the receive window to see which one burns
it. Board → PC is not affected (30 KB in ≈ 30 ms including scheduling).

## Memory

RAM 292 KB of 511 KB: RX JPEG buffer 64 KB, decoded 84×84 RGB 21 KB, model input 21 KB, USB rings 32 KB, Axon
interlayer buffer 120 KB (`prj.conf`; 36 KB needed). Flash 358 KB of 2 MB (model constants 208 KB).
