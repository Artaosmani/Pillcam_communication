# Board: polyp classifier on the nRF54LM20 DK (Axon NPU, USB CDC-ACM on J3)

Zephyr / nRF Connect SDK app. Per frame: receive a 336×336 JPEG over USB, decode it at reduced scale, run
MobileNetV1 0.25 INT8 on the Axon NPU, send one result line back, plus the frame when a polyp is detected.
`receiver.py` is the PC side and writes the classifier CSV that `Anomally_detection/pipeline.py --clf` reads.

> **Not built or run here.** The nRF Connect SDK and the Edge AI add-on aren't installed on this machine. The code
> follows the add-on's documented Axon API and Zephyr's CDC-ACM sample, but it has not been compiled. The PC side
> (`receiver.py`) is tested against a fake board (`python Board/test_receiver.py`).

## Files

| File | |
|---|---|
| `src/main.c` | USB CDC-ACM I/O, JPEG decode (TJpgDec), area resize + input LUT, Axon inference, timing, protocol |
| `prj.conf`, `app.overlay`, `CMakeLists.txt` | USB device stack "next" + one CDC-ACM UART on `zephyr_udc0` (the J3 high-speed port), Axon driver |
| `axon/polyp_compile.yaml` | input for the Axon TFLite compiler |
| `receiver.py` | PC: sends frames, reads results + echoed frames, writes the CSV with board and USB timings |
| `test_receiver.py` | self-check for `receiver.py` against a fake board |

## Protocol

PC → board: `"FRM0"`, u32 frame_id (LE), u32 length (LE), JPEG bytes. Board → PC, one line per frame:

```
R,<frame_id>,<t_first_us>,<t_rx_us>,<t_infer_start_us>,<t_infer_end_us>,<t_sent_us>,<raw_int8>,<score>,<detected>,<n>\n
```

followed by `n` bytes (the same JPEG) when `detected=1`. Times are board uptime in µs. `t_first` = header received,
`t_rx` = last byte received, `t_infer_start/end` = only the Axon call, `t_sent` = just before the reply.
`receiver.py` anchors them to the host clock and adds `usb_down_ns` (payload arriving at the board, board clock) and
`usb_ns` (host round trip minus the board's own time: result + frame back to the PC plus USB scheduling latency,
so an upper bound for board → PC).

## Steps

1. **Train and export** (`Anomally_detection/Classifier/train.py`). Copy `runs/classifier_96/model_int8.h` to
   `Board/src/model_int8.h`. Only its constants and `polyp_input_lut` are used, not the embedded `.tflite`.
   For the 84×84 model use `runs/classifier_84/` (the app picks the decode scale from `POLYP_INPUT_SIZE`).
2. **Check the ops** with the add-on's scanner (it prints `PASS` or `FAIL` with details):
   ```bash
   python axons_tflite_model_scan.py <full path>/runs/classifier_96/model_int8.tflite
   ```
   The model contains only `CONV_2D`, `DEPTHWISE_CONV_2D` and `AVERAGE_POOL_2D` (no PAD, MEAN, RESHAPE, sigmoid).
3. **Compile for Axon.** Set the `.tflite` path in `axon/polyp_compile.yaml`, then from the add-on's
   `tools/axon/compiler` (Python 3.11 env with `scripts/requirements.txt` installed):
   ```bash
   python scripts/axons_ml_nn_compiler_executor.py <path>/Board/axon/polyp_compile.yaml
   ```
   Copy the generated `nrf_axon_model_polyp_.h` to `Board/src/generated/`. Set
   `CONFIG_NRF_AXON_INTERLAYER_BUFFER_SIZE` in `prj.conf` to at least the buffer size the compiler reports.
4. **JPEG decoder.** Download TJpgDec (R0.03) from <http://elm-chan.org/fsw/tjpgd/> and put `tjpgd.c`, `tjpgd.h`
   and `tjpgdcnf.h` in `Board/src/`. In `tjpgdcnf.h` set `JD_FORMAT 0` (RGB888) and `JD_USE_SCALE 1`.
5. **Build and flash:**
   ```bash
   west build -b nrf54lm20dk/nrf54lm20b/cpuapp Board
   west flash
   ```
6. **Run.** Connect J3 (USB-C) to the PC. The CDC-ACM port shows up as `/dev/tty.usbmodem*` on macOS. Logs go to the
   debugger's VCOM, not to J3.
   ```bash
   python Board/receiver.py --port /dev/tty.usbmodemXXXX --frames Anomally_detection/Data/test/frames \
       --out Anomally_detection/runs/pipeline_board/board.csv
   python Anomally_detection/pipeline.py --clf Anomally_detection/runs/pipeline_board/board.csv --out runs/pipeline_board
   ```
   The latency summary then labels board inference and USB as **measured**.

## Check on first run

- **Input layout.** Axon stores channels outermost (CHW), TFLite innermost (HWC), so `to_input()` writes CHW.
  Compare with the generated header (and the compiler's `RESHAPE_INPUT` option). If the compiled model expects HWC,
  change the index in `to_input()`. A wrong layout still runs but gives wrong scores. Compare the raw int8 outputs
  for a few frames with `Classifier/run_csv.py` on the PC; they should match within ±1.
- **Output.** `output[0]` is read as the int8 logit. Check the output buffer type and size in the generated header.
- **High-speed USB** is marked experimental by Nordic. If enumeration or transfers fail, fall back to full speed,
  e.g. `maximum-speed = "full-speed";` on `&zephyr_udc0` in `app.overlay` (check the property in your SDK's UDC
  binding). Full speed is 12 Mbit/s, so a 40 KB frame takes ≥ 27 ms on the wire each way.
- **Timing.** Timestamps use `k_cycle_get_64()` (µs resolution). Inference time is the wall time of the synchronous
  Axon call. Nordic's figure for this architecture at 96×96 is about 14.4 ms.

## Memory

RX JPEG buffer 64 KB, decoded image 85 KB at 96 (168×168 RGB, 1/2-scale decode) or 21 KB at 84, model input 27 KB,
USB rings 32 KB, Axon interlayer buffer as set in `prj.conf`. The `.tflite` peak activations are 54 KB.
Together this is well under the 512 KB RAM.
