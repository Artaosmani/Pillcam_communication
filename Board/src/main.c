/*
 * Polyp classifier on the nRF54LM20 DK: JPEG frame in over USB CDC-ACM (J3), MobileNetV1 0.25 INT8 on the Axon NPU,
 * one result line out per frame, plus the frame bytes when a polyp is detected.
 *
 * Protocol (PC -> board), one frame at a time:
 *   "FRM0" | u32 frame_id (LE) | u32 jpeg_len (LE) | jpeg bytes            (336x336 baseline JPEG, ~40 KB)
 * Reply (board -> PC):
 *   "R,<frame_id>,<t_first_us>,<t_rx_us>,<t_infer_start_us>,<t_infer_end_us>,<t_sent_us>,<raw>,<score>,<detected>,<n>\n"
 *   followed by n bytes (the same JPEG) when detected=1, else n=0.
 * Times are board uptime in microseconds (k_cycle_get_64): t_first = header received, t_rx = last frame byte
 * received, t_infer_start/end around the Axon call only, t_sent = just before the reply is written.
 * t_rx -> t_infer_start is the preprocessing (JPEG decode at reduced scale, resize, input LUT, HWC -> CHW).
 *
 * Model files (see ../README.md): src/generated/nrf_axon_model_polyp_.h from the Axon compiler, and
 * src/model_int8.h from Classifier/train.py (only its constants and polyp_input_lut are used).
 */
#include <string.h>
#include <math.h>
#include <zephyr/kernel.h>
#include <zephyr/device.h>
#include <zephyr/drivers/uart.h>
#include <zephyr/sys/byteorder.h>
#include <zephyr/sys/ring_buffer.h>
#include <zephyr/logging/log.h>
#include <sample_usbd.h>

#include "drivers/axon/nrf_axon_driver.h"
#include "drivers/axon/nrf_axon_nn_infer.h"
#include "generated/nrf_axon_model_polyp_.h"
#include "model_int8.h"
#include "tjpgd.h"

LOG_MODULE_REGISTER(polyp, LOG_LEVEL_INF);

#define MODEL      nrf_axon_model_polyp  /* name set by MODEL_NAME in axon/polyp_compile.yaml */
#define FRAME_SIDE 336
#define MAX_JPEG   (64 * 1024)
#define S          POLYP_INPUT_SIZE
/* 84 = 336/4: a 1/4-scale decode is the model input. 96: decode at 1/2 (168 px) and area-resize to 96. */
#define JPEG_SCALE ((S) == 84 ? 2 : 1)   /* tjpgd scale: 0 = 1/1, 1 = 1/2, 2 = 1/4, 3 = 1/8 */
#define DEC_SIDE   (FRAME_SIDE >> JPEG_SCALE)

static const struct device *const cdc = DEVICE_DT_GET_ONE(zephyr_cdc_acm_uart);

RING_BUF_DECLARE(rx_ring, 16 * 1024);
RING_BUF_DECLARE(tx_ring, 16 * 1024);
static K_SEM_DEFINE(rx_sem, 0, 1);
static K_SEM_DEFINE(tx_sem, 0, 1);

static uint8_t jpeg[MAX_JPEG];
static uint8_t decoded[DEC_SIDE * DEC_SIDE * 3];   /* RGB888 at the decode scale */
static uint8_t tjpgd_pool[4096];
static int8_t input_chw[3 * S * S];               /* Axon wants channels outermost (TFLite is HWC) */
static int8_t output[16];

/* ------------------------------------------------------------------ USB CDC-ACM */

static void uart_cb(const struct device *dev, void *user)
{
	uint8_t buf[512];

	while (uart_irq_update(dev) && uart_irq_is_pending(dev)) {
		if (uart_irq_rx_ready(dev)) {
			int n = uart_fifo_read(dev, buf, sizeof(buf));

			if (n > 0) {
				ring_buf_put(&rx_ring, buf, n); /* ring holds 16 KB; reader drains it continuously */
				k_sem_give(&rx_sem);
			}
		}
		if (uart_irq_tx_ready(dev)) {
			uint8_t *p;
			uint32_t n = ring_buf_get_claim(&tx_ring, &p, 512);

			if (!n) {
				uart_irq_tx_disable(dev);
			} else {
				ring_buf_get_finish(&tx_ring, MAX(uart_fifo_fill(dev, p, n), 0));
			}
			k_sem_give(&tx_sem);
		}
	}
}

static void read_exact(uint8_t *dst, size_t len)
{
	while (len) {
		uint32_t n = ring_buf_get(&rx_ring, dst, len);

		if (!n) {
			k_sem_take(&rx_sem, K_FOREVER);
			continue;
		}
		dst += n;
		len -= n;
	}
}

/* Queue bytes for the TX interrupt and return once all of them are handed to the USB stack. */
static void write_all(const uint8_t *src, size_t len)
{
	while (len) {
		uint32_t n = ring_buf_put(&tx_ring, src, len);

		src += n;
		len -= n;
		uart_irq_tx_enable(cdc);
		if (len || !ring_buf_is_empty(&tx_ring)) {
			k_sem_take(&tx_sem, K_MSEC(10));
		}
	}
	while (!ring_buf_is_empty(&tx_ring)) {
		k_sem_take(&tx_sem, K_MSEC(10));
	}
}

static inline uint64_t now_us(void)
{
	return k_cyc_to_us_floor64(k_cycle_get_64());
}

/* ------------------------------------------------------------------ JPEG -> model input */

struct src { const uint8_t *p; size_t left; };

static size_t jd_in(JDEC *jd, uint8_t *buf, size_t n)
{
	struct src *s = jd->device;

	n = MIN(n, s->left);
	if (buf) {
		memcpy(buf, s->p, n);
	}
	s->p += n;
	s->left -= n;
	return n;
}

static int jd_out(JDEC *jd, void *bitmap, JRECT *r)
{
	const uint8_t *px = bitmap;
	size_t w = (r->right - r->left + 1) * 3;

	for (int y = r->top; y <= r->bottom; y++, px += w) {
		memcpy(&decoded[(y * DEC_SIDE + r->left) * 3], px, w);
	}
	return 1;
}

/* Area (box-filter) resize DEC_SIDE -> S per channel, then uint8 -> int8 through the LUT, written as CHW.
 * At S = 84 DEC_SIDE is 84 too and this is a copy + LUT. */
static void to_input(void)
{
	const float k = (float)DEC_SIDE / S;

	for (int oy = 0; oy < S; oy++) {
		float y0 = oy * k, y1 = y0 + k;

		for (int ox = 0; ox < S; ox++) {
			float x0 = ox * k, x1 = x0 + k, acc[3] = {0}, area = 0;

			for (int iy = (int)y0; iy < (int)ceilf(y1); iy++) {
				float wy = MIN(y1, iy + 1) - MAX(y0, iy);

				for (int ix = (int)x0; ix < (int)ceilf(x1); ix++) {
					float wgt = wy * (MIN(x1, ix + 1) - MAX(x0, ix));
					const uint8_t *p = &decoded[(iy * DEC_SIDE + ix) * 3];

					acc[0] += wgt * p[0];
					acc[1] += wgt * p[1];
					acc[2] += wgt * p[2];
					area += wgt;
				}
			}
			for (int c = 0; c < 3; c++) {
				input_chw[c * S * S + oy * S + ox] = polyp_input_lut[(uint8_t)lroundf(acc[c] / area)];
			}
		}
	}
}

static int preprocess(size_t len)
{
	JDEC jd;
	struct src s = {jpeg, len};
	JRESULT rc = jd_prepare(&jd, jd_in, tjpgd_pool, sizeof(tjpgd_pool), &s);

	if (rc == JDR_OK) {
		if (jd.width != FRAME_SIDE || jd.height != FRAME_SIDE) {
			return -EINVAL;
		}
		rc = jd_decomp(&jd, jd_out, JPEG_SCALE);
	}
	if (rc != JDR_OK) {
		return -EIO;
	}
	to_input();
	return 0;
}

/* ------------------------------------------------------------------ main loop */

int main(void)
{
	char line[160];

	if (nrf_axon_platform_init() != 0 || nrf_axon_nn_model_validate(&MODEL) != 0) {
		LOG_ERR("Axon init / model validate failed");
		return 0;
	}
	struct usbd_context *usbd = sample_usbd_init_device(NULL);

	if (!device_is_ready(cdc) || !usbd || usbd_enable(usbd)) {
		LOG_ERR("USB CDC-ACM not ready");
		return 0;
	}
	uart_irq_callback_set(cdc, uart_cb);
	uart_irq_rx_enable(cdc);
	LOG_INF("polyp classifier ready: %dx%d input, threshold %d", S, S, POLYP_THRESHOLD_INT8);

	while (1) {
		uint8_t hdr[12];

		read_exact(hdr, 4);
		if (memcmp(hdr, "FRM0", 4)) {
			continue; /* resync byte by byte */
		}
		read_exact(hdr + 4, 8);
		uint64_t t_first = now_us();
		uint32_t fid = sys_get_le32(hdr + 4), len = sys_get_le32(hdr + 8);

		if (len > MAX_JPEG) {
			LOG_ERR("frame %u too large (%u)", fid, len);
			continue;
		}
		read_exact(jpeg, len);
		uint64_t t_rx = now_us();

		int err = preprocess(len);
		uint64_t t0 = now_us();
		int infer = err ? err : nrf_axon_nn_model_infer_sync(&MODEL, input_chw, output);
		uint64_t t1 = now_us();

		int raw = infer ? -128 : output[0];
		int det = !infer && raw >= POLYP_THRESHOLD_INT8;
		float score = 1.0f / (1.0f + expf(-(raw - POLYP_OUTPUT_ZERO_POINT) * POLYP_OUTPUT_SCALE));

		if (infer) {
			LOG_ERR("frame %u: preprocess/inference error %d", fid, infer);
		}
		uint64_t t_sent = now_us();
		int n = snprintk(line, sizeof(line), "R,%u,%llu,%llu,%llu,%llu,%llu,%d,%.4f,%d,%u\n", fid, t_first, t_rx,
				 t0, t1, t_sent, raw, (double)score, det, det ? len : 0);

		write_all(line, n);
		if (det) {
			write_all(jpeg, len);
		}
	}
	return 0;
}
