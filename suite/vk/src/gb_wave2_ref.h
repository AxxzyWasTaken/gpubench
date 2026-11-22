// SPDX-License-Identifier: GPL-2.0-only
/* Shared input generators and deterministic CPU reference oracles for the
 * 90%-coverage Wave-2 heavy-game / AAA surface benchmarks. Compiled into
 * gpubench-vk, gpubench-mtl and gpubench-gl so every executor executes
 * identical math and validates against the exact same expected outputs. */
#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "gb_common.h"

/* ---- 1. render.shadow ---- */
void gb_shadow_check_fb(const uint8_t *rgba8, uint32_t w, uint32_t h, bool alpha_test, struct gb_check *chk);

/* ---- 2. compute.cull_mdi ---- */
struct gb_cull_obj {
   float cx, cy, r;
   uint32_t color_rgba8;
};
uint32_t gb_cull_mdi_fill_objects(uint32_t objs, uint32_t visible_pct, struct gb_cull_obj *out);
uint32_t gb_cull_mdi_ref_fb(const struct gb_cull_obj *objs, uint32_t n_objs, uint32_t *fb_rgba8);

/* ---- 3. render.bindless_ubo ---- */
void gb_bubo_fill_bank(uint32_t bank_id, float *rgba_1024);
uint32_t gb_bubo_ref_fb(uint32_t draws, uint32_t *fb_rgba8);

/* ---- 4. compute.fp16 ---- */
void gb_fp16_gemv_fill(uint32_t *w_packed, uint32_t *x_fp16_pairs);
uint32_t gb_fp16_gemv_ref(const uint32_t *w_packed, const uint32_t *x_fp16_pairs, float *out_m);
uint32_t gb_fp16_alumix_ref(uint32_t nthreads, uint32_t iters, float a, float b, float *out);

/* ---- 5. transfer.ops ---- */
uint32_t gb_transfer_clear_color_u32(void);
uint32_t gb_transfer_clear_ref_crc(uint32_t w, uint32_t h);
void gb_transfer_fill_rgba(uint32_t w, uint32_t h, uint32_t *rgba);
void gb_transfer_fill_mip0(uint32_t w, uint32_t h, uint32_t *rgba);
void gb_transfer_check_mipchain(const uint8_t *mip4_rgba, uint32_t m4w, uint32_t m4h,
                                const uint8_t *mip_last_rgba, struct gb_check *chk);

/* ---- 6. render.postfx ---- */
void gb_postfx_check_fb(const uint8_t *rgba8, uint32_t w, uint32_t h, struct gb_check *chk);
