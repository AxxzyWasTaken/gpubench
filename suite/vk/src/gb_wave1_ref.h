// SPDX-License-Identifier: GPL-2.0-only
/* Shared input generators and CPU reference oracles for the 90%-coverage
 * Wave-1 benchmarks. Compiled into both gpubench-vk and gpubench-mtl so
 * both executors run identical inputs and check against the exact same
 * expected CRC-32 values. */
#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "gb_common.h"

/* ---- compute.uber ---- */
enum gb_uber_kind {
   GB_UBER_DIV = 0,
   GB_UBER_UNI = 1,
   GB_UBER_ILP = 2,
   GB_UBER_SRAM = 3,
};

void gb_uber_fill_ilp_input(uint32_t *buf, size_t nwords);
uint32_t gb_uber_ref(enum gb_uber_kind kind, const uint32_t *ilp_in, uint32_t *out, uint32_t nthreads);

/* ---- compute.conv2d ---- */
void gb_conv2d_fill_input(uint32_t *img, uint32_t w, uint32_t h);
uint32_t gb_conv2d_ref(bool bilateral, const uint32_t *in, uint32_t *out, uint32_t w, uint32_t h);

/* ---- compute.stereo ---- */
void gb_stereo_fill_inputs(uint32_t *left_packed, uint32_t *right_packed, uint32_t disp);
uint32_t gb_stereo_ref(const uint32_t *left_packed, const uint32_t *right_packed, uint32_t disp, uint32_t *out);

/* ---- compute.edge ---- */
void gb_edge_fill_input(uint32_t *img, uint32_t w, uint32_t h);
uint32_t gb_edge_ref(bool harris, const uint32_t *in, uint32_t *out, uint32_t *corner_count_out,
                     uint32_t w, uint32_t h);

/* ---- render.mdi ---- */
uint32_t gb_mdi_tex_color(uint32_t tex_id);
void gb_mdi_fill_tex(uint32_t tex_id, uint32_t *rgba16x16);
uint32_t gb_mdi_ref_fb(uint32_t draws, uint32_t bindless, uint32_t *fb_rgba8);

/* ---- render.deferred ---- */
/* Verifies the rendered RGBA8 framebuffer against the analytical CPU lighting
 * model at 64 interior grid sample points (within +-3/255 tolerance for FP16
 * normal / MSAA resolve rounding) and computes its CRC-32. */
void gb_deferred_check_fb(const uint8_t *rgba8, uint32_t w, uint32_t h, struct gb_check *chk);
