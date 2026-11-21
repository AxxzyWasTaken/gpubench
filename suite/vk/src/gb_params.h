// SPDX-License-Identifier: GPL-2.0-only
/* Workload dimensions shared by the Vulkan and Metal executors. Changing a
 * value here changes the benchmark: bump the result's meaning in README. */
#pragma once

#define GB_LIGHT_LOCAL 32u              /* compute.light: 1 group of 32 */
#define GB_ALU_GROUPS 1024u             /* compute.heavy kind=alu */
#define GB_ALU_LOCAL 256u
#define GB_ALU_ITERS 128u               /* x 32 FMA per iteration */
#define GB_ALU_A 0.999f
#define GB_ALU_B 0.001f
#define GB_ALU_FLOP ((double)GB_ALU_GROUPS * GB_ALU_LOCAL * GB_ALU_ITERS * 32.0 * 2.0)
#define GB_BW_BYTES (64ull << 20)       /* compute.heavy kind=bw: read + write */
#define GB_BW_LOCAL 256u
#define GB_LIGHT_DIM 64u                /* render.light target */
#define GB_FILL_LAYERS 16u              /* render.heavy kind=fill */
#define GB_TEX_DIM 256u                 /* render.heavy kind=texture */
#define GB_GEOM_TRIS 262144u            /* render.heavy kind=geom */
#define GB_TVB_W 1920u                  /* render.tvb target */
#define GB_TVB_H 1080u
#define GB_COPY_BYTES (64ull << 20)     /* copy.bw */
#define GB_COPY_DIM 4096u               /* copy.bw kind=buf2img, RGBA8 */
#define GB_QRING 128u                   /* query ring */
#define GB_HOST_SIGNAL_DELAY_NS 200000ull /* sync.signal kind=host: park time before the signal */
#define GB_PING_RING 8                  /* sync.signal kind=queue: rounds in flight */
#define GB_CR_DEPTH 2                   /* concurrent.cr mode=queues: per-queue depth */
#define GB_IL_DEPTH 4                   /* concurrent.cr mode=interleave */
#define GB_TOUCH_STRIDE 16384           /* mem.*: one write per 16 KiB page */
#define GB_IDLE_BUDGET_MS 1500           /* idle.wake: samples = budget / gap_ms, clamped */
#define GB_IDLE_MIN_SAMPLES 6
#define GB_IDLE_MAX_SAMPLES 40

/* ---- 90%-coverage Wave-1 workload dimensions ---- */
#define GB_UBER_GROUPS 256u              /* compute.uber: 256 x 256 = 65536 invocations */
#define GB_UBER_LOCAL 256u
#define GB_UBER_THREADS (GB_UBER_GROUPS * GB_UBER_LOCAL)
#define GB_UBER_ITERS 48u                /* narrow post-branch loop iterations */
#define GB_UBER_ILP_WORDS (4u << 20)     /* 16 MiB input table for kind=ilp */
#define GB_UBER_SEED 0x13579bdfu

#define GB_CONV_LOCAL 16u                /* compute.conv2d: 16x16 cooperative halo tile */
#define GB_CONV_RADIUS 8u                /* 17-tap separable filter */

#define GB_STEREO_W 450u                 /* compute.stereo: matches Geekbench Stereo */
#define GB_STEREO_H 375u
#define GB_STEREO_PITCH 512u
#define GB_STEREO_RADIUS 5u              /* 11x11 SAD window */
#define GB_STEREO_LOCAL 16u

#define GB_EDGE_LOCAL 16u                /* compute.edge: 16x16 + 1px halo = 18x18 tile */

#define GB_DEFERRED_QUADS 4096u          /* render.deferred: 4096 quads = 8192 tris, 4 layers */

#define GB_MDI_DIM 1024u                 /* render.mdi: 1024x1024 RGBA8 target */
#define GB_MDI_TEX_DIM 16u               /* 16x16 bindless textures */
#define GB_MDI_MAX_BINDLESS 1024u
#define GB_MDI_MAX_DRAWS 10000u

#define GB_PSO_DEFAULT_COUNT 32u         /* compile.pso: default variant batch */
