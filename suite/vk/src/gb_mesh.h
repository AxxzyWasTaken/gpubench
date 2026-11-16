// SPDX-License-Identifier: GPL-2.0-only
/* render.mesh workload definition shared by the Vulkan and Metal executors:
 * parameters, counts, the colour hash and the topo=line verifier. The
 * shaders (mesh.mesh/.task/.frag and mesh.metal) implement the same
 * definitions; suite/README.md "render.mesh" documents them. */
#pragma once

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gb_common.h"
#include "gb_params.h"

struct gb_mesh_cfg {
   bool task, line, vert, perprim;
   uint32_t wg, views, groups, res;
   uint32_t nv, np;            /* outputs per mesh workgroup */
   uint32_t gw, gh;            /* topo=tri: vertex grid of a tile */
   uint32_t tiles_x, tiles_total;
};

static const char *const gb_mesh_params[] = {"stage", "wg", "shape", "perprim", "views", "topo", "groups", "res", NULL};

/* Parses and validates the parameters; returns an error or NULL. */
static inline const char *
gb_mesh_parse(const struct gb_opts *o, struct gb_mesh_cfg *c)
{
   memset(c, 0, sizeof(*c));
   const char *stage = gb_param(o, "stage", "mesh"), *shape = gb_param(o, "shape", "prim");
   const char *topo = gb_param(o, "topo", "tri");
   long wg = gb_param_int(o, "wg", 32), pp = gb_param_int(o, "perprim", 0), views = gb_param_int(o, "views", 1);
   long groups = gb_param_int(o, "groups", GB_MESH_GROUPS), res = gb_param_int(o, "res", GB_MESH_RES);
   if (strcmp(stage, "mesh") && strcmp(stage, "task"))
      return "stage must be mesh or task";
   if (wg != 32 && wg != 128)
      return "wg must be 32 or 128";
   if (strcmp(shape, "prim") && strcmp(shape, "vert"))
      return "shape must be prim or vert";
   if (strcmp(topo, "tri") && strcmp(topo, "line"))
      return "topo must be tri or line";
   if (pp != 0 && pp != 1)
      return "perprim must be 0 or 1";
   if (views != 1 && views != 2)
      return "views must be 1 or 2";
   if (groups < 1 || groups > 65535)
      return "groups must be 1..65535";
   if (res < 64 || res > 8192)
      return "res must be 64..8192";
   c->task = !strcmp(stage, "task");
   c->line = !strcmp(topo, "line");
   c->vert = !c->line && !strcmp(shape, "vert");
   c->perprim = pp;
   c->wg = (uint32_t)wg;
   c->views = (uint32_t)views;
   c->groups = (uint32_t)groups;
   c->res = (uint32_t)res;
   if (c->task && c->groups % GB_MESH_FANOUT)
      return "stage=task needs groups to be a multiple of 32";
   if (c->line) {
      c->nv = 2 * c->wg;
      c->np = c->wg;
   } else if (c->vert) {
      /* capped at 128: Metal limits a metal::mesh to 16 KB (128 x 96 B) */
      c->nv = 2 * c->wg < 128 ? 2 * c->wg : 128;
      c->np = c->nv / 4;
   } else {
      c->nv = c->wg;
      c->np = 2 * c->wg;
   }
   c->gw = c->wg == 32 ? 8 : 16;
   c->gh = c->wg / c->gw;
   c->tiles_x = c->res / (c->gw * GB_MESH_CELL);
   c->tiles_total = c->tiles_x * (c->res / (c->gh * GB_MESH_CELL));
   if (!c->line && !c->tiles_total)
      return "res too small for one tile";
   if (o->check_each && !c->line)
      return "--check-each supports topo=line only";
   if (o->check_each && o->mode == GB_MODE_QUEUES)
      return "--check-each needs mode queue or procs";
   return NULL;
}

/* lowbias32, as gb_hash() in the shaders */
static inline uint32_t
gb_mesh_hash(uint32_t x)
{
   x ^= x >> 16;
   x *= 0x7feb352du;
   x ^= x >> 15;
   x *= 0x846ca68bu;
   x ^= x >> 16;
   return x;
}

/* Colour of workgroup g in view v as RGBA8 bytes R, G, B, A in a little-endian
 * word (the target's memory order): R = h & 255, G = h >> 8 & 255,
 * B = h >> 16 & 255, A = 255, h = hash(2g + v). */
static inline uint32_t
gb_mesh_color(uint32_t g, uint32_t view)
{
   return (gb_mesh_hash(g * 2u + view) & 0xffffffu) | 0xff000000u;
}

/* Primitives rasterized per draw (all views). */
static inline double
gb_mesh_prims(const struct gb_mesh_cfg *c)
{
   return (double)c->groups * c->np * c->views;
}

static inline bool
gb_mesh_near(uint32_t a, uint32_t b)
{
   for (int i = 0; i < 32; i += 8) {
      int d = (int)((a >> i) & 255) - (int)((b >> i) & 255);
      if (d < -1 || d > 1)
         return false;
   }
   return true;
}

struct gb_mesh_ver {
   struct gb_mesh_cfg c;
   uint32_t *expect[2];        /* colour per workgroup and view */
   uint8_t *owns, *lit;        /* scratch, one per workgroup */
};

static inline void
gb_mesh_ver_init(struct gb_mesh_ver *v, const struct gb_mesh_cfg *c)
{
   memset(v, 0, sizeof(*v));
   v->c = *c;
   for (uint32_t w = 0; w < c->views; w++) {
      v->expect[w] = malloc(c->groups * sizeof(uint32_t));
      for (uint32_t g = 0; g < c->groups; g++)
         v->expect[w][g] = gb_mesh_color(g, w);
   }
   v->owns = malloc(c->groups);
   v->lit = malloc(c->groups);
}

static inline void
gb_mesh_ver_free(struct gb_mesh_ver *v)
{
   free(v->expect[0]);
   free(v->expect[1]);
   free(v->owns);
   free(v->lit);
   memset(v, 0, sizeof(*v));
}

/* topo=line: `px` holds every layer of the target (views x res^2 words).
 * Pixel i of view v must hold the colour of the last workgroup whose span
 * covers it (span of g: lin = g*wg .. g*wg + wg-1, pixel lin mod res^2), each
 * channel within +-1, or the clear colour 0 (exactly) where no span reaches.
 *   bad            pixels that do not
 *   wrong_view     bad pixels holding the other view's colour of their group
 *   missing_blocks (view, group) pairs owning >= 1 pixel, all still clear
 * `log` > 0 prints that many bad pixels to stderr. */
static inline void
gb_mesh_verify(struct gb_mesh_ver *v, const uint32_t *px, int log, struct gb_check *out)
{
   const struct gb_mesh_cfg *c = &v->c;
   const uint64_t npix = (uint64_t)c->res * c->res, total = (uint64_t)c->groups * c->wg;
   memset(out, 0, sizeof(*out));
   out->checked = 1;
   for (uint32_t w = 0; w < c->views; w++) {
      memset(v->owns, 0, c->groups);
      memset(v->lit, 0, c->groups);
      const uint32_t *L = px + w * npix;
      for (uint64_t i = 0; i < npix; i++) {
         uint32_t want = 0, got = L[i];
         int64_t g = -1;
         if (i < total) {
            uint64_t lin = i + (total - 1 - i) / npix * npix;
            g = (int64_t)(lin / c->wg);
            want = v->expect[w][g];
            v->owns[g] = 1;
            if (got != 0)
               v->lit[g] = 1;
         }
         out->pixels++;
         if (g >= 0 ? gb_mesh_near(got, want) : got == want)
            continue;
         out->bad++;
         if (g >= 0 && c->views > 1 && gb_mesh_near(got, v->expect[1 - w][g]))
            out->wrong_view++;
         if (log > 0) {
            gb_log("render.mesh check: view %u pixel (%llu,%llu) = 0x%08x, want 0x%08x", w,
                   (unsigned long long)(i % c->res), (unsigned long long)(i / c->res), got, want);
            log--;
         }
      }
      for (uint32_t g = 0; g < c->groups; g++)
         if (v->owns[g] && !v->lit[g])
            out->missing_blocks++;
   }
   out->pass = out->bad == 0;
}

/* --check-each: adds one submission's result to the totals. */
static inline void
gb_mesh_each_add(struct gb_check *each, const struct gb_check *one)
{
   each->checked = 1;
   each->iters++;
   each->bad_iters += !one->pass;
   each->pixels += one->pixels;
   each->bad += one->bad;
   each->missing_blocks += one->missing_blocks;
   each->wrong_view += one->wrong_view;
   each->pass = each->bad_iters == 0;
}

/* The "workload" object of the result's "pipeline" field. */
static inline void
gb_mesh_json_workload(struct gb_json *j, const struct gb_mesh_cfg *c, const char *multiview)
{
   gb_json_obj_begin(j, "workload");
   gb_json_int(j, "task_groups", c->task ? c->groups / GB_MESH_FANOUT : 0);
   gb_json_int(j, "task_threads", c->task ? GB_MESH_FANOUT : 0);
   gb_json_int(j, "task_payload_bytes", c->task ? 4 : 0);
   gb_json_int(j, "mesh_groups", c->groups);
   gb_json_int(j, "mesh_threads", c->wg);
   gb_json_int(j, "max_vertices", c->nv);
   gb_json_int(j, "max_primitives", c->np);
   gb_json_int(j, "views", c->views);
   gb_json_str(j, "multiview", multiview);
   gb_json_int(j, "tiles_x", c->tiles_x);
   gb_json_int(j, "tiles_total", c->tiles_total);
   gb_json_obj_end(j);
}
