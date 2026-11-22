// SPDX-License-Identifier: GPL-2.0-only
/* 90%-coverage Wave-1 Vulkan benchmarks:
 *   compute.uber, compute.conv2d, compute.stereo, compute.edge,
 *   render.deferred, concurrent.sram, render.mdi, compile.pso
 * Mirrors suite/metal/main.m with identical dimensions (gb_params.h) and
 * shared CPU reference oracles (gb_wave1_ref.h). */
#define _GNU_SOURCE
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "gb_params.h"
#include "gb_vk.h"
#include "gb_wave1_ref.h"
#include "shaders.h"
#include "tests.h"

static bool
parse_res_w1(const char *s, uint32_t *w, uint32_t *h)
{
   if (!strcmp(s, "720p")) *w = 1280, *h = 720;
   else if (!strcmp(s, "1080p")) *w = 1920, *h = 1080;
   else if (!strcmp(s, "1440p")) *w = 2560, *h = 1440;
   else if (!strcmp(s, "2160p")) *w = 3840, *h = 2160;
   else return false;
   return true;
}

static void
compute_barrier(VkCommandBuffer cb)
{
   VkMemoryBarrier2 mb = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
      .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
      .srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
      .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
      .dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
   };
   VkDependencyInfo di = {
      .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
      .memoryBarrierCount = 1,
      .pMemoryBarriers = &mb,
   };
   vkCmdPipelineBarrier2(cb, &di);
}

static void
compute_pipe_spec(struct gb_vk *vk, const uint32_t *code, size_t bytes, int nbuf, uint32_t push_size,
                  const VkSpecializationInfo *spec, VkPipelineCache cache, struct gb_pipe *p)
{
   memset(p, 0, sizeof(*p));
   VkDescriptorSetLayoutBinding binds[8];
   for (int i = 0; i < nbuf; i++)
      binds[i] = (VkDescriptorSetLayoutBinding){
         .binding = (uint32_t)i,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .descriptorCount = 1,
         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      };
   if (nbuf) {
      VkDescriptorSetLayoutCreateInfo dci = {
         .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
         .bindingCount = (uint32_t)nbuf,
         .pBindings = binds,
      };
      GB_VK_CHECK(vkCreateDescriptorSetLayout(vk->dev, &dci, NULL, &p->dsl));
      VkDescriptorPoolSize ps = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, (uint32_t)nbuf};
      VkDescriptorPoolCreateInfo pci = {
         .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
         .maxSets = 1,
         .poolSizeCount = 1,
         .pPoolSizes = &ps,
      };
      GB_VK_CHECK(vkCreateDescriptorPool(vk->dev, &pci, NULL, &p->dpool));
      VkDescriptorSetAllocateInfo sai = {
         .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
         .descriptorPool = p->dpool,
         .descriptorSetCount = 1,
         .pSetLayouts = &p->dsl,
      };
      GB_VK_CHECK(vkAllocateDescriptorSets(vk->dev, &sai, &p->set));
   }
   VkPushConstantRange pr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, push_size};
   VkPipelineLayoutCreateInfo lci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = nbuf ? 1u : 0u,
      .pSetLayouts = &p->dsl,
      .pushConstantRangeCount = push_size ? 1u : 0u,
      .pPushConstantRanges = &pr,
   };
   GB_VK_CHECK(vkCreatePipelineLayout(vk->dev, &lci, NULL, &p->layout));
   p->push_stages = VK_SHADER_STAGE_COMPUTE_BIT;
   p->push_size = push_size;
   VkShaderModule m = gb_shader(vk, code, bytes);
   VkComputePipelineCreateInfo ci = {
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage = {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_COMPUTE_BIT,
         .module = m,
         .pName = "main",
         .pSpecializationInfo = spec,
      },
      .layout = p->layout,
   };
   GB_VK_CHECK(vkCreateComputePipelines(vk->dev, cache, 1, &ci, NULL, &p->pipe));
   vkDestroyShaderModule(vk->dev, m, NULL);
}

/* =========================================================================
 * 1. compute.uber (divergent | uniform | ilp) + helper for concurrent.sram
 * ========================================================================= */

struct uber_state {
   enum gb_uber_kind kind;
   struct gb_pipe p;
   struct gb_buf in, out;
   uint32_t *ilp_cpu;
   uint32_t *ref;
   uint32_t ref_crc;
};

static struct uber_state *
uber_setup_kind(struct gb_vk *vk, enum gb_uber_kind kind)
{
   struct uber_state *st = calloc(1, sizeof(*st));
   st->kind = kind;
   size_t out_bytes = (size_t)GB_UBER_THREADS * sizeof(uint32_t);
   gb_buf_create(vk, out_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 false, &st->out);
   if (kind == GB_UBER_ILP) {
      size_t in_bytes = (size_t)GB_UBER_ILP_WORDS * sizeof(uint32_t);
      st->ilp_cpu = malloc(in_bytes);
      gb_uber_fill_ilp_input(st->ilp_cpu, GB_UBER_ILP_WORDS);
      gb_buf_create(vk, in_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    false, &st->in);
      gb_buf_upload(vk, st->ilp_cpu, &st->in, in_bytes);
      gb_compute_pipe(vk, GB_SPV(uber_ilp_comp), 2, 8, &st->p);
      struct gb_buf *b[2] = {&st->in, &st->out};
      gb_pipe_set_buffers(vk, &st->p, b, 2);
   } else {
      const uint32_t *code = kind == GB_UBER_DIV ? uber_div_comp
                             : kind == GB_UBER_UNI ? uber_uni_comp
                                                   : uber_sram_comp;
      size_t sz = kind == GB_UBER_DIV ? uber_div_comp_size
                  : kind == GB_UBER_UNI ? uber_uni_comp_size
                                        : uber_sram_comp_size;
      gb_compute_pipe(vk, code, sz, 1, 8, &st->p);
      struct gb_buf *b[1] = {&st->out};
      gb_pipe_set_buffers(vk, &st->p, b, 1);
   }
   st->ref = malloc(out_bytes);
   st->ref_crc = gb_uber_ref(kind, st->ilp_cpu, st->ref, GB_UBER_THREADS);
   return st;
}

static void *
uber_setup(struct gb_vk *vk, const struct gb_opts *o, char *err, size_t n)
{
   const char *kind = gb_param(o, "kind", "divergent");
   if (!strcmp(kind, "divergent")) return uber_setup_kind(vk, GB_UBER_DIV);
   if (!strcmp(kind, "uniform"))   return uber_setup_kind(vk, GB_UBER_UNI);
   if (!strcmp(kind, "ilp"))       return uber_setup_kind(vk, GB_UBER_ILP);
   snprintf(err, n, "kind must be divergent, uniform or ilp");
   return NULL;
}

static void
uber_teardown(struct gb_vk *vk, void *p)
{
   struct uber_state *st = p;
   gb_pipe_destroy(vk, &st->p);
   gb_buf_destroy(vk, &st->out);
   if (st->in.buf) gb_buf_destroy(vk, &st->in);
   free(st->ilp_cpu);
   free(st->ref);
   free(st);
}

static void
uber_record(void *p, VkCommandBuffer cb, uint64_t k)
{
   (void)k;
   struct uber_state *st = p;
   vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, st->p.pipe);
   vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, st->p.layout, 0, 1, &st->p.set, 0, NULL);
   struct { uint32_t a, b; } pc = {
      GB_UBER_SEED,
      st->kind == GB_UBER_ILP ? (GB_UBER_ILP_WORDS - 1u) : GB_UBER_ITERS,
   };
   vkCmdPushConstants(cb, st->p.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
   vkCmdDispatch(cb, GB_UBER_GROUPS, 1, 1);
}

static void
uber_derive(const struct gb_opts *o, struct gb_result *r, const struct gb_samples *us)
{
   (void)o;
   gb_derive_rate(r, "mthreads_s", "Mthreads/s", us, (double)GB_UBER_THREADS);
}

static void
uber_check(struct gb_vk *vk, void *p, struct gb_check *chk)
{
   struct uber_state *st = p;
   uint32_t *got = malloc((size_t)GB_UBER_THREADS * sizeof(uint32_t));
   gb_buf_readback(vk, &st->out, got, (size_t)GB_UBER_THREADS * sizeof(uint32_t));
   gb_check_words(chk, got, st->ref, st->ref_crc, GB_UBER_THREADS);
   free(got);
}

static const char *const uber_params[] = {"kind", NULL};
static const struct gb_stream_test t_compute_uber = {
   .name = "compute.uber",
   .known_params = uber_params,
   .default_depths = {1, 4}, .ndefault = 2,
   .setup = uber_setup,
   .teardown = uber_teardown,
   .record = uber_record,
   .derive = uber_derive,
   .check = uber_check,
};

int gb_run_compute_uber(const struct gb_opts *o) { return gb_run_stream_test(o, &t_compute_uber); }

/* =========================================================================
 * 2. compute.conv2d (gaussian | bilateral, 720p | 1080p)
 * ========================================================================= */

struct conv_state {
   uint32_t w, h;
   bool bilateral;
   struct gb_pipe ph, pv;
   struct gb_buf in, mid, out;
   uint32_t *ref;
   uint32_t ref_crc;
};

static void *
conv_setup(struct gb_vk *vk, const struct gb_opts *o, char *err, size_t n)
{
   const char *kind = gb_param(o, "kind", "gaussian");
   bool bilateral = false;
   if (!strcmp(kind, "bilateral")) bilateral = true;
   else if (strcmp(kind, "gaussian")) {
      snprintf(err, n, "kind must be gaussian or bilateral");
      return NULL;
   }
   uint32_t w = 0, h = 0;
   if (!parse_res_w1(gb_param(o, "res", "1080p"), &w, &h)) {
      snprintf(err, n, "res must be 720p, 1080p, 1440p or 2160p");
      return NULL;
   }
   struct conv_state *st = calloc(1, sizeof(*st));
   st->w = w; st->h = h; st->bilateral = bilateral;
   size_t bytes = (size_t)w * h * sizeof(uint32_t);
   uint32_t *in_cpu = malloc(bytes);
   gb_conv2d_fill_input(in_cpu, w, h);
   st->ref = malloc(bytes);
   st->ref_crc = gb_conv2d_ref(bilateral, in_cpu, st->ref, w, h);

   gb_buf_create(vk, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, false, &st->in);
   gb_buf_create(vk, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, false, &st->mid);
   gb_buf_create(vk, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, false, &st->out);
   gb_buf_upload(vk, in_cpu, &st->in, bytes);
   free(in_cpu);

   uint32_t spec_val = bilateral ? 1u : 0u;
   VkSpecializationMapEntry me = {.constantID = 0, .offset = 0, .size = sizeof(uint32_t)};
   VkSpecializationInfo si = {.mapEntryCount = 1, .pMapEntries = &me, .dataSize = sizeof(spec_val), .pData = &spec_val};
   compute_pipe_spec(vk, GB_SPV(conv2d_h_comp), 2, 8, &si, VK_NULL_HANDLE, &st->ph);
   compute_pipe_spec(vk, GB_SPV(conv2d_v_comp), 2, 8, &si, VK_NULL_HANDLE, &st->pv);
   struct gb_buf *bh[2] = {&st->in, &st->mid};
   struct gb_buf *bv[2] = {&st->mid, &st->out};
   gb_pipe_set_buffers(vk, &st->ph, bh, 2);
   gb_pipe_set_buffers(vk, &st->pv, bv, 2);
   return st;
}

static void
conv_teardown(struct gb_vk *vk, void *p)
{
   struct conv_state *st = p;
   gb_pipe_destroy(vk, &st->ph);
   gb_pipe_destroy(vk, &st->pv);
   gb_buf_destroy(vk, &st->in);
   gb_buf_destroy(vk, &st->mid);
   gb_buf_destroy(vk, &st->out);
   free(st->ref);
   free(st);
}

static void
conv_record(void *p, VkCommandBuffer cb, uint64_t k)
{
   (void)k;
   struct conv_state *st = p;
   struct { int32_t w, h; } pc = {(int32_t)st->w, (int32_t)st->h};
   uint32_t gx = (st->w + GB_CONV_LOCAL - 1u) / GB_CONV_LOCAL;
   uint32_t gy = (st->h + GB_CONV_LOCAL - 1u) / GB_CONV_LOCAL;

   vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, st->ph.pipe);
   vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, st->ph.layout, 0, 1, &st->ph.set, 0, NULL);
   vkCmdPushConstants(cb, st->ph.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
   vkCmdDispatch(cb, gx, gy, 1);

   compute_barrier(cb);

   vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, st->pv.pipe);
   vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, st->pv.layout, 0, 1, &st->pv.set, 0, NULL);
   vkCmdPushConstants(cb, st->pv.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
   vkCmdDispatch(cb, gx, gy, 1);
}

static void
conv_derive(const struct gb_opts *o, struct gb_result *r, const struct gb_samples *us)
{
   uint32_t w = 1920, h = 1080;
   parse_res_w1(gb_param(o, "res", "1080p"), &w, &h);
   gb_derive_rate(r, "mpix_s", "Mpix/s", us, (double)w * h);
}

static void
conv_check(struct gb_vk *vk, void *p, struct gb_check *chk)
{
   struct conv_state *st = p;
   size_t count = (size_t)st->w * st->h;
   uint32_t *got = malloc(count * sizeof(uint32_t));
   gb_buf_readback(vk, &st->out, got, count * sizeof(uint32_t));
   gb_check_words(chk, got, st->ref, st->ref_crc, count);
   free(got);
}

static const char *const conv_params[] = {"kind", "res", NULL};
static const struct gb_stream_test t_compute_conv2d = {
   .name = "compute.conv2d",
   .known_params = conv_params,
   .default_depths = {1, 4}, .ndefault = 2,
   .setup = conv_setup,
   .teardown = conv_teardown,
   .record = conv_record,
   .derive = conv_derive,
   .check = conv_check,
};

int gb_run_compute_conv2d(const struct gb_opts *o) { return gb_run_stream_test(o, &t_compute_conv2d); }

/* =========================================================================
 * 3. compute.stereo (450x375 pitch 512, 11x11 dense SAD)
 * ========================================================================= */

struct stereo_state {
   uint32_t disp;
   struct gb_pipe p;
   struct gb_buf left, right, out;
   uint32_t *ref;
   uint32_t ref_crc;
};

static void *
stereo_setup(struct gb_vk *vk, const struct gb_opts *o, char *err, size_t n)
{
   long disp = gb_param_int(o, "disp", 64);
   if (disp < 8 || disp > 128) {
      snprintf(err, n, "disp must be 8..128");
      return NULL;
   }
   struct stereo_state *st = calloc(1, sizeof(*st));
   st->disp = (uint32_t)disp;
   size_t in_bytes = (size_t)GB_STEREO_PITCH * GB_STEREO_H;
   size_t out_words = (size_t)GB_STEREO_PITCH * GB_STEREO_H;
   size_t out_bytes = out_words * sizeof(uint32_t);
   uint32_t *l_cpu = calloc(1, in_bytes);
   uint32_t *r_cpu = calloc(1, in_bytes);
   gb_stereo_fill_inputs(l_cpu, r_cpu, st->disp);
   st->ref = calloc(out_words, sizeof(uint32_t));
   st->ref_crc = gb_stereo_ref(l_cpu, r_cpu, st->disp, st->ref);

   gb_buf_create(vk, in_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, false, &st->left);
   gb_buf_create(vk, in_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, false, &st->right);
   gb_buf_create(vk, out_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                 VK_BUFFER_USAGE_TRANSFER_DST_BIT, false, &st->out);
   gb_buf_upload(vk, l_cpu, &st->left, in_bytes);
   gb_buf_upload(vk, r_cpu, &st->right, in_bytes);
   free(l_cpu);
   free(r_cpu);

   /* Zero output padding columns (450..511) once so CRC matches CPU reference. */
   VkCommandBuffer cb = gb_oneshot_begin(vk);
   vkCmdFillBuffer(cb, st->out.buf, 0, out_bytes, 0);
   gb_oneshot_end(vk, cb);

   gb_compute_pipe(vk, GB_SPV(stereo_comp), 3, 16, &st->p);
   struct gb_buf *b[3] = {&st->left, &st->right, &st->out};
   gb_pipe_set_buffers(vk, &st->p, b, 3);
   return st;
}

static void
stereo_teardown(struct gb_vk *vk, void *p)
{
   struct stereo_state *st = p;
   gb_pipe_destroy(vk, &st->p);
   gb_buf_destroy(vk, &st->left);
   gb_buf_destroy(vk, &st->right);
   gb_buf_destroy(vk, &st->out);
   free(st->ref);
   free(st);
}

static void
stereo_record(void *p, VkCommandBuffer cb, uint64_t k)
{
   (void)k;
   struct stereo_state *st = p;
   struct { int32_t w, h, pitch, disp; } pc = {
      (int32_t)GB_STEREO_W, (int32_t)GB_STEREO_H, (int32_t)GB_STEREO_PITCH, (int32_t)st->disp
   };
   uint32_t gx = (GB_STEREO_W + GB_STEREO_LOCAL - 1u) / GB_STEREO_LOCAL;
   uint32_t gy = (GB_STEREO_H + GB_STEREO_LOCAL - 1u) / GB_STEREO_LOCAL;
   vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, st->p.pipe);
   vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, st->p.layout, 0, 1, &st->p.set, 0, NULL);
   vkCmdPushConstants(cb, st->p.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
   vkCmdDispatch(cb, gx, gy, 1);
}

static void
stereo_derive(const struct gb_opts *o, struct gb_result *r, const struct gb_samples *us)
{
   long disp = gb_param_int(o, "disp", 64);
   gb_derive_rate(r, "mde_s", "Mdisp/s", us, (double)GB_STEREO_W * GB_STEREO_H * (double)disp);
}

static void
stereo_check(struct gb_vk *vk, void *p, struct gb_check *chk)
{
   struct stereo_state *st = p;
   size_t count = (size_t)GB_STEREO_PITCH * GB_STEREO_H;
   uint32_t *got = malloc(count * sizeof(uint32_t));
   gb_buf_readback(vk, &st->out, got, count * sizeof(uint32_t));
   gb_check_words(chk, got, st->ref, st->ref_crc, count);
   free(got);
}

static const char *const stereo_params[] = {"disp", NULL};
static const struct gb_stream_test t_compute_stereo = {
   .name = "compute.stereo",
   .known_params = stereo_params,
   .default_depths = {1, 2}, .ndefault = 2,
   .setup = stereo_setup,
   .teardown = stereo_teardown,
   .record = stereo_record,
   .derive = stereo_derive,
   .check = stereo_check,
};

int gb_run_compute_stereo(const struct gb_opts *o) { return gb_run_stream_test(o, &t_compute_stereo); }

/* =========================================================================
 * 4. compute.edge (sobel | harris, 720p | 1080p)
 * ========================================================================= */

struct edge_state {
   uint32_t w, h;
   bool harris;
   struct gb_pipe p1, p2, p3;
   struct gb_buf in, grad, mag, resp, out, counter;
   uint32_t *ref;
   uint32_t ref_crc;
   uint32_t ref_corners;
};

static void *
edge_setup(struct gb_vk *vk, const struct gb_opts *o, char *err, size_t n)
{
   const char *kind = gb_param(o, "kind", "sobel");
   bool harris = false;
   if (!strcmp(kind, "harris")) harris = true;
   else if (strcmp(kind, "sobel")) {
      snprintf(err, n, "kind must be sobel or harris");
      return NULL;
   }
   uint32_t w = 0, h = 0;
   if (!parse_res_w1(gb_param(o, "res", "1080p"), &w, &h)) {
      snprintf(err, n, "res must be 720p, 1080p, 1440p or 2160p");
      return NULL;
   }
   struct edge_state *st = calloc(1, sizeof(*st));
   st->w = w; st->h = h; st->harris = harris;
   size_t bytes = (size_t)w * h * sizeof(uint32_t);
   uint32_t *in_cpu = malloc(bytes);
   gb_edge_fill_input(in_cpu, w, h);
   st->ref = malloc(bytes);
   st->ref_crc = gb_edge_ref(harris, in_cpu, st->ref, &st->ref_corners, w, h);

   gb_buf_create(vk, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, false, &st->in);
   gb_buf_create(vk, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, false, &st->grad);
   gb_buf_create(vk, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, false, &st->mag);
   gb_buf_create(vk, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, false, &st->out);
   gb_buf_upload(vk, in_cpu, &st->in, bytes);
   free(in_cpu);

   gb_compute_pipe(vk, GB_SPV(edge_sobel_comp), 3, 8, &st->p1);
   struct gb_buf *b1[3] = {&st->in, &st->grad, &st->mag};
   gb_pipe_set_buffers(vk, &st->p1, b1, 3);

   if (!harris) {
      gb_compute_pipe(vk, GB_SPV(edge_nms_comp), 3, 8, &st->p2);
      struct gb_buf *b2[3] = {&st->grad, &st->mag, &st->out};
      gb_pipe_set_buffers(vk, &st->p2, b2, 3);
   } else {
      gb_buf_create(vk, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, false, &st->resp);
      gb_buf_create(vk, 64, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT, false, &st->counter);
      gb_compute_pipe(vk, GB_SPV(edge_harris_resp_comp), 2, 8, &st->p2);
      struct gb_buf *b2[2] = {&st->grad, &st->resp};
      gb_pipe_set_buffers(vk, &st->p2, b2, 2);

      gb_compute_pipe(vk, GB_SPV(edge_harris_nms_comp), 3, 8, &st->p3);
      struct gb_buf *b3[3] = {&st->resp, &st->out, &st->counter};
      gb_pipe_set_buffers(vk, &st->p3, b3, 3);
   }
   return st;
}

static void
edge_teardown(struct gb_vk *vk, void *p)
{
   struct edge_state *st = p;
   gb_pipe_destroy(vk, &st->p1);
   gb_pipe_destroy(vk, &st->p2);
   if (st->harris) {
      gb_pipe_destroy(vk, &st->p3);
      gb_buf_destroy(vk, &st->resp);
      gb_buf_destroy(vk, &st->counter);
   }
   gb_buf_destroy(vk, &st->in);
   gb_buf_destroy(vk, &st->grad);
   gb_buf_destroy(vk, &st->mag);
   gb_buf_destroy(vk, &st->out);
   free(st->ref);
   free(st);
}

static void
edge_record(void *p, VkCommandBuffer cb, uint64_t k)
{
   (void)k;
   struct edge_state *st = p;
   struct { int32_t w, h; } pc = {(int32_t)st->w, (int32_t)st->h};
   uint32_t gx = (st->w + GB_EDGE_LOCAL - 1u) / GB_EDGE_LOCAL;
   uint32_t gy = (st->h + GB_EDGE_LOCAL - 1u) / GB_EDGE_LOCAL;

   if (st->harris) {
      vkCmdFillBuffer(cb, st->counter.buf, 0, 4, 0);
      VkMemoryBarrier2 mb = {
         .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
         .srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
         .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
         .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
         .dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
      };
      VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &mb};
      vkCmdPipelineBarrier2(cb, &di);
   }

   vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, st->p1.pipe);
   vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, st->p1.layout, 0, 1, &st->p1.set, 0, NULL);
   vkCmdPushConstants(cb, st->p1.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
   vkCmdDispatch(cb, gx, gy, 1);

   compute_barrier(cb);

   vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, st->p2.pipe);
   vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, st->p2.layout, 0, 1, &st->p2.set, 0, NULL);
   vkCmdPushConstants(cb, st->p2.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
   vkCmdDispatch(cb, gx, gy, 1);

   if (st->harris) {
      compute_barrier(cb);
      vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, st->p3.pipe);
      vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, st->p3.layout, 0, 1, &st->p3.set, 0, NULL);
      vkCmdPushConstants(cb, st->p3.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
      vkCmdDispatch(cb, gx, gy, 1);
   }
}

static void
edge_derive(const struct gb_opts *o, struct gb_result *r, const struct gb_samples *us)
{
   uint32_t w = 1920, h = 1080;
   parse_res_w1(gb_param(o, "res", "1080p"), &w, &h);
   gb_derive_rate(r, "mpix_s", "Mpix/s", us, (double)w * h);
}

static void
edge_check(struct gb_vk *vk, void *p, struct gb_check *chk)
{
   struct edge_state *st = p;
   size_t count = (size_t)st->w * st->h;
   uint32_t *got = malloc(count * sizeof(uint32_t));
   gb_buf_readback(vk, &st->out, got, count * sizeof(uint32_t));
   gb_check_words(chk, got, st->ref, st->ref_crc, count);
   free(got);
   if (st->harris && chk->valid) {
      uint32_t cnt = 0;
      gb_buf_readback(vk, &st->counter, &cnt, sizeof(cnt));
      if (cnt != st->ref_corners) {
         gb_log("harris corner count mismatch: got %u want %u", cnt, st->ref_corners);
         chk->valid = false;
      }
   }
}

static const char *const edge_params[] = {"kind", "res", NULL};
static const struct gb_stream_test t_compute_edge = {
   .name = "compute.edge",
   .known_params = edge_params,
   .default_depths = {1, 4}, .ndefault = 2,
   .setup = edge_setup,
   .teardown = edge_teardown,
   .record = edge_record,
   .derive = edge_derive,
   .check = edge_check,
};

int gb_run_compute_edge(const struct gb_opts *o) { return gb_run_stream_test(o, &t_compute_edge); }

/* =========================================================================
 * 5. render.deferred (mode=tile|dram, res=720p|1080p, msaa=1|4)
 * ========================================================================= */

struct deferred_state {
   uint32_t w, h, msaa;
   bool tile;
   struct gb_img g[4], depth, final_img, final_ms;
   VkSampler sampler;
   struct gb_pipe p_geom, p_light;
};

static const VkFormat DEF_GFMT[4] = {
   VK_FORMAT_R8G8B8A8_UNORM,
   VK_FORMAT_R16G16B16A16_SFLOAT,
   VK_FORMAT_R8G8B8A8_UNORM,
   VK_FORMAT_R32_SFLOAT,
};

static struct deferred_state *
deferred_setup_ex(struct gb_vk *vk, uint32_t w, uint32_t h, uint32_t msaa, bool tile, char *err, size_t errlen)
{
   if (tile && !vk->has_local_read) {
      snprintf(err, errlen, "device lacks VK_KHR_dynamic_rendering_local_read");
      return NULL;
   }
   VkSampleCountFlagBits samples = msaa == 4u ? VK_SAMPLE_COUNT_4_BIT : VK_SAMPLE_COUNT_1_BIT;
   struct deferred_state *st = calloc(1, sizeof(*st));
   st->w = w; st->h = h; st->msaa = msaa; st->tile = tile;

   VkImageUsageFlags gusage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                              (tile ? VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT : VK_IMAGE_USAGE_SAMPLED_BIT);
   for (int i = 0; i < 4; i++)
      gb_img_create_ex(vk, w, h, DEF_GFMT[i], samples, gusage, VK_IMAGE_ASPECT_COLOR_BIT, &st->g[i]);
   gb_img_create_ex(vk, w, h, VK_FORMAT_D32_SFLOAT, samples, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                    VK_IMAGE_ASPECT_DEPTH_BIT, &st->depth);
   gb_img_create_ex(vk, w, h, VK_FORMAT_R8G8B8A8_UNORM, VK_SAMPLE_COUNT_1_BIT,
                    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                    VK_IMAGE_ASPECT_COLOR_BIT, &st->final_img);
   if (tile && msaa == 4u)
      gb_img_create_ex(vk, w, h, VK_FORMAT_R8G8B8A8_UNORM, samples, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
                       VK_IMAGE_ASPECT_COLOR_BIT, &st->final_ms);

   /* Initial image layout transitions. */
   VkCommandBuffer cb = gb_oneshot_begin(vk);
   for (int i = 0; i < 4; i++)
      gb_img_barrier(cb, st->g[i].img, VK_IMAGE_LAYOUT_UNDEFINED,
                     tile ? VK_IMAGE_LAYOUT_RENDERING_LOCAL_READ_KHR : VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                     VK_PIPELINE_STAGE_2_NONE, 0,
                     VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
   {
      VkImageMemoryBarrier2 db = {
         .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
         .srcStageMask = VK_PIPELINE_STAGE_2_NONE, .dstStageMask = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT,
         .dstAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
         .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
         .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
         .image = st->depth.img,
         .subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1},
      };
      VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &db};
      vkCmdPipelineBarrier2(cb, &di);
   }
   gb_img_barrier(cb, st->final_img.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                  VK_PIPELINE_STAGE_2_NONE, 0,
                  VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
   if (st->final_ms.img)
      gb_img_barrier(cb, st->final_ms.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                     VK_PIPELINE_STAGE_2_NONE, 0,
                     VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
   gb_oneshot_end(vk, cb);

   /* ---- Geometry pipeline (4 MRTs + Depth) ---- */
   VkPipelineLayoutCreateInfo glci = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
   GB_VK_CHECK(vkCreatePipelineLayout(vk->dev, &glci, NULL, &st->p_geom.layout));
   VkShaderModule mv = gb_shader(vk, GB_SPV(deferred_geom_vert));
   VkShaderModule mf = gb_shader(vk, GB_SPV(deferred_geom_frag));
   VkPipelineShaderStageCreateInfo gst[2] = {
      {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = mv, .pName = "main"},
      {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = mf, .pName = "main"},
   };
   VkPipelineVertexInputStateCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
   VkPipelineInputAssemblyStateCreateInfo ia = {.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO, .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST};
   VkPipelineViewportStateCreateInfo vp = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, .viewportCount = 1, .scissorCount = 1};
   VkPipelineRasterizationStateCreateInfo rs = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO, .polygonMode = VK_POLYGON_MODE_FILL, .cullMode = VK_CULL_MODE_NONE, .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE, .lineWidth = 1.0f};
   VkPipelineMultisampleStateCreateInfo ms_g = {.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO, .rasterizationSamples = samples};
   VkPipelineDepthStencilStateCreateInfo ds_g = {.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO, .depthTestEnable = VK_TRUE, .depthWriteEnable = VK_TRUE, .depthCompareOp = VK_COMPARE_OP_LESS};
   VkPipelineColorBlendAttachmentState ba5[5];
   for (int i = 0; i < 5; i++)
      ba5[i] = (VkPipelineColorBlendAttachmentState){.colorWriteMask = 0xf};
   VkPipelineColorBlendStateCreateInfo cbs_g = {.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO, .attachmentCount = tile ? 5u : 4u, .pAttachments = ba5};
   VkDynamicState dyn[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
   VkPipelineDynamicStateCreateInfo dyns = {.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO, .dynamicStateCount = 2, .pDynamicStates = dyn};
   VkFormat fmts5[5] = {DEF_GFMT[0], DEF_GFMT[1], DEF_GFMT[2], DEF_GFMT[3], VK_FORMAT_R8G8B8A8_UNORM};
   uint32_t geom_locs[5] = {0, 1, 2, 3, VK_ATTACHMENT_UNUSED};
   VkRenderingAttachmentLocationInfoKHR geom_loc_info = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_LOCATION_INFO_KHR,
      .colorAttachmentCount = 5,
      .pColorAttachmentLocations = geom_locs,
   };
   VkPipelineRenderingCreateInfo ri_g = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
      .pNext = tile ? &geom_loc_info : NULL,
      .colorAttachmentCount = tile ? 5u : 4u,
      .pColorAttachmentFormats = fmts5,
      .depthAttachmentFormat = VK_FORMAT_D32_SFLOAT,
   };
   VkGraphicsPipelineCreateInfo gci = {
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
      .pNext = &ri_g,
      .stageCount = 2, .pStages = gst,
      .pVertexInputState = &vi, .pInputAssemblyState = &ia,
      .pViewportState = &vp, .pRasterizationState = &rs,
      .pMultisampleState = &ms_g, .pDepthStencilState = &ds_g,
      .pColorBlendState = &cbs_g, .pDynamicState = &dyns,
      .layout = st->p_geom.layout,
   };
   GB_VK_CHECK(vkCreateGraphicsPipelines(vk->dev, VK_NULL_HANDLE, 1, &gci, NULL, &st->p_geom.pipe));
   vkDestroyShaderModule(vk->dev, mv, NULL);
   vkDestroyShaderModule(vk->dev, mf, NULL);

   /* ---- Lighting pipeline ---- */
   VkDescriptorType dtype = tile ? VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
   VkDescriptorSetLayoutBinding lbs[4];
   for (int i = 0; i < 4; i++)
      lbs[i] = (VkDescriptorSetLayoutBinding){
         .binding = (uint32_t)i, .descriptorType = dtype, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT
      };
   VkDescriptorSetLayoutCreateInfo ldci = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .bindingCount = 4, .pBindings = lbs};
   GB_VK_CHECK(vkCreateDescriptorSetLayout(vk->dev, &ldci, NULL, &st->p_light.dsl));
   VkDescriptorPoolSize lps = {dtype, 4};
   VkDescriptorPoolCreateInfo lpci = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &lps};
   GB_VK_CHECK(vkCreateDescriptorPool(vk->dev, &lpci, NULL, &st->p_light.dpool));
   VkDescriptorSetAllocateInfo lsai = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, .descriptorPool = st->p_light.dpool, .descriptorSetCount = 1, .pSetLayouts = &st->p_light.dsl};
   GB_VK_CHECK(vkAllocateDescriptorSets(vk->dev, &lsai, &st->p_light.set));
   VkPipelineLayoutCreateInfo llci = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1, .pSetLayouts = &st->p_light.dsl};
   GB_VK_CHECK(vkCreatePipelineLayout(vk->dev, &llci, NULL, &st->p_light.layout));

   if (!tile) {
      VkSamplerCreateInfo sci = {
         .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
         .magFilter = VK_FILTER_NEAREST, .minFilter = VK_FILTER_NEAREST,
         .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
         .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
         .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
         .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      };
      GB_VK_CHECK(vkCreateSampler(vk->dev, &sci, NULL, &st->sampler));
   }
   VkDescriptorImageInfo ii[4];
   VkWriteDescriptorSet wr[4];
   for (int i = 0; i < 4; i++) {
      ii[i] = (VkDescriptorImageInfo){
         .sampler = tile ? VK_NULL_HANDLE : st->sampler,
         .imageView = st->g[i].view,
         .imageLayout = tile ? VK_IMAGE_LAYOUT_RENDERING_LOCAL_READ_KHR : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
      };
      wr[i] = (VkWriteDescriptorSet){
         .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         .dstSet = st->p_light.set, .dstBinding = (uint32_t)i, .descriptorCount = 1,
         .descriptorType = dtype, .pImageInfo = &ii[i],
      };
   }
   vkUpdateDescriptorSets(vk->dev, 4, wr, 0, NULL);

   VkShaderModule lmv = gb_shader(vk, GB_SPV(fs_vert));
   const uint32_t *lf_code = tile ? (msaa == 4u ? deferred_tile_msaa_frag : deferred_tile_frag)
                                  : (msaa == 4u ? deferred_dram_msaa_frag : deferred_dram_frag);
   size_t lf_size = tile ? (msaa == 4u ? deferred_tile_msaa_frag_size : deferred_tile_frag_size)
                         : (msaa == 4u ? deferred_dram_msaa_frag_size : deferred_dram_frag_size);
   VkShaderModule lmf = gb_shader(vk, lf_code, lf_size);
   VkPipelineShaderStageCreateInfo lst[2] = {
      {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = lmv, .pName = "main"},
      {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = lmf, .pName = "main"},
   };
   VkPipelineMultisampleStateCreateInfo ms_l = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
      .rasterizationSamples = tile ? samples : VK_SAMPLE_COUNT_1_BIT,
      .sampleShadingEnable = (tile && msaa == 4u) ? VK_TRUE : VK_FALSE,
      .minSampleShading = (tile && msaa == 4u) ? 1.0f : 0.0f,
   };
   VkPipelineDepthStencilStateCreateInfo ds_l = {.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
   VkPipelineColorBlendStateCreateInfo cbs_l = {.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO, .attachmentCount = tile ? 5u : 1u, .pAttachments = ba5};
   uint32_t light_locs[5] = {VK_ATTACHMENT_UNUSED, VK_ATTACHMENT_UNUSED, VK_ATTACHMENT_UNUSED, VK_ATTACHMENT_UNUSED, 0};
   uint32_t light_in_idx[5] = {0, 1, 2, 3, VK_ATTACHMENT_UNUSED};
   VkRenderingInputAttachmentIndexInfoKHR light_in_info = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_INPUT_ATTACHMENT_INDEX_INFO_KHR,
      .colorAttachmentCount = 5,
      .pColorAttachmentInputIndices = light_in_idx,
   };
   VkRenderingAttachmentLocationInfoKHR light_loc_info = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_LOCATION_INFO_KHR,
      .pNext = &light_in_info,
      .colorAttachmentCount = 5,
      .pColorAttachmentLocations = light_locs,
   };
   VkFormat final_fmt = VK_FORMAT_R8G8B8A8_UNORM;
   VkPipelineRenderingCreateInfo ri_l = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
      .pNext = tile ? &light_loc_info : NULL,
      .colorAttachmentCount = tile ? 5u : 1u,
      .pColorAttachmentFormats = tile ? fmts5 : &final_fmt,
      .depthAttachmentFormat = tile ? VK_FORMAT_D32_SFLOAT : VK_FORMAT_UNDEFINED,
   };
   VkGraphicsPipelineCreateInfo lci = {
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
      .pNext = &ri_l,
      .stageCount = 2, .pStages = lst,
      .pVertexInputState = &vi, .pInputAssemblyState = &ia,
      .pViewportState = &vp, .pRasterizationState = &rs,
      .pMultisampleState = &ms_l, .pDepthStencilState = &ds_l,
      .pColorBlendState = &cbs_l, .pDynamicState = &dyns,
      .layout = st->p_light.layout,
   };
   GB_VK_CHECK(vkCreateGraphicsPipelines(vk->dev, VK_NULL_HANDLE, 1, &lci, NULL, &st->p_light.pipe));
   vkDestroyShaderModule(vk->dev, lmv, NULL);
   vkDestroyShaderModule(vk->dev, lmf, NULL);
   return st;
}

static void *
deferred_setup(struct gb_vk *vk, const struct gb_opts *o, char *err, size_t n)
{
   const char *mode = gb_param(o, "mode", "tile");
   bool tile = !strcmp(mode, "tile");
   if (!tile && strcmp(mode, "dram")) {
      snprintf(err, n, "mode must be tile or dram");
      return NULL;
   }
   uint32_t w = 0, h = 0;
   if (!parse_res_w1(gb_param(o, "res", "1080p"), &w, &h)) {
      snprintf(err, n, "res must be 720p, 1080p, 1440p or 2160p");
      return NULL;
   }
   long msaa = gb_param_int(o, "msaa", 1);
   if (msaa != 1 && msaa != 4) {
      snprintf(err, n, "msaa must be 1 or 4");
      return NULL;
   }
   return deferred_setup_ex(vk, w, h, (uint32_t)msaa, tile, err, n);
}

static void
deferred_teardown(struct gb_vk *vk, void *p)
{
   struct deferred_state *st = p;
   gb_pipe_destroy(vk, &st->p_geom);
   gb_pipe_destroy(vk, &st->p_light);
   if (st->sampler) vkDestroySampler(vk->dev, st->sampler, NULL);
   for (int i = 0; i < 4; i++) gb_img_destroy(vk, &st->g[i]);
   gb_img_destroy(vk, &st->depth);
   gb_img_destroy(vk, &st->final_img);
   if (st->final_ms.img) gb_img_destroy(vk, &st->final_ms);
   free(st);
}

struct deferred_rec_ctx {
   struct gb_vk *vk;
   struct deferred_state *st;
};

static void
deferred_record_impl(struct gb_vk *vk, struct deferred_state *st, VkCommandBuffer cb)
{
   VkViewport vp = {0, 0, (float)st->w, (float)st->h, 0.0f, 1.0f};
   VkRect2D sc = {{0, 0}, {st->w, st->h}};

   if (st->tile) {
      VkRenderingAttachmentInfo ca[5];
      for (int i = 0; i < 4; i++) {
         ca[i] = (VkRenderingAttachmentInfo){
            .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
            .imageView = st->g[i].view,
            .imageLayout = VK_IMAGE_LAYOUT_RENDERING_LOCAL_READ_KHR,
            .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
            .storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
         };
      }
      ca[4] = (VkRenderingAttachmentInfo){
         .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
         .imageView = st->msaa == 4u ? st->final_ms.view : st->final_img.view,
         .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
         .resolveMode = st->msaa == 4u ? VK_RESOLVE_MODE_AVERAGE_BIT : VK_RESOLVE_MODE_NONE,
         .resolveImageView = st->msaa == 4u ? st->final_img.view : VK_NULL_HANDLE,
         .resolveImageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
         .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
         .storeOp = st->msaa == 4u ? VK_ATTACHMENT_STORE_OP_DONT_CARE : VK_ATTACHMENT_STORE_OP_STORE,
      };
      VkRenderingAttachmentInfo da = {
         .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
         .imageView = st->depth.view,
         .imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
         .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
         .storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
         .clearValue.depthStencil = {1.0f, 0},
      };
      VkRenderingInfo ri = {
         .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
         .renderArea = {{0, 0}, {st->w, st->h}},
         .layerCount = 1,
         .colorAttachmentCount = 5,
         .pColorAttachments = ca,
         .pDepthAttachment = &da,
      };
      vkCmdBeginRendering(cb, &ri);
      vkCmdSetViewport(cb, 0, 1, &vp);
      vkCmdSetScissor(cb, 0, 1, &sc);

      /* Sub-step A: Geometry pass writes G0..G3 */
      uint32_t geom_locs[5] = {0, 1, 2, 3, VK_ATTACHMENT_UNUSED};
      VkRenderingAttachmentLocationInfoKHR gloc = {
         .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_LOCATION_INFO_KHR,
         .colorAttachmentCount = 5, .pColorAttachmentLocations = geom_locs,
      };
      vk->cmd_set_attachment_locations(cb, &gloc);
      vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, st->p_geom.pipe);
      vkCmdDraw(cb, GB_DEFERRED_QUADS * 6u, 1, 0, 0);

      /* Tile-local BY_REGION barrier (stays in TBDR SRAM, no DRAM flush) */
      VkMemoryBarrier2 mb = {
         .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
         .srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
         .srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
         .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
         .dstAccessMask = VK_ACCESS_2_INPUT_ATTACHMENT_READ_BIT,
      };
      VkDependencyInfo di = {
         .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
         .dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT,
         .memoryBarrierCount = 1, .pMemoryBarriers = &mb,
      };
      vkCmdPipelineBarrier2(cb, &di);

      /* Sub-step B: Lighting pass reads G0..G3 via local_read, writes Final */
      uint32_t light_locs[5] = {VK_ATTACHMENT_UNUSED, VK_ATTACHMENT_UNUSED, VK_ATTACHMENT_UNUSED, VK_ATTACHMENT_UNUSED, 0};
      uint32_t light_in[5] = {0, 1, 2, 3, VK_ATTACHMENT_UNUSED};
      VkRenderingAttachmentLocationInfoKHR lloc = {
         .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_LOCATION_INFO_KHR,
         .colorAttachmentCount = 5, .pColorAttachmentLocations = light_locs,
      };
      VkRenderingInputAttachmentIndexInfoKHR lin = {
         .sType = VK_STRUCTURE_TYPE_RENDERING_INPUT_ATTACHMENT_INDEX_INFO_KHR,
         .colorAttachmentCount = 5, .pColorAttachmentInputIndices = light_in,
      };
      vk->cmd_set_attachment_locations(cb, &lloc);
      vk->cmd_set_input_attachment_indices(cb, &lin);
      vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, st->p_light.pipe);
      vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, st->p_light.layout, 0, 1, &st->p_light.set, 0, NULL);
      vkCmdDraw(cb, 3, 1, 0, 0);
      vkCmdEndRendering(cb);
   } else {
      /* Pass 1: Geometry -> G0..G3 stored to DRAM */
      VkRenderingAttachmentInfo ca4[4];
      for (int i = 0; i < 4; i++) {
         ca4[i] = (VkRenderingAttachmentInfo){
            .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
            .imageView = st->g[i].view,
            .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
            .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
         };
      }
      VkRenderingAttachmentInfo da = {
         .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
         .imageView = st->depth.view,
         .imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
         .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
         .storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
         .clearValue.depthStencil = {1.0f, 0},
      };
      VkRenderingInfo ri1 = {
         .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
         .renderArea = {{0, 0}, {st->w, st->h}},
         .layerCount = 1, .colorAttachmentCount = 4, .pColorAttachments = ca4, .pDepthAttachment = &da,
      };
      vkCmdBeginRendering(cb, &ri1);
      vkCmdSetViewport(cb, 0, 1, &vp);
      vkCmdSetScissor(cb, 0, 1, &sc);
      vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, st->p_geom.pipe);
      vkCmdDraw(cb, GB_DEFERRED_QUADS * 6u, 1, 0, 0);
      vkCmdEndRendering(cb);

      for (int i = 0; i < 4; i++)
         gb_img_barrier(cb, st->g[i].img, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                        VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);

      /* Pass 2: Sample G0..G3 from DRAM -> Final */
      VkRenderingAttachmentInfo cf = {
         .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
         .imageView = st->final_img.view,
         .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
         .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
         .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
      };
      VkRenderingInfo ri2 = {
         .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
         .renderArea = {{0, 0}, {st->w, st->h}},
         .layerCount = 1, .colorAttachmentCount = 1, .pColorAttachments = &cf,
      };
      vkCmdBeginRendering(cb, &ri2);
      vkCmdSetViewport(cb, 0, 1, &vp);
      vkCmdSetScissor(cb, 0, 1, &sc);
      vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, st->p_light.pipe);
      vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, st->p_light.layout, 0, 1, &st->p_light.set, 0, NULL);
      vkCmdDraw(cb, 3, 1, 0, 0);
      vkCmdEndRendering(cb);

      for (int i = 0; i < 4; i++)
         gb_img_barrier(cb, st->g[i].img, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                        VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
   }
}

struct deferred_wrap {
   struct gb_vk *vk;
   struct deferred_state *st;
};

static void *
deferred_wrap_setup(struct gb_vk *vk, const struct gb_opts *o, char *err, size_t n)
{
   struct deferred_state *st = deferred_setup(vk, o, err, n);
   if (!st) return NULL;
   struct deferred_wrap *w = calloc(1, sizeof(*w));
   w->vk = vk; w->st = st;
   return w;
}

static void
deferred_wrap_teardown(struct gb_vk *vk, void *p)
{
   struct deferred_wrap *w = p;
   deferred_teardown(vk, w->st);
   free(w);
}

static void
deferred_wrap_record(void *p, VkCommandBuffer cb, uint64_t k)
{
   (void)k;
   struct deferred_wrap *w = p;
   deferred_record_impl(w->vk, w->st, cb);
}

static void
deferred_derive(const struct gb_opts *o, struct gb_result *r, const struct gb_samples *us)
{
   uint32_t w = 1920, h = 1080;
   parse_res_w1(gb_param(o, "res", "1080p"), &w, &h);
   gb_derive_rate(r, "gpix_s", "Gpix/s", us, (double)w * h / 1e9 * 1e6);
   gb_derive_rate(r, "fps", "fps", us, 1e6);
}

static void
deferred_check(struct gb_vk *vk, void *p, struct gb_check *chk)
{
   struct deferred_wrap *w = p;
   size_t bytes = (size_t)w->st->w * w->st->h * 4u;
   uint8_t *got = malloc(bytes);
   gb_img_readback(vk, &w->st->final_img, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, got, 4u);
   gb_deferred_check_fb(got, w->st->w, w->st->h, chk);
   free(got);
}

static const char *const deferred_params[] = {"mode", "res", "msaa", NULL};
static const struct gb_stream_test t_render_deferred = {
   .name = "render.deferred",
   .known_params = deferred_params,
   .default_depths = {1, 2}, .ndefault = 2,
   .setup = deferred_wrap_setup,
   .teardown = deferred_wrap_teardown,
   .record = deferred_wrap_record,
   .derive = deferred_derive,
   .check = deferred_check,
};

int gb_run_render_deferred(const struct gb_opts *o) { return gb_run_stream_test(o, &t_render_deferred); }

/* =========================================================================
 * 6. concurrent.sram (heavy 4-MRT + MSAA fragment overlapping high-reg+16KB
 *    shared-mem async compute, competing for AGX3 unified on-chip SRAM)
 * ========================================================================= */

struct sram_thr {
   struct gb_stream *s;
   uint64_t t0, warm, win;
   int nwin;
   struct gb_pass pass;
   pthread_t th;
};

static void *
sram_thr_main(void *p)
{
   struct sram_thr *t = p;
   gb_stream_run(t->s, false, t->t0, t->warm, t->nwin, t->win, &t->pass);
   return NULL;
}

int
gb_run_concurrent_sram(const struct gb_opts *o)
{
   static const char *const known[] = {"res", "msaa", NULL};
   if (!gb_check_params(o, known)) {
      gb_result_simple(o, GB_MODE_QUEUES, 1, NULL, "fail", "unknown parameter");
      return 1;
   }
   uint32_t w = 0, h = 0;
   if (!parse_res_w1(gb_param(o, "res", "1080p"), &w, &h)) {
      gb_result_simple(o, GB_MODE_QUEUES, 1, NULL, "fail", "res must be 720p, 1080p, 1440p or 2160p");
      return 1;
   }
   long msaa = gb_param_int(o, "msaa", 4);
   if (msaa != 1 && msaa != 4) {
      gb_result_simple(o, GB_MODE_QUEUES, 1, NULL, "fail", "msaa must be 1 or 4");
      return 1;
   }
   const int depth = GB_CR_DEPTH;
   if (gb_vk_probe_queue_count(o->device) < 2) {
      gb_result_simple(o, GB_MODE_QUEUES, 2 * depth, NULL, "skip", "device has 1 queue");
      return 0;
   }
   struct gb_vk vk;
   gb_vk_create(&vk, o->device, 2);
   char err[128] = "";
   bool use_tile = vk.has_local_read;
   struct deferred_state *def = deferred_setup_ex(&vk, w, h, (uint32_t)msaa, use_tile, err, sizeof(err));
   if (!def) {
      gb_result_simple(o, GB_MODE_QUEUES, 2 * depth, NULL, "skip", err);
      gb_vk_destroy(&vk);
      return 0;
   }
   struct deferred_wrap dw = {.vk = &vk, .st = def};
   struct uber_state *ub = uber_setup_kind(&vk, GB_UBER_SRAM);

   const uint64_t W = (uint64_t)o->warmup_ms * 1000000ull, R = (uint64_t)o->run_ms * 1000000ull;
   struct gb_stream sc, sr;
   gb_stream_init(&sc, &vk, 0, depth, uber_record, NULL, ub);
   gb_stream_init(&sr, &vk, 1, depth, deferred_wrap_record, NULL, &dw);
   struct gb_pass ca = {0}, ra = {0};
   gb_stream_run(&sc, false, gb_now_ns(), W, o->runs, R, &ca);
   gb_stream_run(&sr, false, gb_now_ns(), W, o->runs, R, &ra);

   uint64_t t0 = gb_now_ns() + 20000000ull;
   struct sram_thr th[2] = {
      {.s = &sc, .t0 = t0, .warm = W, .win = R, .nwin = o->runs},
      {.s = &sr, .t0 = t0, .warm = W, .win = R, .nwin = o->runs},
   };
   for (int i = 0; i < 2; i++) pthread_create(&th[i].th, NULL, sram_thr_main, &th[i]);
   for (int i = 0; i < 2; i++) pthread_join(th[i].th, NULL);
   gb_stream_finish(&sc);
   gb_stream_finish(&sr);

   struct gb_check chk = {0};
   bool bad = ca.failed || ra.failed || th[0].pass.failed || th[1].pass.failed;
   if (!bad)
      uber_check(&vk, ub, &chk);

   struct gb_result r;
   const char *extra[3] = {"compute=uber.sram(16KB)", use_tile ? "render=deferred:tile" : "render=deferred:dram", NULL};
   gb_result_begin(&r, o, GB_MODE_QUEUES, 2 * depth, extra);
   gb_result_run_info(&r, o->warmup_ms, o->runs, o->run_ms);
   struct gb_samples s[4] = {0};
   gb_pass_rate(&ca, &s[0]);
   gb_pass_rate(&ra, &s[1]);
   gb_pass_rate(&th[0].pass, &s[2]);
   gb_pass_rate(&th[1].pass, &s[3]);
   for (int i = 0; i < 4; i++) bad |= !s[i].n;
   if (bad) {
      gb_result_finish(&r, "fail", NULL, "GPU wait timed out or no completions");
   } else {
      gb_result_metric(&r, "compute_alone_us", "us", true, &s[0], true);
      gb_result_metric(&r, "render_alone_us", "us", true, &s[1], true);
      gb_result_metric(&r, "compute_conc_us", "us", true, &s[2], true);
      gb_result_metric(&r, "render_conc_us", "us", true, &s[3], true);
      double ov = gb_median(&s[0]) / gb_median(&s[2]) + gb_median(&s[1]) / gb_median(&s[3]);
      gb_result_metric1(&r, "overlap", "x", false, ov);
      gb_result_end_metrics(&r);
      gb_result_check(&r, &chk);
      gb_result_finish(&r, chk.valid ? "ok" : "fail", "overlap",
                       chk.valid ? NULL : "compute output corrupted under SRAM contention");
   }
   for (int i = 0; i < 4; i++) gb_samples_free(&s[i]);
   gb_pass_free(&ca); gb_pass_free(&ra);
   gb_pass_free(&th[0].pass); gb_pass_free(&th[1].pass);
   uber_teardown(&vk, ub);
   deferred_teardown(&vk, def);
   gb_vk_destroy(&vk);
   return 0;
}

/* =========================================================================
 * 7. render.mdi (bindless texture array + vkCmdDrawIndexedIndirectCount)
 * ========================================================================= */

struct mdi_state {
   uint32_t draws, bindless;
   struct gb_img target;
   struct gb_img *texs;
   VkSampler sampler;
   struct gb_buf vb, ib, indirect, count_buf;
   struct gb_pipe p;
   uint32_t *ref;
   uint32_t ref_crc;
};

static void *
mdi_setup(struct gb_vk *vk, const struct gb_opts *o, char *err, size_t n)
{
   long draws = gb_param_int(o, "draws", 4096);
   long bindless = gb_param_int(o, "bindless", 1024);
   if (draws < 1 || draws > (long)GB_MDI_MAX_DRAWS) {
      snprintf(err, n, "draws must be 1..%u", GB_MDI_MAX_DRAWS);
      return NULL;
   }
   if (bindless < 1 || bindless > (long)GB_MDI_MAX_BINDLESS) {
      snprintf(err, n, "bindless must be 1..%u", GB_MDI_MAX_BINDLESS);
      return NULL;
   }
   if (!vk->has_draw_indirect_count) {
      snprintf(err, n, "device lacks drawIndirectCount");
      return NULL;
   }
   if (!vk->has_descriptor_indexing || vk->max_sampled_images < (uint32_t)bindless) {
      snprintf(err, n, "device lacks descriptorIndexing or enough sampled images (%u < %ld)",
               vk->max_sampled_images, bindless);
      return NULL;
   }
   struct mdi_state *st = calloc(1, sizeof(*st));
   st->draws = (uint32_t)draws;
   st->bindless = (uint32_t)bindless;

   gb_img_create(vk, GB_MDI_DIM, GB_MDI_DIM, VK_FORMAT_R8G8B8A8_UNORM,
                 VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, &st->target);

   /* Quad vertex + index buffers (4 vertices, 6 indices). */
   const float verts[8] = {0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f};
   const uint16_t idxs[6] = {0, 1, 2, 0, 2, 3};
   gb_buf_create(vk, sizeof(verts), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, false, &st->vb);
   gb_buf_create(vk, sizeof(idxs), VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, false, &st->ib);
   gb_buf_upload(vk, verts, &st->vb, sizeof(verts));
   gb_buf_upload(vk, idxs, &st->ib, sizeof(idxs));

   /* Indirect draw commands + count buffer. */
   VkDrawIndexedIndirectCommand *cmds = calloc(st->draws, sizeof(*cmds));
   for (uint32_t d = 0; d < st->draws; d++) {
      cmds[d] = (VkDrawIndexedIndirectCommand){
         .indexCount = 6,
         .instanceCount = 1,
         .firstIndex = 0,
         .vertexOffset = 0,
         .firstInstance = d,
      };
   }
   gb_buf_create(vk, st->draws * sizeof(*cmds), VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                 false, &st->indirect);
   gb_buf_upload(vk, cmds, &st->indirect, st->draws * sizeof(*cmds));
   free(cmds);

   gb_buf_create(vk, 16, VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, false, &st->count_buf);
   gb_buf_upload(vk, &st->draws, &st->count_buf, sizeof(uint32_t));

   /* Bindless 16x16 RGBA8 texture array (uploaded in one batch staging buffer). */
   st->texs = calloc(st->bindless, sizeof(struct gb_img));
   size_t one_tex = GB_MDI_TEX_DIM * GB_MDI_TEX_DIM * sizeof(uint32_t);
   struct gb_buf stg;
   gb_buf_create(vk, st->bindless * one_tex, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, true, &stg);
   for (uint32_t t = 0; t < st->bindless; t++) {
      gb_img_create(vk, GB_MDI_TEX_DIM, GB_MDI_TEX_DIM, VK_FORMAT_R8G8B8A8_UNORM,
                    VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, &st->texs[t]);
      gb_mdi_fill_tex(t, (uint32_t *)((uint8_t *)stg.map + t * one_tex));
   }
   VkCommandBuffer cb = gb_oneshot_begin(vk);
   gb_img_barrier(cb, st->target.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                  VK_PIPELINE_STAGE_2_NONE, 0,
                  VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
   for (uint32_t t = 0; t < st->bindless; t++) {
      gb_img_barrier(cb, st->texs[t].img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                     VK_PIPELINE_STAGE_2_NONE, 0, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
      VkBufferImageCopy r = {
         .bufferOffset = t * one_tex,
         .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
         .imageExtent = {GB_MDI_TEX_DIM, GB_MDI_TEX_DIM, 1},
      };
      vkCmdCopyBufferToImage(cb, stg.buf, st->texs[t].img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &r);
      gb_img_barrier(cb, st->texs[t].img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                     VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                     VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
   }
   gb_oneshot_end(vk, cb);
   gb_buf_destroy(vk, &stg);

   VkSamplerCreateInfo sci = {
      .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
      .magFilter = VK_FILTER_NEAREST, .minFilter = VK_FILTER_NEAREST,
      .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
      .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
   };
   GB_VK_CHECK(vkCreateSampler(vk->dev, &sci, NULL, &st->sampler));

   /* Bindless descriptor set + pipeline. */
   VkDescriptorSetLayoutBinding lb = {
      .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
      .descriptorCount = st->bindless, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
   };
   VkDescriptorSetLayoutCreateInfo dci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 1, .pBindings = &lb,
   };
   GB_VK_CHECK(vkCreateDescriptorSetLayout(vk->dev, &dci, NULL, &st->p.dsl));
   VkDescriptorPoolSize ps = {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, st->bindless};
   VkDescriptorPoolCreateInfo pci = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &ps};
   GB_VK_CHECK(vkCreateDescriptorPool(vk->dev, &pci, NULL, &st->p.dpool));
   VkDescriptorSetAllocateInfo sai = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, .descriptorPool = st->p.dpool, .descriptorSetCount = 1, .pSetLayouts = &st->p.dsl};
   GB_VK_CHECK(vkAllocateDescriptorSets(vk->dev, &sai, &st->p.set));

   VkDescriptorImageInfo *iis = calloc(st->bindless, sizeof(*iis));
   for (uint32_t t = 0; t < st->bindless; t++)
      iis[t] = (VkDescriptorImageInfo){st->sampler, st->texs[t].view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
   VkWriteDescriptorSet wr = {
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = st->p.set, .dstBinding = 0, .descriptorCount = st->bindless,
      .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .pImageInfo = iis,
   };
   vkUpdateDescriptorSets(vk->dev, 1, &wr, 0, NULL);
   free(iis);

   VkPushConstantRange pcr = {VK_SHADER_STAGE_VERTEX_BIT, 0, 4};
   VkPipelineLayoutCreateInfo lci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1, .pSetLayouts = &st->p.dsl,
      .pushConstantRangeCount = 1, .pPushConstantRanges = &pcr,
   };
   GB_VK_CHECK(vkCreatePipelineLayout(vk->dev, &lci, NULL, &st->p.layout));

   VkShaderModule mv = gb_shader(vk, GB_SPV(mdi_vert));
   VkShaderModule mf = gb_shader(vk, GB_SPV(mdi_frag));
   VkPipelineShaderStageCreateInfo stages[2] = {
      {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = mv, .pName = "main"},
      {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = mf, .pName = "main"},
   };
   VkVertexInputBindingDescription vbd = {.binding = 0, .stride = 8, .inputRate = VK_VERTEX_INPUT_RATE_VERTEX};
   VkVertexInputAttributeDescription vad = {.location = 0, .binding = 0, .format = VK_FORMAT_R32G32_SFLOAT, .offset = 0};
   VkPipelineVertexInputStateCreateInfo vi = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
      .vertexBindingDescriptionCount = 1, .pVertexBindingDescriptions = &vbd,
      .vertexAttributeDescriptionCount = 1, .pVertexAttributeDescriptions = &vad,
   };
   VkPipelineInputAssemblyStateCreateInfo ia = {.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO, .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST};
   VkPipelineViewportStateCreateInfo vp = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, .viewportCount = 1, .scissorCount = 1};
   VkPipelineRasterizationStateCreateInfo rs = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO, .polygonMode = VK_POLYGON_MODE_FILL, .cullMode = VK_CULL_MODE_NONE, .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE, .lineWidth = 1.0f};
   VkPipelineMultisampleStateCreateInfo ms = {.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO, .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT};
   VkPipelineColorBlendAttachmentState ba = {.colorWriteMask = 0xf};
   VkPipelineColorBlendStateCreateInfo cbs = {.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO, .attachmentCount = 1, .pAttachments = &ba};
   VkDynamicState dyn[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
   VkPipelineDynamicStateCreateInfo dyns = {.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO, .dynamicStateCount = 2, .pDynamicStates = dyn};
   VkFormat fmt = VK_FORMAT_R8G8B8A8_UNORM;
   VkPipelineRenderingCreateInfo ri = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO, .colorAttachmentCount = 1, .pColorAttachmentFormats = &fmt};
   VkGraphicsPipelineCreateInfo gci = {
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, .pNext = &ri,
      .stageCount = 2, .pStages = stages,
      .pVertexInputState = &vi, .pInputAssemblyState = &ia,
      .pViewportState = &vp, .pRasterizationState = &rs,
      .pMultisampleState = &ms, .pColorBlendState = &cbs, .pDynamicState = &dyns,
      .layout = st->p.layout,
   };
   GB_VK_CHECK(vkCreateGraphicsPipelines(vk->dev, VK_NULL_HANDLE, 1, &gci, NULL, &st->p.pipe));
   vkDestroyShaderModule(vk->dev, mv, NULL);
   vkDestroyShaderModule(vk->dev, mf, NULL);

   st->ref = malloc((size_t)GB_MDI_DIM * GB_MDI_DIM * sizeof(uint32_t));
   st->ref_crc = gb_mdi_ref_fb(st->draws, st->bindless, st->ref);
   return st;
}

static void
mdi_teardown(struct gb_vk *vk, void *p)
{
   struct mdi_state *st = p;
   gb_pipe_destroy(vk, &st->p);
   if (st->sampler) vkDestroySampler(vk->dev, st->sampler, NULL);
   for (uint32_t t = 0; t < st->bindless; t++)
      gb_img_destroy(vk, &st->texs[t]);
   free(st->texs);
   gb_img_destroy(vk, &st->target);
   gb_buf_destroy(vk, &st->vb);
   gb_buf_destroy(vk, &st->ib);
   gb_buf_destroy(vk, &st->indirect);
   gb_buf_destroy(vk, &st->count_buf);
   free(st->ref);
   free(st);
}

static void
mdi_record(void *p, VkCommandBuffer cb, uint64_t k)
{
   (void)k;
   struct mdi_state *st = p;
   static const float CLEAR_ZERO[4] = {0.0f, 0.0f, 0.0f, 1.0f};
   gb_begin_render(cb, &st->target, CLEAR_ZERO);
   vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, st->p.pipe);
   vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, st->p.layout, 0, 1, &st->p.set, 0, NULL);
   vkCmdPushConstants(cb, st->p.layout, VK_SHADER_STAGE_VERTEX_BIT, 0, 4, &st->bindless);
   VkDeviceSize off = 0;
   vkCmdBindVertexBuffers(cb, 0, 1, &st->vb.buf, &off);
   vkCmdBindIndexBuffer(cb, st->ib.buf, 0, VK_INDEX_TYPE_UINT16);
   vkCmdDrawIndexedIndirectCount(cb, st->indirect.buf, 0, st->count_buf.buf, 0,
                                 st->draws, sizeof(VkDrawIndexedIndirectCommand));
   vkCmdEndRendering(cb);
}

static void
mdi_derive(const struct gb_opts *o, struct gb_result *r, const struct gb_samples *us)
{
   long draws = gb_param_int(o, "draws", 4096);
   gb_derive_rate(r, "mdraws_s", "Mdraws/s", us, (double)draws);
}

static void
mdi_check(struct gb_vk *vk, void *p, struct gb_check *chk)
{
   struct mdi_state *st = p;
   size_t count = (size_t)GB_MDI_DIM * GB_MDI_DIM;
   uint32_t *got = malloc(count * sizeof(uint32_t));
   gb_img_readback(vk, &st->target, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, got, 4u);
   gb_check_words(chk, got, st->ref, st->ref_crc, count);
   free(got);
}

static const char *const mdi_params[] = {"draws", "bindless", NULL};
static const struct gb_stream_test t_render_mdi = {
   .name = "render.mdi",
   .known_params = mdi_params,
   .default_depths = {1, 4}, .ndefault = 2,
   .setup = mdi_setup,
   .teardown = mdi_teardown,
   .record = mdi_record,
   .derive = mdi_derive,
   .check = mdi_check,
};

int gb_run_render_mdi(const struct gb_opts *o) { return gb_run_stream_test(o, &t_render_mdi); }

/* =========================================================================
 * 8. compile.pso (cold vs warm pipeline compilation latency / stutter)
 * ========================================================================= */

int
gb_run_compile_pso(const struct gb_opts *o)
{
   static const char *const known[] = {"kind", "count", "salt", NULL};
   if (!gb_check_params(o, known)) {
      gb_result_simple(o, GB_MODE_QUEUE, 1, NULL, "fail", "unknown parameter");
      return 1;
   }
   const char *kind = gb_param(o, "kind", "uber");
   bool is_comp = !strcmp(kind, "compute");
   bool is_uber = !strcmp(kind, "uber");
   bool is_gfx = !strcmp(kind, "graphics");
   if (!is_comp && !is_uber && !is_gfx) {
      gb_result_simple(o, GB_MODE_QUEUE, 1, NULL, "fail", "kind must be compute, graphics or uber");
      return 1;
   }
   long count = gb_param_int(o, "count", (long)GB_PSO_DEFAULT_COUNT);
   long salt = gb_param_int(o, "salt", 1);
   if (count < 1 || count > 256 || (salt != 0 && salt != 1)) {
      gb_result_simple(o, GB_MODE_QUEUE, 1, NULL, "fail", "count 1..256, salt 0|1");
      return 1;
   }

   struct gb_vk vk;
   gb_vk_create(&vk, o->device, 1);

   /* Per-process monotonic nonce salted into specialization constant 3 so disk
    * caches cannot falsely short-circuit cold compile measurement. */
   uint32_t nonce = salt ? (uint32_t)((gb_now_ns() ^ ((uint64_t)getpid() << 16)) | 1u) : 0x51a71234u;

   VkPipelineCacheCreateInfo cci = {.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
   VkPipelineCache cache;
   GB_VK_CHECK(vkCreatePipelineCache(vk.dev, &cci, NULL, &cache));

   VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
   VkDescriptorPool dpool = VK_NULL_HANDLE;
   VkDescriptorSet dset = VK_NULL_HANDLE;
   VkPipelineLayout layout = VK_NULL_HANDLE;
   struct gb_buf out_buf = {0};
   struct gb_img out_img = {0};
   VkShaderModule m1 = VK_NULL_HANDLE, m2 = VK_NULL_HANDLE;

   if (!is_gfx) {
      gb_buf_create(&vk, 4096, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, true, &out_buf);
      VkDescriptorSetLayoutBinding lb = {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL};
      VkDescriptorSetLayoutCreateInfo dci = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .bindingCount = 1, .pBindings = &lb};
      GB_VK_CHECK(vkCreateDescriptorSetLayout(vk.dev, &dci, NULL, &dsl));
      VkDescriptorPoolSize ps = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1};
      VkDescriptorPoolCreateInfo pci = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &ps};
      GB_VK_CHECK(vkCreateDescriptorPool(vk.dev, &pci, NULL, &dpool));
      VkDescriptorSetAllocateInfo sai = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, .descriptorPool = dpool, .descriptorSetCount = 1, .pSetLayouts = &dsl};
      GB_VK_CHECK(vkAllocateDescriptorSets(vk.dev, &sai, &dset));
      VkDescriptorBufferInfo bi = {out_buf.buf, 0, VK_WHOLE_SIZE};
      VkWriteDescriptorSet wr = {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = dset, .dstBinding = 0, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &bi};
      vkUpdateDescriptorSets(vk.dev, 1, &wr, 0, NULL);
      VkPipelineLayoutCreateInfo lci = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1, .pSetLayouts = &dsl};
      GB_VK_CHECK(vkCreatePipelineLayout(vk.dev, &lci, NULL, &layout));
      m1 = gb_shader(&vk, is_uber ? pso_uber_comp : pso_comp_comp, is_uber ? pso_uber_comp_size : pso_comp_comp_size);
   } else {
      gb_img_create(&vk, 64, 64, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, &out_img);
      VkCommandBuffer cb = gb_oneshot_begin(&vk);
      gb_img_barrier(cb, out_img.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                     VK_PIPELINE_STAGE_2_NONE, 0, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
      gb_oneshot_end(&vk, cb);
      VkPipelineLayoutCreateInfo lci = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
      GB_VK_CHECK(vkCreatePipelineLayout(vk.dev, &lci, NULL, &layout));
      m1 = gb_shader(&vk, GB_SPV(pso_gfx_vert));
      m2 = gb_shader(&vk, GB_SPV(pso_gfx_frag));
   }

   VkSpecializationMapEntry me[4] = {
      {.constantID = 0, .offset = 0, .size = 4},
      {.constantID = 1, .offset = 4, .size = 4},
      {.constantID = 2, .offset = 8, .size = 4},
      {.constantID = 3, .offset = 12, .size = 4},
   };
   VkPipeline *pipes = calloc((size_t)count, sizeof(VkPipeline));
   struct gb_samples cold_s = {0}, warm_s = {0};
   double cold_total = 0, warm_total = 0;

   for (int phase = 0; phase < 2; phase++) {
      uint64_t t_batch0 = gb_now_ns();
      for (long i = 0; i < count; i++) {
         uint32_t sd[4] = {
            (uint32_t)(i + 1),
            (uint32_t)(is_gfx ? (4u + (uint32_t)(i & 7)) : (8u + (uint32_t)(i & 15))),
            (uint32_t)(is_uber ? 7u : 3u),
            nonce + (uint32_t)i * 2654435761u,
         };
         VkSpecializationInfo si = {.mapEntryCount = 4, .pMapEntries = me, .dataSize = sizeof(sd), .pData = sd};
         uint64_t t0 = gb_now_ns();
         if (!is_gfx) {
            VkComputePipelineCreateInfo ci = {
               .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
               .stage = {
                  .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                  .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                  .module = m1, .pName = "main", .pSpecializationInfo = &si,
               },
               .layout = layout,
            };
            GB_VK_CHECK(vkCreateComputePipelines(vk.dev, cache, 1, &ci, NULL, &pipes[i]));
         } else {
            VkPipelineShaderStageCreateInfo st[2] = {
               {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = m1, .pName = "main", .pSpecializationInfo = &si},
               {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = m2, .pName = "main", .pSpecializationInfo = &si},
            };
            VkPipelineVertexInputStateCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
            VkPipelineInputAssemblyStateCreateInfo ia = {.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO, .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST};
            VkPipelineViewportStateCreateInfo vp = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, .viewportCount = 1, .scissorCount = 1};
            VkPipelineRasterizationStateCreateInfo rs = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO, .polygonMode = VK_POLYGON_MODE_FILL, .cullMode = VK_CULL_MODE_NONE, .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE, .lineWidth = 1.0f};
            VkPipelineMultisampleStateCreateInfo ms = {.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO, .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT};
            VkPipelineColorBlendAttachmentState ba = {.colorWriteMask = 0xf};
            VkPipelineColorBlendStateCreateInfo cbs = {.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO, .attachmentCount = 1, .pAttachments = &ba};
            VkDynamicState dyn[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
            VkPipelineDynamicStateCreateInfo dyns = {.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO, .dynamicStateCount = 2, .pDynamicStates = dyn};
            VkFormat fmt = VK_FORMAT_R8G8B8A8_UNORM;
            VkPipelineRenderingCreateInfo ri = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO, .colorAttachmentCount = 1, .pColorAttachmentFormats = &fmt};
            VkGraphicsPipelineCreateInfo ci = {
               .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, .pNext = &ri,
               .stageCount = 2, .pStages = st,
               .pVertexInputState = &vi, .pInputAssemblyState = &ia,
               .pViewportState = &vp, .pRasterizationState = &rs,
               .pMultisampleState = &ms, .pColorBlendState = &cbs, .pDynamicState = &dyns,
               .layout = layout,
            };
            GB_VK_CHECK(vkCreateGraphicsPipelines(vk.dev, cache, 1, &ci, NULL, &pipes[i]));
         }
         uint64_t t1 = gb_now_ns();
         gb_samples_push(phase == 0 ? &cold_s : &warm_s, (double)(t1 - t0) / 1e6);
      }
      uint64_t t_batch1 = gb_now_ns();
      if (phase == 0) {
         cold_total = (double)(t_batch1 - t_batch0) / 1e6;
         for (long i = 0; i < count; i++) {
            vkDestroyPipeline(vk.dev, pipes[i], NULL);
            pipes[i] = VK_NULL_HANDLE;
         }
      } else {
         warm_total = (double)(t_batch1 - t_batch0) / 1e6;
      }
   }

   /* Smoke-execute the last compiled pipeline so compilation is proven live. */
   VkCommandBuffer cb = gb_oneshot_begin(&vk);
   if (!is_gfx) {
      vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipes[count - 1]);
      vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &dset, 0, NULL);
      vkCmdDispatch(cb, 1, 1, 1);
   } else {
      static const float clr[4] = {0, 0, 0, 1};
      gb_begin_render(cb, &out_img, clr);
      vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipes[count - 1]);
      vkCmdDraw(cb, 3, 1, 0, 0);
      vkCmdEndRendering(cb);
   }
   gb_oneshot_end(&vk, cb);

   struct gb_result r;
   gb_result_begin(&r, o, GB_MODE_QUEUE, 1, NULL);
   gb_result_metric(&r, "cold_per_pso_ms", "ms", true, &cold_s, true);
   gb_result_metric1(&r, "cold_total_ms", "ms", true, cold_total);
   gb_result_metric(&r, "warm_per_pso_ms", "ms", true, &warm_s, true);
   gb_result_metric1(&r, "warm_total_ms", "ms", true, warm_total);
   gb_result_metric1(&r, "cache_speedup", "x", false, warm_total > 0.0 ? cold_total / warm_total : 0.0);
   gb_result_end_metrics(&r);
   gb_json_obj_begin(&r.j, "counts");
   gb_json_int(&r.j, "pipelines", count);
   gb_json_obj_end(&r.j);
   gb_result_finish(&r, "ok", "cold_per_pso_ms", NULL);

   for (long i = 0; i < count; i++)
      if (pipes[i]) vkDestroyPipeline(vk.dev, pipes[i], NULL);
   free(pipes);
   gb_samples_free(&cold_s);
   gb_samples_free(&warm_s);
   vkDestroyShaderModule(vk.dev, m1, NULL);
   if (m2) vkDestroyShaderModule(vk.dev, m2, NULL);
   vkDestroyPipelineLayout(vk.dev, layout, NULL);
   if (dpool) vkDestroyDescriptorPool(vk.dev, dpool, NULL);
   if (dsl) vkDestroyDescriptorSetLayout(vk.dev, dsl, NULL);
   if (out_buf.buf) gb_buf_destroy(&vk, &out_buf);
   if (out_img.img) gb_img_destroy(&vk, &out_img);
   vkDestroyPipelineCache(vk.dev, cache, NULL);
   gb_vk_destroy(&vk);
   return 0;
}

/* =========================================================================
 * 9. compress.bw (mrt4 | rgba32f | msaa4 | array | storage)
 * Exposes lossless framebuffer / MSAA / layered / storage-image compression.
 * ========================================================================= */

enum cmp_kind { CMP_MRT4, CMP_RGBA32F, CMP_MSAA4, CMP_ARRAY, CMP_STORAGE };

struct cmp_state {
   enum cmp_kind kind;
   uint32_t w, h;
   struct gb_img g[4], msaa_img, arr_img, final_img;
   VkImageView arr_views[6];
   VkSampler sampler;
   struct gb_pipe p_write, p_read;
   double logical_bytes;
};

static void
cmp_make_gfx(struct gb_vk *vk, const uint32_t *fs, size_t fs_bytes, int n_att, const VkFormat *fmts,
             VkSampleCountFlagBits samples, int n_samplers, bool array_tex, uint32_t push_size, struct gb_pipe *p)
{
   memset(p, 0, sizeof(*p));
   VkDescriptorSetLayoutBinding lbs[4];
   for (int i = 0; i < n_samplers; i++)
      lbs[i] = (VkDescriptorSetLayoutBinding){
         .binding = (uint32_t)i, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
         .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
      };
   (void)array_tex;
   if (n_samplers) {
      VkDescriptorSetLayoutCreateInfo dci = {
         .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
         .bindingCount = (uint32_t)n_samplers, .pBindings = lbs,
      };
      GB_VK_CHECK(vkCreateDescriptorSetLayout(vk->dev, &dci, NULL, &p->dsl));
      VkDescriptorPoolSize ps = {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, (uint32_t)n_samplers};
      VkDescriptorPoolCreateInfo pci = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &ps};
      GB_VK_CHECK(vkCreateDescriptorPool(vk->dev, &pci, NULL, &p->dpool));
      VkDescriptorSetAllocateInfo sai = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, .descriptorPool = p->dpool, .descriptorSetCount = 1, .pSetLayouts = &p->dsl};
      GB_VK_CHECK(vkAllocateDescriptorSets(vk->dev, &sai, &p->set));
   }
   VkPushConstantRange pr = {VK_SHADER_STAGE_FRAGMENT_BIT, 0, push_size};
   VkPipelineLayoutCreateInfo lci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = n_samplers ? 1u : 0u, .pSetLayouts = &p->dsl,
      .pushConstantRangeCount = push_size ? 1u : 0u, .pPushConstantRanges = &pr,
   };
   GB_VK_CHECK(vkCreatePipelineLayout(vk->dev, &lci, NULL, &p->layout));
   p->push_stages = VK_SHADER_STAGE_FRAGMENT_BIT;
   p->push_size = push_size;

   VkShaderModule mv = gb_shader(vk, GB_SPV(fs_vert));
   VkShaderModule mf = gb_shader(vk, fs, fs_bytes);
   VkPipelineShaderStageCreateInfo st[2] = {
      {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = mv, .pName = "main"},
      {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = mf, .pName = "main"},
   };
   VkPipelineVertexInputStateCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
   VkPipelineInputAssemblyStateCreateInfo ia = {.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO, .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST};
   VkPipelineViewportStateCreateInfo vp = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, .viewportCount = 1, .scissorCount = 1};
   VkPipelineRasterizationStateCreateInfo rs = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO, .polygonMode = VK_POLYGON_MODE_FILL, .cullMode = VK_CULL_MODE_NONE, .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE, .lineWidth = 1.0f};
   VkPipelineMultisampleStateCreateInfo ms = {.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO, .rasterizationSamples = samples};
   VkPipelineColorBlendAttachmentState ba[4];
   for (int i = 0; i < 4; i++) ba[i] = (VkPipelineColorBlendAttachmentState){.colorWriteMask = 0xf};
   VkPipelineColorBlendStateCreateInfo cbs = {.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO, .attachmentCount = (uint32_t)n_att, .pAttachments = ba};
   VkDynamicState dyn[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
   VkPipelineDynamicStateCreateInfo dyns = {.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO, .dynamicStateCount = 2, .pDynamicStates = dyn};
   VkPipelineRenderingCreateInfo ri = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
      .colorAttachmentCount = (uint32_t)n_att, .pColorAttachmentFormats = fmts,
   };
   VkGraphicsPipelineCreateInfo gci = {
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, .pNext = &ri,
      .stageCount = 2, .pStages = st,
      .pVertexInputState = &vi, .pInputAssemblyState = &ia,
      .pViewportState = &vp, .pRasterizationState = &rs,
      .pMultisampleState = &ms, .pColorBlendState = &cbs, .pDynamicState = &dyns,
      .layout = p->layout,
   };
   GB_VK_CHECK(vkCreateGraphicsPipelines(vk->dev, VK_NULL_HANDLE, 1, &gci, NULL, &p->pipe));
   vkDestroyShaderModule(vk->dev, mv, NULL);
   vkDestroyShaderModule(vk->dev, mf, NULL);
}

static void *
cmp_setup(struct gb_vk *vk, const struct gb_opts *o, char *err, size_t n)
{
   const char *kind = gb_param(o, "kind", "mrt4");
   enum cmp_kind k;
   if (!strcmp(kind, "mrt4")) k = CMP_MRT4;
   else if (!strcmp(kind, "rgba32f")) k = CMP_RGBA32F;
   else if (!strcmp(kind, "msaa4")) k = CMP_MSAA4;
   else if (!strcmp(kind, "array")) k = CMP_ARRAY;
   else if (!strcmp(kind, "storage")) k = CMP_STORAGE;
   else { snprintf(err, n, "kind must be mrt4, rgba32f, msaa4, array or storage"); return NULL; }

   uint32_t w = 0, h = 0;
   if (!parse_res_w1(gb_param(o, "res", "1440p"), &w, &h)) {
      snprintf(err, n, "res must be 720p, 1080p, 1440p or 2160p");
      return NULL;
   }
   struct cmp_state *st = calloc(1, sizeof(*st));
   st->kind = k;
   if (k == CMP_ARRAY) {
      st->w = w >= 2560u ? 1024u : 512u;
      st->h = st->w;
   } else {
      st->w = w; st->h = h;
   }

   VkSamplerCreateInfo sci = {
      .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
      .magFilter = VK_FILTER_NEAREST, .minFilter = VK_FILTER_NEAREST,
      .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
      .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
   };
   GB_VK_CHECK(vkCreateSampler(vk->dev, &sci, NULL, &st->sampler));

   gb_img_create(vk, st->w, st->h, VK_FORMAT_R8G8B8A8_UNORM,
                 VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, &st->final_img);
   VkFormat rgba8 = VK_FORMAT_R8G8B8A8_UNORM;
   VkFormat rgba32f = VK_FORMAT_R32G32B32A32_SFLOAT;

   VkCommandBuffer cb = gb_oneshot_begin(vk);
   gb_img_barrier(cb, st->final_img.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                  VK_PIPELINE_STAGE_2_NONE, 0, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);

   if (k == CMP_MRT4) {
      st->logical_bytes = 44.0 * (double)st->w * st->h;
      for (int i = 0; i < 4; i++) {
         gb_img_create(vk, st->w, st->h, DEF_GFMT[i],
                       VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, &st->g[i]);
         gb_img_barrier(cb, st->g[i].img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                        VK_PIPELINE_STAGE_2_NONE, 0, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
      }
      cmp_make_gfx(vk, GB_SPV(compress_mrt4_frag), 4, DEF_GFMT, VK_SAMPLE_COUNT_1_BIT, 0, false, 16, &st->p_write);
      cmp_make_gfx(vk, GB_SPV(compress_sample4_frag), 1, &rgba8, VK_SAMPLE_COUNT_1_BIT, 4, false, 0, &st->p_read);
      VkDescriptorImageInfo ii[4];
      VkWriteDescriptorSet wr[4];
      for (int i = 0; i < 4; i++) {
         ii[i] = (VkDescriptorImageInfo){st->sampler, st->g[i].view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
         wr[i] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = st->p_read.set,
            .dstBinding = (uint32_t)i, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .pImageInfo = &ii[i]};
      }
      vkUpdateDescriptorSets(vk->dev, 4, wr, 0, NULL);
   } else if (k == CMP_RGBA32F) {
      st->logical_bytes = 36.0 * (double)st->w * st->h;
      gb_img_create(vk, st->w, st->h, rgba32f, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, &st->g[0]);
      gb_img_barrier(cb, st->g[0].img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                     VK_PIPELINE_STAGE_2_NONE, 0, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
      cmp_make_gfx(vk, GB_SPV(compress_write1_frag), 1, &rgba32f, VK_SAMPLE_COUNT_1_BIT, 0, false, 16, &st->p_write);
      cmp_make_gfx(vk, GB_SPV(compress_sample1_frag), 1, &rgba8, VK_SAMPLE_COUNT_1_BIT, 1, false, 0, &st->p_read);
      VkDescriptorImageInfo ii = {st->sampler, st->g[0].view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
      VkWriteDescriptorSet wr = {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = st->p_read.set,
         .dstBinding = 0, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .pImageInfo = &ii};
      vkUpdateDescriptorSets(vk->dev, 1, &wr, 0, NULL);
   } else if (k == CMP_MSAA4) {
      st->logical_bytes = 28.0 * (double)st->w * st->h;
      gb_img_create_ex(vk, st->w, st->h, rgba8, VK_SAMPLE_COUNT_4_BIT, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
                       VK_IMAGE_ASPECT_COLOR_BIT, &st->msaa_img);
      gb_img_create(vk, st->w, st->h, rgba8, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, &st->g[0]);
      gb_img_barrier(cb, st->msaa_img.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                     VK_PIPELINE_STAGE_2_NONE, 0, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
      gb_img_barrier(cb, st->g[0].img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                     VK_PIPELINE_STAGE_2_NONE, 0, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
      cmp_make_gfx(vk, GB_SPV(compress_write1_frag), 1, &rgba8, VK_SAMPLE_COUNT_4_BIT, 0, false, 16, &st->p_write);
      cmp_make_gfx(vk, GB_SPV(compress_sample1_frag), 1, &rgba8, VK_SAMPLE_COUNT_1_BIT, 1, false, 0, &st->p_read);
      VkDescriptorImageInfo ii = {st->sampler, st->g[0].view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
      VkWriteDescriptorSet wr = {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = st->p_read.set,
         .dstBinding = 0, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .pImageInfo = &ii};
      vkUpdateDescriptorSets(vk->dev, 1, &wr, 0, NULL);
   } else if (k == CMP_ARRAY) {
      st->logical_bytes = 52.0 * (double)st->w * st->h;
      VkImageCreateInfo ici = {
         .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D, .format = rgba8,
         .extent = {st->w, st->h, 1}, .mipLevels = 1, .arrayLayers = 6,
         .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
         .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
         .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
      };
      GB_VK_CHECK(vkCreateImage(vk->dev, &ici, NULL, &st->arr_img.img));
      st->arr_img.w = st->w; st->arr_img.h = st->h; st->arr_img.fmt = rgba8;
      VkMemoryRequirements req;
      vkGetImageMemoryRequirements(vk->dev, st->arr_img.img, &req);
      uint32_t mt = vk->mem_dev;
      if (!(req.memoryTypeBits & (1u << mt)))
         for (uint32_t i = 0; i < vk->memprops.memoryTypeCount; i++)
            if (req.memoryTypeBits & (1u << i)) { mt = i; break; }
      VkMemoryAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = req.size, .memoryTypeIndex = mt};
      GB_VK_CHECK(vkAllocateMemory(vk->dev, &ai, NULL, &st->arr_img.mem));
      GB_VK_CHECK(vkBindImageMemory(vk->dev, st->arr_img.img, st->arr_img.mem, 0));
      VkImageViewCreateInfo avi = {
         .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = st->arr_img.img,
         .viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY, .format = rgba8,
         .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 6},
      };
      GB_VK_CHECK(vkCreateImageView(vk->dev, &avi, NULL, &st->arr_img.view));
      for (uint32_t l = 0; l < 6; l++) {
         VkImageViewCreateInfo lvi = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = st->arr_img.img,
            .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = rgba8,
            .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, l, 1},
         };
         GB_VK_CHECK(vkCreateImageView(vk->dev, &lvi, NULL, &st->arr_views[l]));
      }
      VkImageMemoryBarrier2 ab = {
         .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
         .srcStageMask = VK_PIPELINE_STAGE_2_NONE, .dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
         .dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
         .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
         .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
         .image = st->arr_img.img, .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 6},
      };
      VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &ab};
      vkCmdPipelineBarrier2(cb, &di);

      cmp_make_gfx(vk, GB_SPV(compress_write1_frag), 1, &rgba8, VK_SAMPLE_COUNT_1_BIT, 0, false, 16, &st->p_write);
      cmp_make_gfx(vk, GB_SPV(compress_array_frag), 1, &rgba8, VK_SAMPLE_COUNT_1_BIT, 1, true, 0, &st->p_read);
      VkDescriptorImageInfo ii = {st->sampler, st->arr_img.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
      VkWriteDescriptorSet wr = {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = st->p_read.set,
         .dstBinding = 0, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .pImageInfo = &ii};
      vkUpdateDescriptorSets(vk->dev, 1, &wr, 0, NULL);
   } else { /* CMP_STORAGE */
      st->logical_bytes = 12.0 * (double)st->w * st->h;
      gb_img_create(vk, st->w, st->h, rgba8, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, &st->g[0]);
      gb_img_barrier(cb, st->g[0].img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                     VK_PIPELINE_STAGE_2_NONE, 0, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

      /* Storage-image compute pipeline */
      VkDescriptorSetLayoutBinding lb = {0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL};
      VkDescriptorSetLayoutCreateInfo dci = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .bindingCount = 1, .pBindings = &lb};
      GB_VK_CHECK(vkCreateDescriptorSetLayout(vk->dev, &dci, NULL, &st->p_write.dsl));
      VkDescriptorPoolSize ps = {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1};
      VkDescriptorPoolCreateInfo pci = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &ps};
      GB_VK_CHECK(vkCreateDescriptorPool(vk->dev, &pci, NULL, &st->p_write.dpool));
      VkDescriptorSetAllocateInfo sai = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, .descriptorPool = st->p_write.dpool, .descriptorSetCount = 1, .pSetLayouts = &st->p_write.dsl};
      GB_VK_CHECK(vkAllocateDescriptorSets(vk->dev, &sai, &st->p_write.set));
      VkDescriptorImageInfo sii = {VK_NULL_HANDLE, st->g[0].view, VK_IMAGE_LAYOUT_GENERAL};
      VkWriteDescriptorSet swr = {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = st->p_write.set,
         .dstBinding = 0, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .pImageInfo = &sii};
      vkUpdateDescriptorSets(vk->dev, 1, &swr, 0, NULL);
      VkPushConstantRange pr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 16};
      VkPipelineLayoutCreateInfo lci = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1, .pSetLayouts = &st->p_write.dsl, .pushConstantRangeCount = 1, .pPushConstantRanges = &pr};
      GB_VK_CHECK(vkCreatePipelineLayout(vk->dev, &lci, NULL, &st->p_write.layout));
      VkShaderModule m = gb_shader(vk, GB_SPV(compress_store_comp));
      VkComputePipelineCreateInfo cci = {
         .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
         .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = m, .pName = "main"},
         .layout = st->p_write.layout,
      };
      GB_VK_CHECK(vkCreateComputePipelines(vk->dev, VK_NULL_HANDLE, 1, &cci, NULL, &st->p_write.pipe));
      vkDestroyShaderModule(vk->dev, m, NULL);

      cmp_make_gfx(vk, GB_SPV(compress_sample1_frag), 1, &rgba8, VK_SAMPLE_COUNT_1_BIT, 1, false, 0, &st->p_read);
      VkDescriptorImageInfo ii = {st->sampler, st->g[0].view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
      VkWriteDescriptorSet wr = {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = st->p_read.set,
         .dstBinding = 0, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .pImageInfo = &ii};
      vkUpdateDescriptorSets(vk->dev, 1, &wr, 0, NULL);
   }
   gb_oneshot_end(vk, cb);
   return st;
}

static void
cmp_teardown(struct gb_vk *vk, void *p)
{
   struct cmp_state *st = p;
   gb_pipe_destroy(vk, &st->p_write);
   gb_pipe_destroy(vk, &st->p_read);
   if (st->sampler) vkDestroySampler(vk->dev, st->sampler, NULL);
   for (int i = 0; i < 4; i++) if (st->g[i].img) gb_img_destroy(vk, &st->g[i]);
   if (st->msaa_img.img) gb_img_destroy(vk, &st->msaa_img);
   for (int i = 0; i < 6; i++) if (st->arr_views[i]) vkDestroyImageView(vk->dev, st->arr_views[i], NULL);
   if (st->arr_img.img) gb_img_destroy(vk, &st->arr_img);
   gb_img_destroy(vk, &st->final_img);
   free(st);
}

static void
cmp_record(void *p, VkCommandBuffer cb, uint64_t k)
{
   struct cmp_state *st = p;
   float kf = (float)(k & 15u) * 0.03125f;
   VkViewport vp = {0, 0, (float)st->w, (float)st->h, 0.0f, 1.0f};
   VkRect2D sc = {{0, 0}, {st->w, st->h}};

   if (st->kind == CMP_MRT4) {
      VkRenderingAttachmentInfo ca[4];
      for (int i = 0; i < 4; i++)
         ca[i] = (VkRenderingAttachmentInfo){.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
            .imageView = st->g[i].view, .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            .loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE, .storeOp = VK_ATTACHMENT_STORE_OP_STORE};
      VkRenderingInfo ri = {.sType = VK_STRUCTURE_TYPE_RENDERING_INFO, .renderArea = sc, .layerCount = 1, .colorAttachmentCount = 4, .pColorAttachments = ca};
      vkCmdBeginRendering(cb, &ri);
      vkCmdSetViewport(cb, 0, 1, &vp); vkCmdSetScissor(cb, 0, 1, &sc);
      vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, st->p_write.pipe);
      float pc[4] = {kf, 0.25f, 0.5f, 0.0f};
      vkCmdPushConstants(cb, st->p_write.layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, 16, pc);
      vkCmdDraw(cb, 3, 1, 0, 0);
      vkCmdEndRendering(cb);
      for (int i = 0; i < 4; i++)
         gb_img_barrier(cb, st->g[i].img, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                        VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
   } else if (st->kind == CMP_RGBA32F || st->kind == CMP_MSAA4) {
      VkRenderingAttachmentInfo ca = {
         .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
         .imageView = st->kind == CMP_MSAA4 ? st->msaa_img.view : st->g[0].view,
         .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
         .resolveMode = st->kind == CMP_MSAA4 ? VK_RESOLVE_MODE_AVERAGE_BIT : VK_RESOLVE_MODE_NONE,
         .resolveImageView = st->kind == CMP_MSAA4 ? st->g[0].view : VK_NULL_HANDLE,
         .resolveImageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
         .loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
         .storeOp = st->kind == CMP_MSAA4 ? VK_ATTACHMENT_STORE_OP_DONT_CARE : VK_ATTACHMENT_STORE_OP_STORE,
      };
      VkRenderingInfo ri = {.sType = VK_STRUCTURE_TYPE_RENDERING_INFO, .renderArea = sc, .layerCount = 1, .colorAttachmentCount = 1, .pColorAttachments = &ca};
      vkCmdBeginRendering(cb, &ri);
      vkCmdSetViewport(cb, 0, 1, &vp); vkCmdSetScissor(cb, 0, 1, &sc);
      vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, st->p_write.pipe);
      float pc[4] = {kf, 0.25f, 0.5f, 0.0f};
      vkCmdPushConstants(cb, st->p_write.layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, 16, pc);
      vkCmdDraw(cb, 3, 1, 0, 0);
      vkCmdEndRendering(cb);
      gb_img_barrier(cb, st->g[0].img, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                     VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                     VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
   } else if (st->kind == CMP_ARRAY) {
      for (uint32_t l = 0; l < 6; l++) {
         VkRenderingAttachmentInfo ca = {.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
            .imageView = st->arr_views[l], .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            .loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE, .storeOp = VK_ATTACHMENT_STORE_OP_STORE};
         VkRenderingInfo ri = {.sType = VK_STRUCTURE_TYPE_RENDERING_INFO, .renderArea = sc, .layerCount = 1, .colorAttachmentCount = 1, .pColorAttachments = &ca};
         vkCmdBeginRendering(cb, &ri);
         vkCmdSetViewport(cb, 0, 1, &vp); vkCmdSetScissor(cb, 0, 1, &sc);
         vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, st->p_write.pipe);
         float pc[4] = {kf, (float)l * 0.125f, (float)l * 0.0625f, 0.0f};
         vkCmdPushConstants(cb, st->p_write.layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, 16, pc);
         vkCmdDraw(cb, 3, 1, 0, 0);
         vkCmdEndRendering(cb);
      }
      VkImageMemoryBarrier2 ab = {
         .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
         .srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, .srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
         .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, .dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
         .oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, .newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
         .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
         .image = st->arr_img.img, .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 6},
      };
      VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &ab};
      vkCmdPipelineBarrier2(cb, &di);
   } else { /* CMP_STORAGE */
      struct { int32_t w, h; float k, pad; } pc = {(int32_t)st->w, (int32_t)st->h, kf, 0.0f};
      vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, st->p_write.pipe);
      vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, st->p_write.layout, 0, 1, &st->p_write.set, 0, NULL);
      vkCmdPushConstants(cb, st->p_write.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
      vkCmdDispatch(cb, (st->w + 15u) / 16u, (st->h + 15u) / 16u, 1);
      gb_img_barrier(cb, st->g[0].img, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                     VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                     VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
   }

   /* Pass 2: sample back into Final RGBA8 */
   VkRenderingAttachmentInfo cf = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
      .imageView = st->final_img.view, .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
      .loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE, .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
   };
   VkRenderingInfo rf = {.sType = VK_STRUCTURE_TYPE_RENDERING_INFO, .renderArea = sc, .layerCount = 1, .colorAttachmentCount = 1, .pColorAttachments = &cf};
   vkCmdBeginRendering(cb, &rf);
   vkCmdSetViewport(cb, 0, 1, &vp); vkCmdSetScissor(cb, 0, 1, &sc);
   vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, st->p_read.pipe);
   vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, st->p_read.layout, 0, 1, &st->p_read.set, 0, NULL);
   vkCmdDraw(cb, 3, 1, 0, 0);
   vkCmdEndRendering(cb);

   /* Transition back for next submission */
   if (st->kind == CMP_MRT4) {
      for (int i = 0; i < 4; i++)
         gb_img_barrier(cb, st->g[i].img, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                        VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
   } else if (st->kind == CMP_ARRAY) {
      VkImageMemoryBarrier2 ab = {
         .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
         .srcStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, .srcAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
         .dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, .dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
         .oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, .newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
         .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
         .image = st->arr_img.img, .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 6},
      };
      VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &ab};
      vkCmdPipelineBarrier2(cb, &di);
   } else if (st->kind == CMP_STORAGE) {
      gb_img_barrier(cb, st->g[0].img, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
                     VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                     VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
   } else {
      gb_img_barrier(cb, st->g[0].img, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                     VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                     VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
   }
}

static void
cmp_derive(const struct gb_opts *o, struct gb_result *r, const struct gb_samples *us)
{
   const char *kind = gb_param(o, "kind", "mrt4");
   uint32_t w = 2560, h = 1440;
   parse_res_w1(gb_param(o, "res", "1440p"), &w, &h);
   double b = 0;
   if (!strcmp(kind, "mrt4")) b = 44.0 * w * h;
   else if (!strcmp(kind, "rgba32f")) b = 36.0 * w * h;
   else if (!strcmp(kind, "msaa4")) b = 28.0 * w * h;
   else if (!strcmp(kind, "array")) { double d = w >= 2560u ? 1024.0 : 512.0; b = 52.0 * d * d; }
   else b = 12.0 * w * h;
   gb_derive_rate(r, "gbps", "GB/s", us, b / 1e9 * 1e6);
}

static void
cmp_check(struct gb_vk *vk, void *p, struct gb_check *chk)
{
   struct cmp_state *st = p;
   size_t bytes = (size_t)st->w * st->h * 4u;
   uint8_t *got = malloc(bytes);
   gb_img_readback(vk, &st->final_img, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, got, 4u);
   chk->done = true;
   chk->crc = gb_crc32(got, bytes);
   chk->expected = chk->crc;
   chk->valid = true;
   uint32_t nonzero = 0;
   for (uint32_t y = 0; y < st->h; y += st->h / 8u)
      for (uint32_t x = 0; x < st->w; x += st->w / 8u) {
         const uint8_t *px = got + ((size_t)y * st->w + x) * 4u;
         if (px[3] != 255u) chk->valid = false;
         if (px[0] || px[1] || px[2]) nonzero++;
      }
   if (nonzero < 32u) {
      gb_log("compress.bw output appears blank (nonzero=%u)", nonzero);
      chk->valid = false;
      chk->expected = 0u;
   }
   free(got);
}

static const char *const cmp_params[] = {"kind", "res", NULL};
static const struct gb_stream_test t_compress_bw = {
   .name = "compress.bw",
   .known_params = cmp_params,
   .default_depths = {1, 4}, .ndefault = 2,
   .setup = cmp_setup,
   .teardown = cmp_teardown,
   .record = cmp_record,
   .derive = cmp_derive,
   .check = cmp_check,
};

int gb_run_compress_bw(const struct gb_opts *o) { return gb_run_stream_test(o, &t_compress_bw); }

/* =========================================================================
 * 10. overlap.passes (comp_comp | comp_render | render_render | async_cr)
 * Measures barrier-free pass overlap vs explicit barrier serialization.
 * ========================================================================= */

enum ov_kind { OV_CC, OV_CR, OV_RR, OV_ASYNC_CR };
enum ov_phase { OVP_SOLO_A, OVP_SOLO_B, OVP_NOBARRIER, OVP_BARRIER };

struct ov_state {
   enum ov_kind kind;
   enum ov_phase phase;
   struct gb_pipe pc_a, pc_b, pr_a, pr_b;
   struct gb_buf ba, bb;
   struct gb_img ia, ib;
};

static void
ov_rec_comp(struct gb_pipe *p, VkCommandBuffer cb)
{
   struct { uint32_t iters; float a, b; } pc = {GB_ALU_ITERS, GB_ALU_A, GB_ALU_B};
   vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p->pipe);
   vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p->layout, 0, 1, &p->set, 0, NULL);
   vkCmdPushConstants(cb, p->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
   vkCmdDispatch(cb, GB_ALU_GROUPS / 2u, 1, 1);
}

static void
ov_rec_render(struct gb_pipe *p, struct gb_img *im, VkCommandBuffer cb, uint64_t k)
{
   static const float clr[4] = {0.1f, 0.2f, 0.3f, 1.0f};
   float color[4] = {(float)(k & 255u) / 255.0f, 0.5f, 0.25f, 0.5f};
   gb_begin_render(cb, im, clr);
   vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, p->pipe);
   vkCmdPushConstants(cb, p->layout, p->push_stages, 0, 16, color);
   vkCmdDraw(cb, 3, GB_FILL_LAYERS / 2u, 0, 0);
   vkCmdEndRendering(cb);
}

static void
ov_record(void *p, VkCommandBuffer cb, uint64_t k)
{
   struct ov_state *st = p;
   bool run_a = st->phase != OVP_SOLO_B;
   bool run_b = st->phase != OVP_SOLO_A;
   if (run_a) {
      if (st->kind == OV_RR) ov_rec_render(&st->pr_a, &st->ia, cb, k);
      else ov_rec_comp(&st->pc_a, cb);
   }
   if (st->phase == OVP_BARRIER) {
      VkMemoryBarrier2 mb = {
         .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
         .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
         .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
         .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
         .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
      };
      VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &mb};
      vkCmdPipelineBarrier2(cb, &di);
   }
   if (run_b) {
      if (st->kind == OV_CC) ov_rec_comp(&st->pc_b, cb);
      else ov_rec_render(&st->pr_b, &st->ib, cb, k + 1);
   }
}

static void
ov_rec_only_a(void *p, VkCommandBuffer cb, uint64_t k)
{
   (void)k;
   struct ov_state *st = p;
   ov_rec_comp(&st->pc_a, cb);
}

static void
ov_rec_only_b(void *p, VkCommandBuffer cb, uint64_t k)
{
   struct ov_state *st = p;
   ov_rec_render(&st->pr_b, &st->ib, cb, k);
}

int
gb_run_overlap_passes(const struct gb_opts *o)
{
   static const char *const known[] = {"kind", NULL};
   if (!gb_check_params(o, known)) {
      gb_result_simple(o, GB_MODE_QUEUE, 1, NULL, "fail", "unknown parameter");
      return 1;
   }
   const char *kind = gb_param(o, "kind", "comp_comp");
   enum ov_kind k;
   if (!strcmp(kind, "comp_comp")) k = OV_CC;
   else if (!strcmp(kind, "comp_render")) k = OV_CR;
   else if (!strcmp(kind, "render_render")) k = OV_RR;
   else if (!strcmp(kind, "async_cr")) k = OV_ASYNC_CR;
   else {
      gb_result_simple(o, GB_MODE_QUEUE, 1, NULL, "fail", "kind must be comp_comp, comp_render, render_render or async_cr");
      return 1;
   }
   int need_q = (k == OV_ASYNC_CR) ? 2 : 1;
   if (need_q == 2 && gb_vk_probe_queue_count(o->device) < 2) {
      gb_result_simple(o, GB_MODE_QUEUES, 2, NULL, "skip", "device has 1 queue");
      return 0;
   }
   struct gb_vk vk;
   gb_vk_create(&vk, o->device, need_q);
   struct ov_state st = {.kind = k};

   size_t cbytes = (size_t)(GB_ALU_GROUPS / 2u) * GB_ALU_LOCAL * 4u;
   if (k != OV_RR) {
      gb_buf_create(&vk, cbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, false, &st.ba);
      gb_compute_pipe(&vk, GB_SPV(alu_comp), 1, 12, &st.pc_a);
      struct gb_buf *b[1] = {&st.ba};
      gb_pipe_set_buffers(&vk, &st.pc_a, b, 1);
   }
   if (k == OV_CC) {
      gb_buf_create(&vk, cbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, false, &st.bb);
      gb_compute_pipe(&vk, GB_SPV(alu_comp), 1, 12, &st.pc_b);
      struct gb_buf *b[1] = {&st.bb};
      gb_pipe_set_buffers(&vk, &st.pc_b, b, 1);
   }
   if (k == OV_RR) {
      gb_img_create(&vk, 1920, 1080, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, &st.ia);
      gb_graphics_pipe(&vk, GB_SPV(fs_vert), GB_SPV(color_frag), VK_FORMAT_R8G8B8A8_UNORM, GB_BLEND_ALPHA, 16, &st.pr_a);
      VkCommandBuffer cb = gb_oneshot_begin(&vk);
      gb_img_barrier(cb, st.ia.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                     VK_PIPELINE_STAGE_2_NONE, 0, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
      gb_oneshot_end(&vk, cb);
   }
   if (k != OV_CC) {
      gb_img_create(&vk, 1920, 1080, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, &st.ib);
      gb_graphics_pipe(&vk, GB_SPV(fs_vert), GB_SPV(color_frag), VK_FORMAT_R8G8B8A8_UNORM, GB_BLEND_ALPHA, 16, &st.pr_b);
      VkCommandBuffer cb = gb_oneshot_begin(&vk);
      gb_img_barrier(cb, st.ib.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                     VK_PIPELINE_STAGE_2_NONE, 0, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
      gb_oneshot_end(&vk, cb);
   }

   const uint64_t W = (uint64_t)o->warmup_ms * 1000000ull, R = (uint64_t)o->run_ms * 1000000ull;
   struct gb_pass pa = {0}, pb = {0}, pnb = {0}, pbar = {0};
   bool bad = false;

   if (k != OV_ASYNC_CR) {
      const int depth = 2;
      struct gb_stream s;
      gb_stream_init(&s, &vk, 0, depth, ov_record, NULL, &st);
      st.phase = OVP_SOLO_A;    gb_stream_run(&s, false, gb_now_ns(), W, o->runs, R, &pa);
      st.phase = OVP_SOLO_B;    gb_stream_run(&s, false, gb_now_ns(), W, o->runs, R, &pb);
      st.phase = OVP_NOBARRIER; gb_stream_run(&s, false, gb_now_ns(), W, o->runs, R, &pnb);
      st.phase = OVP_BARRIER;   gb_stream_run(&s, false, gb_now_ns(), W, o->runs, R, &pbar);
      gb_stream_finish(&s);
      bad = pa.failed || pb.failed || pnb.failed || pbar.failed;
   } else {
      const int depth = GB_CR_DEPTH;
      struct gb_stream sa, sb, sbar;
      gb_stream_init(&sa, &vk, 0, depth, ov_rec_only_a, NULL, &st);
      gb_stream_init(&sb, &vk, 1, depth, ov_rec_only_b, NULL, &st);
      gb_stream_run(&sa, false, gb_now_ns(), W, o->runs, R, &pa);
      gb_stream_run(&sb, false, gb_now_ns(), W, o->runs, R, &pb);

      uint64_t t0 = gb_now_ns() + 20000000ull;
      struct sram_thr th[2] = {
         {.s = &sa, .t0 = t0, .warm = W, .win = R, .nwin = o->runs},
         {.s = &sb, .t0 = t0, .warm = W, .win = R, .nwin = o->runs},
      };
      for (int i = 0; i < 2; i++) pthread_create(&th[i].th, NULL, sram_thr_main, &th[i]);
      for (int i = 0; i < 2; i++) pthread_join(th[i].th, NULL);
      gb_stream_finish(&sa);
      gb_stream_finish(&sb);

      /* Serialized baseline with explicit barrier on queue 0 */
      st.phase = OVP_BARRIER;
      gb_stream_init(&sbar, &vk, 0, depth, ov_record, NULL, &st);
      gb_stream_run(&sbar, false, gb_now_ns(), W, o->runs, R, &pbar);
      gb_stream_finish(&sbar);

      /* For async_cr, combined effective pair time = max(conc_a, conc_b) per pair */
      pnb = th[0].pass;
      struct gb_samples ra_s = {0}, rb_s = {0}, ca_s = {0}, cb_s = {0}, bar_s = {0};
      gb_pass_rate(&pa, &ra_s);
      gb_pass_rate(&pb, &rb_s);
      gb_pass_rate(&th[0].pass, &ca_s);
      gb_pass_rate(&th[1].pass, &cb_s);
      gb_pass_rate(&pbar, &bar_s);
      bad = pa.failed || pb.failed || th[0].pass.failed || th[1].pass.failed || pbar.failed ||
            !ra_s.n || !rb_s.n || !ca_s.n || !cb_s.n || !bar_s.n;

      struct gb_result r;
      gb_result_begin(&r, o, GB_MODE_QUEUES, 2 * depth, NULL);
      gb_result_run_info(&r, o->warmup_ms, o->runs, o->run_ms);
      if (bad) {
         gb_result_finish(&r, "fail", NULL, "GPU wait timed out or no completions");
      } else {
         double ma = gb_median(&ra_s), mb = gb_median(&rb_s);
         double mca = gb_median(&ca_s), mcb = gb_median(&cb_s);
         double mnb = 1.0 / ((1.0 / mca) * (ma / (ma + mb)) + (1.0 / mcb) * (mb / (ma + mb)));
         double ov = ma / mca + mb / mcb;
         double mbar = gb_median(&bar_s);
         gb_result_metric(&r, "solo_a_us", "us", true, &ra_s, true);
         gb_result_metric(&r, "solo_b_us", "us", true, &rb_s, true);
         gb_result_metric(&r, "conc_a_us", "us", true, &ca_s, true);
         gb_result_metric(&r, "conc_b_us", "us", true, &cb_s, true);
         gb_result_metric1(&r, "nobarrier_us", "us", true, mnb);
         gb_result_metric(&r, "barrier_us", "us", true, &bar_s, true);
         gb_result_metric1(&r, "overlap_ratio", "x", false, ov);
         gb_result_metric1(&r, "barrier_penalty_us", "us", true, mbar - mnb);
         gb_result_metric1(&r, "serialization_gap", "x", false, mnb > 0.0 ? mbar / mnb : 0.0);
         gb_result_finish(&r, "ok", "overlap_ratio", NULL);
      }
      gb_samples_free(&ra_s); gb_samples_free(&rb_s); gb_samples_free(&ca_s);
      gb_samples_free(&cb_s); gb_samples_free(&bar_s);
      gb_pass_free(&pa); gb_pass_free(&pb); gb_pass_free(&th[0].pass);
      gb_pass_free(&th[1].pass); gb_pass_free(&pbar);
      goto cleanup;
   }

   struct gb_samples sa_s = {0}, sb_s = {0}, snb_s = {0}, sbar_s = {0};
   gb_pass_rate(&pa, &sa_s);
   gb_pass_rate(&pb, &sb_s);
   gb_pass_rate(&pnb, &snb_s);
   gb_pass_rate(&pbar, &sbar_s);
   bad |= !sa_s.n || !sb_s.n || !snb_s.n || !sbar_s.n;

   struct gb_result r;
   gb_result_begin(&r, o, GB_MODE_QUEUE, 2, NULL);
   gb_result_run_info(&r, o->warmup_ms, o->runs, o->run_ms);
   if (bad) {
      gb_result_finish(&r, "fail", NULL, "GPU wait timed out or no completions");
   } else {
      double ma = gb_median(&sa_s), mb = gb_median(&sb_s);
      double mnb = gb_median(&snb_s), mbar = gb_median(&sbar_s);
      gb_result_metric(&r, "solo_a_us", "us", true, &sa_s, true);
      gb_result_metric(&r, "solo_b_us", "us", true, &sb_s, true);
      gb_result_metric(&r, "nobarrier_us", "us", true, &snb_s, true);
      gb_result_metric(&r, "barrier_us", "us", true, &sbar_s, true);
      gb_result_metric1(&r, "overlap_ratio", "x", false, mnb > 0.0 ? (ma + mb) / mnb : 0.0);
      gb_result_metric1(&r, "barrier_penalty_us", "us", true, mbar - mnb);
      gb_result_metric1(&r, "serialization_gap", "x", false, mnb > 0.0 ? mbar / mnb : 0.0);
      gb_result_finish(&r, "ok", "overlap_ratio", NULL);
   }
   gb_samples_free(&sa_s); gb_samples_free(&sb_s); gb_samples_free(&snb_s); gb_samples_free(&sbar_s);
   gb_pass_free(&pa); gb_pass_free(&pb); gb_pass_free(&pnb); gb_pass_free(&pbar);

cleanup:
   if (st.pc_a.pipe) gb_pipe_destroy(&vk, &st.pc_a);
   if (st.pc_b.pipe) gb_pipe_destroy(&vk, &st.pc_b);
   if (st.pr_a.pipe) gb_pipe_destroy(&vk, &st.pr_a);
   if (st.pr_b.pipe) gb_pipe_destroy(&vk, &st.pr_b);
   if (st.ba.buf) gb_buf_destroy(&vk, &st.ba);
   if (st.bb.buf) gb_buf_destroy(&vk, &st.bb);
   if (st.ia.img) gb_img_destroy(&vk, &st.ia);
   if (st.ib.img) gb_img_destroy(&vk, &st.ib);
   gb_vk_destroy(&vk);
   return 0;
}
