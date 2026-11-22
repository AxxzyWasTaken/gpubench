// SPDX-License-Identifier: GPL-2.0-only
// Wave-2 heavy-game / AAA Metal shaders: exact algorithmic twins of the
// Wave-2 GLSL shaders (shadow_*, cull_mdi.*, bubo.*, fp16_*, postfx_*).
#include <metal_stdlib>
using namespace metal;

/* ======================= 1. render.shadow ======================= */

struct ShadowDepthPC { uint cascade; uint res; };
struct ShadowPcfPC { float inv_res; };
struct ShadowVOut { float4 pos [[position]]; float2 uv; };

static constant float2 kW2Quad6[6] = {
   float2(0.0f, 0.0f), float2(1.0f, 0.0f), float2(1.0f, 1.0f),
   float2(0.0f, 0.0f), float2(1.0f, 1.0f), float2(0.0f, 1.0f)
};

vertex ShadowVOut shadow_depth_vert(uint vid [[vertex_id]], uint iid [[instance_id]],
                                    constant ShadowDepthPC &pc [[buffer(0)]])
{
   uint cell = iid & 1023u;
   uint gx = cell & 31u;
   uint gy = cell >> 5;
   float2 corner = kW2Quad6[vid % 6u];
   float2 uv = (float2(float(gx), float(gy)) + corner) * (1.0f / 32.0f);
   bool caster = (((gx + gy + pc.cascade) & 1u) == 0u);
   float z = caster ? ((iid < 1024u) ? (0.30f + 0.05f * float(pc.cascade))
                                     : (0.60f + 0.05f * float(pc.cascade)))
                    : 0.85f;
   ShadowVOut o;
   o.pos = float4(uv.x * 2.0f - 1.0f, 1.0f - uv.y * 2.0f, z, 1.0f);
   o.uv = corner;
   return o;
}

fragment void shadow_alpha_frag(ShadowVOut in [[stage_in]])
{
   float m = sin(in.uv.x * 18.8495559f) * sin(in.uv.y * 18.8495559f);
   if (m < 0.0f)
      discard_fragment();
}

fragment float4 shadow_pcf_frag(float4 fc [[position]],
                                depth2d_array<float> smap [[texture(0)]],
                                sampler smp [[sampler(0)]],
                                constant ShadowPcfPC &pc [[buffer(0)]])
{
   float2 uv = fc.xy * (1.0f / 1024.0f);
   uint qx = uv.x >= 0.5f ? 1u : 0u;
   uint qy = uv.y >= 0.5f ? 1u : 0u;
   uint layer = qx + 2u * qy;
   float2 luv = fract(uv * 2.0f);

   float sum = 0.0f;
   for (int dy = -1; dy <= 1; dy++) {
      for (int dx = -1; dx <= 1; dx++) {
         float2 suv = clamp(luv + float2(float(dx), float(dy)) * pc.inv_res, 0.0f, 1.0f);
         sum += smap.sample_compare(smp, suv, layer, 0.70f);
      }
   }
   float sh = sum * (1.0f / 9.0f);
   return float4(0.2f + 0.7f * sh, 0.25f + 0.5f * sh, 0.35f + 0.4f * sh, 1.0f);
}

/* ======================= 2. compute.cull_mdi ======================= */

struct CullObj { float cx, cy, r; uint color; };
struct CullPC { uint n_objs; };
struct CullICBArg { command_buffer icb [[id(0)]]; };
struct CullRange { uint location; atomic_uint length; };

kernel void cull_mdi_comp(device const CullObj *objs [[buffer(0)]],
                          device CullICBArg *icb_arg [[buffer(1)]],
                          device CullRange *range [[buffer(2)]],
                          device const ushort *ib [[buffer(3)]],
                          constant CullPC &pc [[buffer(4)]],
                          uint id [[thread_position_in_grid]],
                          uint simd_lane [[thread_index_in_simdgroup]])
{
   bool vis = false;
   if (id < pc.n_objs) {
      CullObj o = objs[id];
      vis = (o.cx + o.r >= -1.0f) && (-o.cx + o.r >= -1.0f) &&
            (o.cy + o.r >= -1.0f) && (-o.cy + o.r >= -1.0f) &&
            (o.cx * o.cx + o.cy * o.cy < 2.0f);
   }
   uint v = vis ? 1u : 0u;
   uint cnt = simd_sum(v);
   uint off = simd_prefix_exclusive_sum(v);
   uint base = 0u;
   if (simd_lane == 0u && cnt > 0u)
      base = atomic_fetch_add_explicit(&range->length, cnt, memory_order_relaxed);
   base = simd_broadcast_first(base);
   if (vis) {
      render_command cmd(icb_arg->icb, base + off);
      cmd.draw_indexed_primitives(primitive_type::triangle, 6u, ib, 1u, 0, id);
   }
}

struct CullVOut { float4 pos [[position]]; uint col [[flat]]; };

vertex CullVOut cull_mdi_vert(uint vid [[vertex_id]], uint iid [[instance_id]],
                              device const CullObj *objs [[buffer(0)]])
{
   uint v = vid & 3u;
   uint gx = iid & 127u;
   uint gy = (iid >> 7) & 127u;
   float2 off = float2((v == 1u || v == 2u) ? 8.0f : 0.0f, v >= 2u ? 8.0f : 0.0f);
   float2 px = float2(float(gx * 8u), float(gy * 8u)) + off;
   float2 uv = px * (1.0f / 1024.0f);
   CullVOut o;
   o.pos = float4(uv.x * 2.0f - 1.0f, 1.0f - uv.y * 2.0f, 0.0f, 1.0f);
   o.col = objs[iid].color;
   return o;
}

fragment float4 cull_mdi_frag(CullVOut in [[stage_in]])
{
   return unpack_unorm4x8_to_float(in.col);
}

/* ======================= 3. render.bindless_ubo ======================= */

struct BuboPC { uint draw_id; };
struct BuboVOut { float4 pos [[position]]; uint draw_id [[flat]]; };
struct BuboBank { float4 lut[1024]; };
struct BuboBanksArg { constant BuboBank *banks[16] [[id(0)]]; };

static constant float2 kBuboQuad[6] = {
   float2(0.0f, 0.0f), float2(16.0f, 0.0f), float2(16.0f, 32.0f),
   float2(0.0f, 0.0f), float2(16.0f, 32.0f), float2(0.0f, 32.0f)
};

vertex BuboVOut bubo_vert(uint vid [[vertex_id]], constant BuboPC &pc [[buffer(0)]])
{
   uint tx = pc.draw_id & 63u;
   uint ty = (pc.draw_id >> 6) & 31u;
   float2 px = float2(float(tx * 16u), float(ty * 32u)) + kBuboQuad[vid % 6u];
   float2 uv = px * (1.0f / 1024.0f);
   BuboVOut o;
   o.pos = float4(uv.x * 2.0f - 1.0f, 1.0f - uv.y * 2.0f, 0.0f, 1.0f);
   o.draw_id = pc.draw_id;
   return o;
}

fragment float4 bubo_frag(BuboVOut in [[stage_in]], constant BuboBanksArg &arg [[buffer(0)]])
{
   uint x = uint(in.pos.x);
   uint y = uint(in.pos.y);
   uint b = in.draw_id & 15u;
   constant float4 *lut = arg.banks[b]->lut;
   uint i0 = (in.draw_id * 13u + ((x & 15u) ^ (y & 15u))) & 1023u;
   float4 c0 = lut[i0];
   uint i1 = (i0 + uint(c0.r * 255.0f + 0.5f)) & 1023u;
   float4 c1 = lut[i1];
   uint i2 = (i1 + uint(c1.g * 255.0f + 0.5f)) & 1023u;
   float4 c2 = lut[i2];
   uint i3 = (i2 + uint(c2.b * 255.0f + 0.5f)) & 1023u;
   return lut[i3];
}

/* ======================= 4. compute.fp16 ======================= */

kernel void fp16_gemv_main(device const uint *w_pk [[buffer(0)]],
                           device const uint *x_pk [[buffer(1)]],
                           device float *y_out [[buffer(2)]],
                           uint lid [[thread_index_in_threadgroup]],
                           uint wgid [[threadgroup_position_in_grid]])
{
   threadgroup float s_sum[64];
   uint r_local = lid >> 4;
   uint lane = lid & 15u;
   uint row = wgid * 4u + r_local;
   uint row_base = row * 1024u;

   float acc = 0.0f;
   for (uint t = lane; t < 1024u; t += 16u) {
      uint pk = w_pk[row_base + t];
      half4 wv = half4(half(pk & 255u), half((pk >> 8) & 255u),
                       half((pk >> 16) & 255u), half(pk >> 24));
      half2 x01 = as_type<half2>(x_pk[t * 2u]);
      half2 x23 = as_type<half2>(x_pk[t * 2u + 1u]);
      half4 xv = half4(x01.x, x01.y, x23.x, x23.y);
      half4 p = wv * xv;
      acc += float(p.x) + float(p.y) + float(p.z) + float(p.w);
   }
   s_sum[lid] = acc;
   threadgroup_barrier(mem_flags::mem_threadgroup);
   if (lane < 8u) s_sum[lid] += s_sum[lid + 8u];
   threadgroup_barrier(mem_flags::mem_threadgroup);
   if (lane < 4u) s_sum[lid] += s_sum[lid + 4u];
   threadgroup_barrier(mem_flags::mem_threadgroup);
   if (lane < 2u) s_sum[lid] += s_sum[lid + 2u];
   threadgroup_barrier(mem_flags::mem_threadgroup);
   if (lane == 0u)
      y_out[row] = s_sum[lid] + s_sum[lid + 1u];
}

struct Fp16MixPC { uint iters; float a; float b; };

kernel void fp16_alumix_main(device float *out_v [[buffer(0)]],
                             constant Fp16MixPC &pc [[buffer(1)]],
                             uint id [[thread_position_in_grid]])
{
   half ha = half(pc.a);
   half hb = half(pc.b);
   float fa = pc.a;
   float fb = pc.b;

   half4 base0 = half4(half((id + 0u) & 7u), half((id + 1u) & 7u),
                       half((id + 2u) & 7u), half((id + 3u) & 7u)) * hb;
   half4 base1 = half4(half((id + 2u) & 7u), half((id + 3u) & 7u),
                       half((id + 4u) & 7u), half((id + 5u) & 7u)) * hb;
   half4 h0 = base0, h1 = base1, h2 = -base0, h3 = -base1;
   float4 f0 = float4(base0) + float4(fb);
   float4 f1 = float4(base1) - float4(fb);

   half hsgn = (id & 1u) ? hb : -hb;
   float fsgn = (id & 1u) ? fb : -fb;

   half4 c0 = base0;
   half4 c1 = half4(half((id + 3u) & 7u), half((id + 4u) & 7u),
                    half((id + 5u) & 7u), half((id + 6u) & 7u)) * hb;
   half4 c2 = half4(half((id + 6u) & 7u), half((id + 7u) & 7u),
                    half((id + 8u) & 7u), half((id + 9u) & 7u)) * hb;
   half4 c3 = half4(half((id + 9u) & 7u), half((id + 10u) & 7u),
                    half((id + 11u) & 7u), half((id + 12u) & 7u)) * hb;

   half4 d0 = c0 + half4(hsgn), d1 = c1 - half4(hsgn);
   half4 d2 = c2 + half4(hsgn), d3 = c3 - half4(hsgn);
   float4 cf0 = float4(c0), cf1 = float4(c1);
   float4 df0 = cf0 + float4(fsgn), df1 = cf1 - float4(fsgn);

   for (uint i = 0u; i < pc.iters; i++) {
      h0 = fma(h0, half4(ha), c0); h1 = fma(h1, half4(ha), c1);
      h2 = fma(h2, half4(ha), c2); h3 = fma(h3, half4(ha), c3);
      h0 = fma(h0, half4(ha), d0); h1 = fma(h1, half4(ha), d1);
      h2 = fma(h2, half4(ha), d2); h3 = fma(h3, half4(ha), d3);
      f0 = fma(f0, float4(fa), cf0); f1 = fma(f1, float4(fa), cf1);
      f0 = fma(f0, float4(fa), df0); f1 = fma(f1, float4(fa), df1);
   }
   float4 tot = float4(h0) + float4(h1) + float4(h2) + float4(h3) + f0 + f1;
   out_v[id] = tot.x + tot.y + tot.z + tot.w;
}

/* ======================= 6. render.postfx ======================= */

struct PostVOut { float4 pos [[position]]; float4 hdr; };
struct PostDownPC { float inv_w, inv_h, src_inv_w, src_inv_h; uint bright_pass; };
struct PostUpPC { float inv_w, inv_h, src_inv_w, src_inv_h; };
struct PostTonePC { float inv_w, inv_h; };

vertex PostVOut postfx_scene_vert(uint vid [[vertex_id]])
{
   uint q = vid / 6u;
   uint v = vid % 6u;
   uint gx = q & 63u;
   uint gy = (q >> 6) & 63u;
   float2 uv = (float2(float(gx), float(gy)) + kW2Quad6[v]) * (1.0f / 64.0f);
   float dx = float(gx) - 31.5f;
   float dy = float(gy) - 31.5f;
   float d2 = dx * dx + dy * dy;
   PostVOut o;
   if (d2 < 100.0f) {
      float boost = 3.5f - d2 * 0.02f;
      o.hdr = float4(boost, boost * 0.65f, boost * 0.30f, 1.0f);
   } else {
      o.hdr = float4(0.15f + 0.55f * uv.x, 0.15f + 0.55f * uv.y, 0.35f, 1.0f);
   }
   float z = 0.2f + 0.6f * float((gx ^ gy) & 15u) / 15.0f;
   o.pos = float4(uv.x * 2.0f - 1.0f, 1.0f - uv.y * 2.0f, z, 1.0f);
   return o;
}

fragment float4 postfx_scene_frag(PostVOut in [[stage_in]])
{
   return in.hdr;
}

fragment float4 postfx_down_frag(float4 fc [[position]],
                                 texture2d<float> src_tex [[texture(0)]],
                                 sampler smp [[sampler(0)]],
                                 constant PostDownPC &pc [[buffer(0)]])
{
   float2 uv = fc.xy * float2(pc.inv_w, pc.inv_h);
   float2 d = float2(pc.src_inv_w, pc.src_inv_h);
   float3 c = src_tex.sample(smp, uv).rgb * 0.5f
            + src_tex.sample(smp, uv + float2(-d.x, -d.y)).rgb * 0.125f
            + src_tex.sample(smp, uv + float2( d.x, -d.y)).rgb * 0.125f
            + src_tex.sample(smp, uv + float2(-d.x,  d.y)).rgb * 0.125f
            + src_tex.sample(smp, uv + float2( d.x,  d.y)).rgb * 0.125f;
   if (pc.bright_pass != 0u)
      c = max(c - float3(1.0f), float3(0.0f));
   return float4(c, 1.0f);
}

fragment float4 postfx_up_frag(float4 fc [[position]],
                               texture2d<float> src_tex [[texture(0)]],
                               sampler smp [[sampler(0)]],
                               constant PostUpPC &pc [[buffer(0)]])
{
   float2 uv = fc.xy * float2(pc.inv_w, pc.inv_h);
   float2 d = float2(pc.src_inv_w, pc.src_inv_h);
   float3 c = (src_tex.sample(smp, uv + float2(-d.x, -d.y)).rgb +
               src_tex.sample(smp, uv + float2( d.x, -d.y)).rgb +
               src_tex.sample(smp, uv + float2(-d.x,  d.y)).rgb +
               src_tex.sample(smp, uv + float2( d.x,  d.y)).rgb) * 0.25f;
   return float4(c * 0.5f, 1.0f);
}

fragment float4 postfx_tonemap_frag(float4 fc [[position]],
                                    texture2d<float> hdr_tex [[texture(0)]],
                                    texture2d<float> bloom_tex [[texture(1)]],
                                    sampler smp [[sampler(0)]],
                                    constant PostTonePC &pc [[buffer(0)]])
{
   float2 uv = fc.xy * float2(pc.inv_w, pc.inv_h);
   float3 hdr = hdr_tex.sample(smp, uv).rgb + 0.35f * bloom_tex.sample(smp, uv).rgb;
   float2 dv = uv - float2(0.5f);
   float vig = clamp(1.0f - dot(dv, dv) * 1.4f, 0.25f, 1.0f);
   hdr *= vig;
   float3 mapped = clamp((hdr * (2.51f * hdr + 0.03f)) / (hdr * (2.43f * hdr + 0.59f) + 0.14f), 0.0f, 1.0f);
   return float4(pow(mapped, float3(1.0f / 2.2f)), 1.0f);
}
