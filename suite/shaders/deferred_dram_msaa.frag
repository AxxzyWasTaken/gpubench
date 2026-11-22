// SPDX-License-Identifier: GPL-2.0-only
#version 450
// render.deferred mode=dram (4x MSAA) lighting pass FS: per-sample DRAM fetch + resolve average.
layout(set = 0, binding = 0) uniform sampler2DMS tex_g0;
layout(set = 0, binding = 1) uniform sampler2DMS tex_g1;
layout(set = 0, binding = 2) uniform sampler2DMS tex_g2;
layout(set = 0, binding = 3) uniform sampler2DMS tex_g3;
layout(location = 0) out vec4 out_color;

const vec3 LDIR[4] = vec3[4](
   vec3( 0.57735027,  0.57735027, 0.57735027),
   vec3(-0.53452248,  0.26726124, 0.80178373),
   vec3( 0.26726124, -0.53452248, 0.80178373),
   vec3(-0.40824829, -0.40824829, 0.81649658)
);
const vec3 LCOL[4] = vec3[4](
   vec3(0.95, 0.90, 0.85),
   vec3(0.40, 0.55, 0.85),
   vec3(0.75, 0.45, 0.30),
   vec3(0.35, 0.70, 0.50)
);

void main()
{
   ivec2 p = ivec2(gl_FragCoord.xy);
   vec3 total = vec3(0.0);
   for (int s = 0; s < 4; s++) {
      vec4 g0 = texelFetch(tex_g0, p, s);
      vec4 g1 = texelFetch(tex_g1, p, s);
      vec4 g2 = texelFetch(tex_g2, p, s);
      float linz = texelFetch(tex_g3, p, s).r;

      vec3 albedo = g0.rgb;
      float rough = g0.a;
      vec3 n = normalize(g1.xyz);
      float metal = g1.w;
      vec3 emissive = g2.rgb;
      float ao = g2.a;

      vec3 col = albedo * 0.12 * ao + emissive;
      float shin = 8.0 + (1.0 - rough) * 24.0;
      float atten = 1.0 / (1.0 + linz * linz);
      for (int i = 0; i < 4; i++) {
         float ndl = max(dot(n, LDIR[i]), 0.0);
         vec3 h = normalize(LDIR[i] + vec3(0.0, 0.0, 1.0));
         float ndh = max(dot(n, h), 0.0);
         float spec = pow(ndh, shin) * (0.2 + 0.8 * metal);
         float diff = ndl * (1.0 - 0.5 * metal);
         col += (albedo * diff + vec3(spec)) * LCOL[i] * atten;
      }
      total += clamp(col, 0.0, 1.0);
   }
   out_color = vec4(total * 0.25, 1.0);
}
