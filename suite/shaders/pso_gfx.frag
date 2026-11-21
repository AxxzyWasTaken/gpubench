// SPDX-License-Identifier: GPL-2.0-only
#version 450
// compile.pso kind=graphics FS: PBR-style multi-light fragment shader with specialization constants.
layout(constant_id = 0) const uint VAR_ID = 0u;
layout(constant_id = 1) const uint LOOP_N = 6u;
layout(constant_id = 2) const uint MASK = 3u;
layout(constant_id = 3) const uint NONCE = 1u;
layout(location = 0) in vec4 v_uv;
layout(location = 0) out vec4 out_color;

void main()
{
   vec3 n = normalize(vec3(v_uv.xy * 2.0 - 1.0, 0.8 + 0.1 * v_uv.z));
   vec3 albedo = vec3(v_uv.x, v_uv.y, v_uv.z);
   vec3 col = albedo * 0.1;
   float phase = float((NONCE ^ (VAR_ID * 17u)) & 1023u) * 0.006;
   for (uint i = 0u; i < LOOP_N; i++) {
      float a = phase + float(i) * 0.7;
      vec3 l = normalize(vec3(sin(a), cos(a), 0.7));
      float ndl = max(dot(n, l), 0.0);
      vec3 h = normalize(l + vec3(0.0, 0.0, 1.0));
      float spec = pow(max(dot(n, h), 0.0), 8.0 + float((i + VAR_ID) & MASK) * 6.0);
      col += (albedo * ndl + vec3(spec * 0.3)) * (0.25 / float(LOOP_N));
   }
   out_color = vec4(clamp(col, 0.0, 1.0), 1.0);
}
