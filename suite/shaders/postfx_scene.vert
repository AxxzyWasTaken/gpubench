// SPDX-License-Identifier: GPL-2.0-only
#version 450
layout(location = 0) out vec4 v_hdr;

void main()
{
   uint q = uint(gl_VertexIndex) / 6u;
   uint v = uint(gl_VertexIndex) % 6u;
   uint gx = q & 63u;
   uint gy = (q >> 6) & 63u;

   const vec2 corners[6] = vec2[6](
      vec2(0.0, 0.0), vec2(1.0, 0.0), vec2(1.0, 1.0),
      vec2(0.0, 0.0), vec2(1.0, 1.0), vec2(0.0, 1.0)
   );
   vec2 uv = (vec2(float(gx), float(gy)) + corners[v]) * (1.0 / 64.0);
   float dx = float(gx) - 31.5;
   float dy = float(gy) - 31.5;
   float d2 = dx * dx + dy * dy;
   if (d2 < 100.0) {
      float boost = 3.5 - d2 * 0.02;
      v_hdr = vec4(boost, boost * 0.65, boost * 0.30, 1.0);
   } else {
      v_hdr = vec4(0.15 + 0.55 * uv.x, 0.15 + 0.55 * uv.y, 0.35, 1.0);
   }
   float z = 0.2 + 0.6 * float((gx ^ gy) & 15u) / 15.0;
   gl_Position = vec4(uv * 2.0 - 1.0, z, 1.0);
}
