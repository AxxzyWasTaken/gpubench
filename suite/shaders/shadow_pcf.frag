// SPDX-License-Identifier: GPL-2.0-only
#version 450
layout(set = 0, binding = 0) uniform sampler2DArrayShadow smap;
layout(push_constant) uniform PC {
   float inv_res;
} pc;
layout(location = 0) out vec4 out_col;

void main()
{
   vec2 uv = gl_FragCoord.xy * (1.0 / 1024.0);
   float qx = uv.x >= 0.5 ? 1.0 : 0.0;
   float qy = uv.y >= 0.5 ? 1.0 : 0.0;
   float layer = qx + 2.0 * qy;
   vec2 luv = fract(uv * 2.0);

   float sum = 0.0;
   for (int dy = -1; dy <= 1; dy++) {
      for (int dx = -1; dx <= 1; dx++) {
         vec2 suv = clamp(luv + vec2(float(dx), float(dy)) * pc.inv_res, 0.0, 1.0);
         sum += texture(smap, vec4(suv, layer, 0.70));
      }
   }
   float sh = sum * (1.0 / 9.0);
   out_col = vec4(0.2 + 0.7 * sh, 0.25 + 0.5 * sh, 0.35 + 0.4 * sh, 1.0);
}
