// SPDX-License-Identifier: GPL-2.0-only
#version 450
layout(push_constant) uniform PC { uint draw_id; } pc;
layout(location = 0) flat out uint v_draw;

void main()
{
   const vec2 corners[6] = vec2[6](
      vec2(0.0, 0.0), vec2(16.0, 0.0), vec2(16.0, 32.0),
      vec2(0.0, 0.0), vec2(16.0, 32.0), vec2(0.0, 32.0)
   );
   uint tx = pc.draw_id & 63u;
   uint ty = (pc.draw_id >> 6) & 31u;
   vec2 px = vec2(float(tx * 16u), float(ty * 32u)) + corners[gl_VertexIndex % 6];
   gl_Position = vec4((px * (1.0 / 1024.0)) * 2.0 - 1.0, 0.0, 1.0);
   v_draw = pc.draw_id;
}
