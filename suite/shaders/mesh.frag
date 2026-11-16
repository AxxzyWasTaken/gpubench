// SPDX-License-Identifier: GPL-2.0-only
#version 460
// render.mesh fragment shader. -DPERPRIM=0|1, -DHEAVY=0|1 (shape=vert: reads
// the four extra varyings).
#if PERPRIM
#extension GL_EXT_mesh_shader : require
#endif
layout(location = 0) flat in vec4 v_color;
#if HEAVY
layout(location = 1) in vec4 v_e0;
layout(location = 2) in vec4 v_e1;
layout(location = 3) in vec4 v_e2;
layout(location = 4) in vec4 v_e3;
#endif
#if PERPRIM
layout(location = 5) perprimitiveEXT flat in vec4 p_color;
layout(location = 6) perprimitiveEXT flat in uint p_id;
#endif
layout(location = 0) out vec4 o;

void main()
{
   vec4 c = v_color;
#if HEAVY
   c += (v_e0 + v_e1 + v_e2 + v_e3) * 0.0625;
#endif
#if PERPRIM
   // p_id < 2^31 always; the select keeps the id live.
   c = p_id < 0x80000000u ? mix(c, p_color, 0.5) : vec4(0.0);
#endif
   o = c;
}
