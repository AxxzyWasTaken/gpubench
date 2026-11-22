// SPDX-License-Identifier: GPL-2.0-only
#version 450
layout(set = 0, binding = 0, std140) uniform Bank {
   vec4 lut[1024];
} banks[16];
layout(location = 0) flat in uint v_draw;
layout(location = 0) out vec4 out_col;

void main()
{
   uint x = uint(gl_FragCoord.x);
   uint y = uint(gl_FragCoord.y);
   uint b = v_draw & 15u;
   uint i0 = (v_draw * 13u + ((x & 15u) ^ (y & 15u))) & 1023u;
   vec4 c0 = banks[b].lut[i0];
   uint i1 = (i0 + uint(c0.r * 255.0 + 0.5)) & 1023u;
   vec4 c1 = banks[b].lut[i1];
   uint i2 = (i1 + uint(c1.g * 255.0 + 0.5)) & 1023u;
   vec4 c2 = banks[b].lut[i2];
   uint i3 = (i2 + uint(c2.b * 255.0 + 0.5)) & 1023u;
   out_col = banks[b].lut[i3];
}
