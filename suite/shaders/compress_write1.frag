// SPDX-License-Identifier: GPL-2.0-only
#version 450
// compress.bw kind=rgba32f / msaa4 / array Pass 1: smooth compressible color output.
layout(location = 0) out vec4 out_color;
layout(push_constant) uniform PC { vec4 param; } pc;

void main()
{
   vec2 p = floor(gl_FragCoord.xy * 0.0625) * 0.015625;
   float k = pc.param.x;
   out_color = vec4(fract(p.x + k), fract(p.y + pc.param.y), fract(p.x + p.y + pc.param.z), 1.0);
}
