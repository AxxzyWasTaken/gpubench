// SPDX-License-Identifier: GPL-2.0-only
#version 450
// compress.bw kind=mrt4 Pass 1: smooth compressible tiled gradients across 4 MRTs.
layout(location = 0) out vec4 g0;
layout(location = 1) out vec4 g1;
layout(location = 2) out vec4 g2;
layout(location = 3) out float g3;
layout(push_constant) uniform PC { vec4 param; } pc;

void main()
{
   vec2 p = floor(gl_FragCoord.xy * 0.0625) * 0.015625;
   float k = pc.param.x;
   g0 = vec4(fract(p.x + k), fract(p.y + 0.25), fract(p.x + p.y), 1.0);
   g1 = vec4(fract(p.y + k), fract(p.x + 0.50), 0.75, 0.5);
   g2 = vec4(0.25, fract(p.x * 0.5 + k), fract(p.y * 0.5), 1.0);
   g3 = 0.25 + 0.5 * fract(p.x + k);
}
