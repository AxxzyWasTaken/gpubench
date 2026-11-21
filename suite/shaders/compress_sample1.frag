// SPDX-License-Identifier: GPL-2.0-only
#version 450
// compress.bw kind=rgba32f / msaa4 / storage Pass 2: sample single 2D attachment.
layout(set = 0, binding = 0) uniform sampler2D t0;
layout(location = 0) out vec4 out_color;

void main()
{
   ivec2 p = ivec2(gl_FragCoord.xy);
   vec4 v = texelFetch(t0, p, 0);
   out_color = vec4(clamp(v.rgb, 0.0, 1.0), 1.0);
}
