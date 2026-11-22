// SPDX-License-Identifier: GPL-2.0-only
#version 450
// compress.bw kind=mrt4 Pass 2: sample all 4 MRT attachments back from memory.
layout(set = 0, binding = 0) uniform sampler2D t0;
layout(set = 0, binding = 1) uniform sampler2D t1;
layout(set = 0, binding = 2) uniform sampler2D t2;
layout(set = 0, binding = 3) uniform sampler2D t3;
layout(location = 0) out vec4 out_color;

void main()
{
   ivec2 p = ivec2(gl_FragCoord.xy);
   vec4 a = texelFetch(t0, p, 0);
   vec4 b = texelFetch(t1, p, 0);
   vec4 c = texelFetch(t2, p, 0);
   float d = texelFetch(t3, p, 0).r;
   out_color = vec4(clamp((a.rgb + b.rgb + c.rgb) * 0.25 + vec3(d * 0.25), 0.0, 1.0), 1.0);
}
