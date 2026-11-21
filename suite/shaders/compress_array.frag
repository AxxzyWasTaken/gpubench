// SPDX-License-Identifier: GPL-2.0-only
#version 450
// compress.bw kind=array Pass 2: sample all 6 array/cube layers and combine.
layout(set = 0, binding = 0) uniform sampler2DArray t_arr;
layout(location = 0) out vec4 out_color;

void main()
{
   ivec2 p = ivec2(gl_FragCoord.xy);
   vec3 acc = vec3(0.0);
   for (int l = 0; l < 6; l++)
      acc += texelFetch(t_arr, ivec3(p, l), 0).rgb;
   out_color = vec4(clamp(acc * (1.0 / 6.0), 0.0, 1.0), 1.0);
}
