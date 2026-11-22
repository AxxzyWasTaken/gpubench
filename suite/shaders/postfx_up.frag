// SPDX-License-Identifier: GPL-2.0-only
#version 450
layout(set = 0, binding = 0) uniform sampler2D src_tex;
layout(push_constant) uniform PC {
   float inv_w;
   float inv_h;
   float src_inv_w;
   float src_inv_h;
} pc;
layout(location = 0) out vec4 out_col;

void main()
{
   vec2 uv = gl_FragCoord.xy * vec2(pc.inv_w, pc.inv_h);
   vec2 d = vec2(pc.src_inv_w, pc.src_inv_h);
   vec3 c = (texture(src_tex, uv + vec2(-d.x, -d.y)).rgb +
             texture(src_tex, uv + vec2( d.x, -d.y)).rgb +
             texture(src_tex, uv + vec2(-d.x,  d.y)).rgb +
             texture(src_tex, uv + vec2( d.x,  d.y)).rgb) * 0.25;
   out_col = vec4(c * 0.5, 1.0);
}
