// SPDX-License-Identifier: GPL-2.0-only
#version 450
layout(set = 0, binding = 0) uniform sampler2D src_tex;
layout(push_constant) uniform PC {
   float inv_w;
   float inv_h;
   float src_inv_w;
   float src_inv_h;
   uint bright_pass;
} pc;
layout(location = 0) out vec4 out_col;

void main()
{
   vec2 uv = gl_FragCoord.xy * vec2(pc.inv_w, pc.inv_h);
   vec2 d = vec2(pc.src_inv_w, pc.src_inv_h);
   vec3 c = texture(src_tex, uv).rgb * 0.5
          + texture(src_tex, uv + vec2(-d.x, -d.y)).rgb * 0.125
          + texture(src_tex, uv + vec2( d.x, -d.y)).rgb * 0.125
          + texture(src_tex, uv + vec2(-d.x,  d.y)).rgb * 0.125
          + texture(src_tex, uv + vec2( d.x,  d.y)).rgb * 0.125;
   if (pc.bright_pass != 0u)
      c = max(c - vec3(1.0), vec3(0.0));
   out_col = vec4(c, 1.0);
}
