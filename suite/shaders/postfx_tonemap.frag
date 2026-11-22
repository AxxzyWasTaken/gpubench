// SPDX-License-Identifier: GPL-2.0-only
#version 450
layout(set = 0, binding = 0) uniform sampler2D hdr_tex;
layout(set = 0, binding = 1) uniform sampler2D bloom_tex;
layout(push_constant) uniform PC {
   float inv_w;
   float inv_h;
} pc;
layout(location = 0) out vec4 out_col;

void main()
{
   vec2 uv = gl_FragCoord.xy * vec2(pc.inv_w, pc.inv_h);
   vec3 hdr = texture(hdr_tex, uv).rgb + 0.35 * texture(bloom_tex, uv).rgb;
   vec2 dv = uv - vec2(0.5);
   float vig = clamp(1.0 - dot(dv, dv) * 1.4, 0.25, 1.0);
   hdr *= vig;
   vec3 mapped = clamp((hdr * (2.51 * hdr + 0.03)) / (hdr * (2.43 * hdr + 0.59) + 0.14), 0.0, 1.0);
   out_col = vec4(pow(mapped, vec3(1.0 / 2.2)), 1.0);
}
