// SPDX-License-Identifier: GPL-2.0-only
#version 450
// render.mdi VS: each indirect draw renders a 10x10 pixel quad on a 1024x1024 target.
// Uses gl_InstanceIndex (= firstInstance = draw_id) so both Vulkan MDI and Metal ICB
// pass the exact draw index identically without depending on shaderDrawParameters quirks.
layout(location = 0) in vec2 in_pos;
layout(location = 0) flat out uint v_tex_idx;
layout(location = 1) out vec2 v_uv;
layout(push_constant) uniform PC { uint bindless; } pc;

void main()
{
   uint draw_id = uint(gl_InstanceIndex);
   uint cell = draw_id % 10000u;
   uint cx = cell % 100u;
   uint cy = cell / 100u;
   vec2 px = vec2(float(cx * 10u), float(cy * 10u)) + in_pos * 10.0;
   vec2 ndc = (px / 1024.0) * 2.0 - 1.0;
   gl_Position = vec4(ndc, 0.0, 1.0);
   v_tex_idx = draw_id % pc.bindless;
   v_uv = in_pos;
}
