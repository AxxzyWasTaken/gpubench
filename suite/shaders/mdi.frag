// SPDX-License-Identifier: GPL-2.0-only
#version 450
#extension GL_EXT_nonuniform_qualifier : require
// render.mdi FS: samples a bindless texture array via nonuniformEXT(v_tex_idx).
layout(location = 0) flat in uint v_tex_idx;
layout(location = 1) in vec2 v_uv;
layout(set = 0, binding = 0) uniform sampler2D textures[];
layout(location = 0) out vec4 out_color;

void main()
{
   out_color = texture(textures[nonuniformEXT(v_tex_idx)], v_uv);
}
