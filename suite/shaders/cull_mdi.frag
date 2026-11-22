// SPDX-License-Identifier: GPL-2.0-only
#version 450
layout(location = 0) flat in uint v_col;
layout(location = 0) out vec4 out_col;

void main()
{
   out_col = unpackUnorm4x8(v_col);
}
