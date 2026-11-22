// SPDX-License-Identifier: GPL-2.0-only
#version 450
layout(location = 0) in vec4 v_hdr;
layout(location = 0) out vec4 out_hdr;

void main()
{
   out_hdr = v_hdr;
}
