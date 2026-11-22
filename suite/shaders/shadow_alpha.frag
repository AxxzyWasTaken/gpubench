// SPDX-License-Identifier: GPL-2.0-only
#version 450
layout(location = 0) in vec2 v_uv;

void main()
{
   float m = sin(v_uv.x * 18.8495559) * sin(v_uv.y * 18.8495559);
   if (m < 0.0)
      discard;
}
