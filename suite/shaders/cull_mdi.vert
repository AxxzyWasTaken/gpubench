// SPDX-License-Identifier: GPL-2.0-only
#version 450

struct Obj {
   float cx, cy, r;
   uint color;
};

layout(std430, set = 0, binding = 0) readonly buffer Objs { Obj objs[]; };
layout(location = 0) flat out uint v_col;

void main()
{
   uint id = uint(gl_InstanceIndex);
   uint vid = uint(gl_VertexIndex) & 3u;
   uint gx = id & 127u;
   uint gy = (id >> 7) & 127u;
   vec2 off = vec2((vid == 1u || vid == 2u) ? 8.0 : 0.0, vid >= 2u ? 8.0 : 0.0);
   vec2 px = vec2(float(gx * 8u), float(gy * 8u)) + off;
   gl_Position = vec4((px * (1.0 / 1024.0)) * 2.0 - 1.0, 0.0, 1.0);
   v_col = objs[id].color;
}
