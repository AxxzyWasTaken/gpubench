// SPDX-License-Identifier: GPL-2.0-only
#version 450
// render.deferred geometry pass VS: 4096 quads = 4 depth layers x 32x32 grid.
// Drawn back-to-front (layer 3 -> 0) so depth test LESS lets layer 0 win at z=0.25.
layout(location = 0) flat out uint v_qid;
layout(location = 1) flat out float v_linz;

void main()
{
   uint quad = uint(gl_VertexIndex) / 6u;
   uint corner = uint(gl_VertexIndex) - quad * 6u;
   /* Back-to-front layer order: draw 0..1023 -> layer 3, ..., 3072..4095 -> layer 0 */
   uint layer = 3u - (quad >> 10u);
   uint cell = quad & 1023u;
   uint qx = cell & 31u;
   uint qy = cell >> 5u;

   vec2 off;
   if (corner == 0u) off = vec2(0.0, 0.0);
   else if (corner == 1u) off = vec2(1.0, 0.0);
   else if (corner == 2u) off = vec2(0.0, 1.0);
   else if (corner == 3u) off = vec2(1.0, 0.0);
   else if (corner == 4u) off = vec2(1.0, 1.0);
   else off = vec2(0.0, 1.0);

   vec2 uv = (vec2(float(qx), float(qy)) + off) / 32.0;
   float z = 0.25 + float(layer) * 0.20;
   gl_Position = vec4(uv * 2.0 - 1.0, z, 1.0);
   v_qid = cell + layer * 1024u;
   v_linz = z;
}
