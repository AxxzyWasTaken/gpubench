// SPDX-License-Identifier: GPL-2.0-only
#version 450
layout(push_constant) uniform PC {
   uint cascade;
   uint res;
} pc;
layout(location = 0) out vec2 v_uv;

void main()
{
   uint i = uint(gl_InstanceIndex);
   uint cell = i & 1023u;
   uint gx = cell & 31u;
   uint gy = cell >> 5;

   const vec2 corners[6] = vec2[6](
      vec2(0.0, 0.0), vec2(1.0, 0.0), vec2(1.0, 1.0),
      vec2(0.0, 0.0), vec2(1.0, 1.0), vec2(0.0, 1.0)
   );
   vec2 corner = corners[gl_VertexIndex % 6];
   vec2 uv = (vec2(float(gx), float(gy)) + corner) * (1.0 / 32.0);
   bool caster = (((gx + gy + pc.cascade) & 1u) == 0u);
   float z = caster ? ((i < 1024u) ? (0.30 + 0.05 * float(pc.cascade))
                                   : (0.60 + 0.05 * float(pc.cascade)))
                    : 0.85;
   gl_Position = vec4(uv * 2.0 - 1.0, z, 1.0);
   v_uv = corner;
}
