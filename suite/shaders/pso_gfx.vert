// SPDX-License-Identifier: GPL-2.0-only
#version 450
// compile.pso kind=graphics VS
layout(constant_id = 0) const uint VAR_ID = 0u;
layout(constant_id = 3) const uint NONCE = 1u;
layout(location = 0) out vec4 v_uv;

void main()
{
   vec2 p = vec2(float((gl_VertexIndex << 1) & 2), float(gl_VertexIndex & 2));
   gl_Position = vec4(p * 2.0 - 1.0, 0.5, 1.0);
   float s = float((VAR_ID ^ NONCE) & 255u) / 255.0;
   v_uv = vec4(p, s, 1.0 - s);
}
