// SPDX-License-Identifier: GPL-2.0-only
#version 450
// render.deferred geometry pass FS: writes 4 MRT G-buffer attachments.
layout(location = 0) flat in uint v_qid;
layout(location = 1) flat in float v_linz;

layout(location = 0) out vec4 g0_albedo_rough;
layout(location = 1) out vec4 g1_normal_metal;
layout(location = 2) out vec4 g2_emissive_ao;
layout(location = 3) out float g3_linz;

void main()
{
   uint qid = v_qid;
   vec3 albedo = vec3(
      0.2 + 0.7 * float((qid * 37u) & 255u) / 255.0,
      0.2 + 0.7 * float((qid * 73u + 51u) & 255u) / 255.0,
      0.2 + 0.7 * float((qid * 131u + 17u) & 255u) / 255.0
   );
   float rough = 0.15 + 0.7 * float((qid * 19u) & 255u) / 255.0;
   vec3 n = normalize(vec3(
      float((qid * 53u) & 255u) / 255.0 * 0.8 - 0.4,
      float((qid * 97u) & 255u) / 255.0 * 0.8 - 0.4,
      1.0
   ));
   float metal = (qid & 3u) == 0u ? 0.8 : 0.1;
   vec3 emissive = vec3(
      0.05 * float((qid * 11u) & 255u) / 255.0,
      0.04 * float((qid * 23u) & 255u) / 255.0,
      0.06 * float((qid * 41u) & 255u) / 255.0
   );
   float ao = 0.6 + 0.4 * float((qid * 29u) & 255u) / 255.0;

   g0_albedo_rough = vec4(albedo, rough);
   g1_normal_metal = vec4(n, metal);
   g2_emissive_ao = vec4(emissive, ao);
   g3_linz = v_linz;
}
