# SPDX-License-Identifier: GPL-2.0-only
# Shared libraries for every gpubench Vulkan executor.
add_library(gb_common STATIC src/gb_common.c)
target_include_directories(gb_common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_options(gb_common PRIVATE -Wall -Wextra -Wno-missing-field-initializers)
target_link_libraries(gb_common PUBLIC m pthread)

add_library(gb_vk STATIC src/gb_vk.c)
target_include_directories(gb_vk PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src ${GB_VK_INC})
target_compile_options(gb_vk PRIVATE -Wall -Wextra -Wno-missing-field-initializers)
target_link_libraries(gb_vk PUBLIC gb_common ${GB_VULKAN_LIB})

# SPIR-V for the suite's shaders, embedded into the executor.
set(GB_SHADER_DIR ${CMAKE_SOURCE_DIR}/suite/shaders)
set(GB_SHADERS light.comp alu.comp bw.comp fs.vert color.frag texture.frag geom.vert tvb.vert)
set(spv_files)
foreach(s IN LISTS GB_SHADERS)
  set(out ${CMAKE_CURRENT_BINARY_DIR}/spv/${s}.spv)
  add_custom_command(OUTPUT ${out}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/spv
    COMMAND ${GLSLANG} -V --target-env vulkan1.3 -o ${out} ${GB_SHADER_DIR}/${s}
    DEPENDS ${GB_SHADER_DIR}/${s} VERBATIM)
  list(APPEND spv_files ${out})
endforeach()
add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/shaders.c
  COMMAND ${CMAKE_COMMAND} "-DSPV_FILES=${spv_files}" -DOUT=${CMAKE_CURRENT_BINARY_DIR}/shaders.c
          -P ${GB_SHADER_DIR}/embed.cmake
  DEPENDS ${spv_files} ${GB_SHADER_DIR}/embed.cmake VERBATIM)

# render.mesh: one SPIR-V module per variant (the declared output counts and
# varyings are compile-time, as in the MSL mesh<> type), named
# mesh_<prim|vert|line>_w<wg>_p<perprim>_v<views>_<m|t>_mesh, the task shader
# mesh_task_task and mesh_frag_p<perprim>_h<heavy>_frag.
set(mesh_spv)
foreach(kind prim vert line)
  if(kind STREQUAL "line")
    set(kd -DTOPO_LINE=1 -DSHAPE_VERT=0)
  elseif(kind STREQUAL "vert")
    set(kd -DTOPO_LINE=0 -DSHAPE_VERT=1)
  else()
    set(kd -DTOPO_LINE=0 -DSHAPE_VERT=0)
  endif()
  foreach(wg 32 128)
    foreach(pp 0 1)
      foreach(v 1 2)
        foreach(st m t)
          if(st STREQUAL "t")
            set(td 1)
          else()
            set(td 0)
          endif()
          set(out ${CMAKE_CURRENT_BINARY_DIR}/spv/mesh_${kind}_w${wg}_p${pp}_v${v}_${st}.mesh.spv)
          add_custom_command(OUTPUT ${out}
            COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/spv
            COMMAND ${GLSLANG} -V --target-env vulkan1.3 ${kd} -DWG=${wg} -DPERPRIM=${pp} -DVIEWS=${v}
                    -DTASK=${td} -o ${out} ${GB_SHADER_DIR}/mesh.mesh
            DEPENDS ${GB_SHADER_DIR}/mesh.mesh VERBATIM)
          list(APPEND mesh_spv ${out})
        endforeach()
      endforeach()
    endforeach()
  endforeach()
endforeach()
set(out ${CMAKE_CURRENT_BINARY_DIR}/spv/mesh_task.task.spv)
add_custom_command(OUTPUT ${out}
  COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/spv
  COMMAND ${GLSLANG} -V --target-env vulkan1.3 -o ${out} ${GB_SHADER_DIR}/mesh.task
  DEPENDS ${GB_SHADER_DIR}/mesh.task VERBATIM)
list(APPEND mesh_spv ${out})
foreach(pp 0 1)
  foreach(h 0 1)
    set(out ${CMAKE_CURRENT_BINARY_DIR}/spv/mesh_frag_p${pp}_h${h}.frag.spv)
    add_custom_command(OUTPUT ${out}
      COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/spv
      COMMAND ${GLSLANG} -V --target-env vulkan1.3 -DPERPRIM=${pp} -DHEAVY=${h} -o ${out} ${GB_SHADER_DIR}/mesh.frag
      DEPENDS ${GB_SHADER_DIR}/mesh.frag VERBATIM)
    list(APPEND mesh_spv ${out})
  endforeach()
endforeach()
add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/mesh_shaders.c
  COMMAND ${CMAKE_COMMAND} "-DSPV_FILES=${mesh_spv}" -DTABLE=gb_mesh_spv
          -DOUT=${CMAKE_CURRENT_BINARY_DIR}/mesh_shaders.c -P ${GB_SHADER_DIR}/embed.cmake
  DEPENDS ${mesh_spv} ${GB_SHADER_DIR}/embed.cmake VERBATIM)

add_executable(gpubench-vk src/main.c src/tests.c ${CMAKE_CURRENT_BINARY_DIR}/shaders.c
               ${CMAKE_CURRENT_BINARY_DIR}/mesh_shaders.c)
target_include_directories(gpubench-vk PRIVATE src)
target_compile_options(gpubench-vk PRIVATE -Wall -Wextra -Wno-missing-field-initializers)
target_compile_definitions(gpubench-vk PRIVATE GB_SUITE_SHA="${GB_SUITE_SHA}")
target_link_libraries(gpubench-vk PRIVATE gb_vk)
