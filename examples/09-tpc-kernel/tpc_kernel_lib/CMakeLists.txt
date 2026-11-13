cmake_minimum_required(VERSION 3.10)
project(add1_kernels CXX C)

# Find tpc-clang and habana tools
find_program(TPC_CLANG tpc-clang REQUIRED)
find_path(HABANA_INCLUDE_DIR gc_interface.h
          PATHS /usr/lib/habanatools/include
          REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Step 1: Compile the TPC kernel (.c -> .o using tpc-clang)
# Use canonical -march=gaudi2 flag from src/Habana_Custom_Kernel/src/CMakeLists.txt
set(TPC_KERNEL_SRC ${CMAKE_SOURCE_DIR}/kernels/gaudi2/add1_fwd_f32_gaudi2.c)
# kernel_config.h is vendored beside the kernel (see ../README.md, "What was
# changed from upstream"). Upstream pointed this at an absolute path inside one
# developer's checkout of HabanaAI/Habana_Custom_Kernel.
set(TPC_INCLUDE_DIR_KERNELS ${CMAKE_SOURCE_DIR}/kernels/include)
set(TPC_KERNEL_OBJ ${CMAKE_BINARY_DIR}/add1_fwd_f32_gaudi2.o)

add_custom_command(
    OUTPUT ${TPC_KERNEL_OBJ}
    COMMAND ${TPC_CLANG} -Wall -Werror -march=gaudi2 -O2
            -I${HABANA_INCLUDE_DIR}/ -I${TPC_INCLUDE_DIR_KERNELS}
            ${TPC_KERNEL_SRC} -c
    DEPENDS ${TPC_KERNEL_SRC}
    COMMENT "Compiling TPC kernel with tpc-clang"
)

# Step 2: Convert to elf64-x86-64 with -B i386:x86-64 (canonical flag)
# Use symlink for filename so symbols are _binary___add1_fwd_f32_gaudi2_o_*
set(TPC_KERNEL_SYMLINK ${CMAKE_BINARY_DIR}/__add1_fwd_f32_gaudi2.o)
set(TPC_KERNEL_EMBED_OBJ ${CMAKE_BINARY_DIR}/__add1_fwd_f32_gaudi2_kernel.o)

add_custom_command(
    OUTPUT ${TPC_KERNEL_EMBED_OBJ}
    COMMAND ${CMAKE_COMMAND} -E create_symlink add1_fwd_f32_gaudi2.o ${TPC_KERNEL_SYMLINK}
    COMMAND cd ${CMAKE_BINARY_DIR} && objcopy -I binary -O elf64-x86-64 -B i386:x86-64
            --rename-section .data=.rodata,alloc,load,readonly,data,contents
            ${TPC_KERNEL_SYMLINK} ${TPC_KERNEL_EMBED_OBJ}
    DEPENDS ${TPC_KERNEL_OBJ}
    COMMENT "Embedding TPC kernel binary into linkable .o"
)

# Step 3: Build the shared library
add_library(add1_kernels SHARED
    src/entry_points.cpp
    src/gaudi2_src/add1_fwd_f32_gaudi2.cpp
    ${TPC_KERNEL_EMBED_OBJ}
)

target_include_directories(add1_kernels PRIVATE
    ${HABANA_INCLUDE_DIR}
    src/gaudi2_src
)

target_compile_options(add1_kernels PRIVATE -O2 -Wall -fPIC)

target_link_options(add1_kernels PRIVATE
    "-Wl,--allow-multiple-definition"
)

set_target_properties(add1_kernels PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

# Rename to lib*.so (GC convention)
add_custom_command(
    TARGET add1_kernels POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E rename
        $<TARGET_FILE:add1_kernels>
        ${CMAKE_BINARY_DIR}/libadd1_kernels.so
    COMMENT "Renaming to lib*.so"
)
