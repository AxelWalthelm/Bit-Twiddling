# See https://developer.nvidia.com/blog/building-cuda-applications-cmake/

cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "$ENV{HOME}")

project(BitTwiddlingProject LANGUAGES CXX CUDA)

include(CTest)

# WARNING: compile for one architecture only to avoid headers being compiled multiple times with different __CUDA_ARCH__ defined.
#add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_53,code=sm_53>")  # target GPU old
add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_72,code=sm_72>")  # target GPU new

# add_compile_options must be before add_library, add_executable
add_compile_options("$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:--debug;--device-debug;--optimize=0>")
add_compile_options("$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:-DNDEBUG;-lineinfo>")  # NDEBUG for assert in host code of .cu files
add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:--relocatable-device-code=true>")  # often recommended, but why exactly?
add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>")  # std::numeric_limits


list(APPEND cxx_warning_flags "-Wall" "-Wextra") # "-Werror"  "-fdiagnostics-show-option" "-Wno-type-limits"
foreach(flag IN LISTS cxx_warning_flags)
    add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:--compiler-options=${flag}>")
    add_compile_options("$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:${flag}>")
endforeach()
add_compile_options("$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:-Wpedantic>")  # cuda generated code does not agree with -Wpedantic


add_library(BitTwiddling STATIC
  BitTwiddling.cpp
  BitTwiddling.cu
  cuda_device_time.cu
)

target_link_libraries(BitTwiddling ${CUDA_LIBRARIES})

# Request that BitTwiddling be built with -std=c++11
# As this is a public compile feature anything that links to BitTwiddling
# will also build with -std=c++11
target_compile_features(BitTwiddling PUBLIC cxx_std_11)

# We need to explicitly state that we need all CUDA files in the BitTwiddling
# library to be built with -dc as the member functions could be called by
# other libraries and executables
set_target_properties(BitTwiddling PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(BitTwiddling PROPERTIES CUDA_ARCHITECTURES "35;50;72")

if(BUILD_TESTING)

  add_executable(BitTwiddling_test
    main.cpp
    ../semi_exhaustive_search_for_8bit_rev.cpp
    HighResolutionTimer.cpp
  )

  target_compile_features(BitTwiddling_test PUBLIC cxx_std_11)

  set_target_properties(BitTwiddling_test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  #set_target_properties(BitTwiddling_test PROPERTIES CUDA_ARCHITECTURES "35;50;72")
  target_link_libraries(BitTwiddling_test PRIVATE BitTwiddling
      /usr/local/cuda/lib64/libcudart.so  # why is this needed?
  )

  #add_test(NAME BitTwiddling_10k COMMAND BitTwiddling_test 10000 )
  #add_test(NAME BitTwiddling_256k COMMAND BitTwiddling_test 256000 )

#  set_property(TARGET BitTwiddling_test 
#               PROPERTY
#               BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})

endif()
