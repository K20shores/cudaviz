################################################################################
# the library
add_library(cudaviz)
add_library(k20shores::cudaviz ALIAS cudaviz)

target_compile_features(cudaviz PRIVATE cxx_std_20)

target_include_directories(cudaviz
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(cudaviz 
    PUBLIC 
        cuda
        cudart
)

set_target_properties(cudaviz PROPERTIES 
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES "all_major"
)

target_sources(cudaviz
    PRIVATE
        cudaviz.cu
        mandelbrot.cpp
)
