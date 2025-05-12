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
    CUDA_ARCHITECTURES "75"
    POSITION_INDEPENDENT_CODE ON
)

target_sources(cudaviz
    PRIVATE
        cudaviz.cu
        diffusion.cpp
        mandelbrot.cpp
)


################################################################################
# an exectuable
add_executable(cudaviz_example)
add_executable(k20shores::cudaviz_example ALIAS cudaviz_example)

target_compile_features(cudaviz_example PRIVATE cxx_std_20)
target_include_directories(cudaviz_example
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(cudaviz_example 
    PUBLIC 
        cudaviz
        cuda
        cudart
)

set_target_properties(cudaviz_example PROPERTIES 
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES "75"
    POSITION_INDEPENDENT_CODE ON
)

target_sources(cudaviz_example
    PRIVATE
        main.cu
)