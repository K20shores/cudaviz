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

set_property(TARGET cudaviz PROPERTY CUDA_STANDARD 20)
set_property(TARGET cudaviz PROPERTY CUDA_STANDARD_REQUIRED ON)
set_property(TARGET cudaviz PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET cudaviz PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

################################################################################
# the executable

target_sources(cudaviz
    PRIVATE
        cudaviz.cu
)

add_executable(main
    main.cpp
)
target_link_libraries(main
    PRIVATE
        cudaviz
        cuda
        cudart
)

set_property(TARGET main PROPERTY CUDA_STANDARD 20)
set_property(TARGET main PROPERTY CUDA_STANDARD_REQUIRED ON)
set_property(TARGET main PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET main PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)