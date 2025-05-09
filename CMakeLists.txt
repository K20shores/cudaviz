################################################################################
# Preamble
cmake_minimum_required(VERSION 3.21)

# project and version must be on the same line so that the docs can extract it
project(cudaviz VERSION 0.0.0 LANGUAGES CXX CUDA)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
      "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel."
      FORCE)
endif(NOT CMAKE_BUILD_TYPE)

message(STATUS "CMake build configuration for ${PROJECT}(${CMAKE_BUILD_TYPE}) ${PROJECT_VERSION}")

################################################################################
# Projet wide setup options

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${PROJECT_SOURCE_DIR}/cmake")

################################################################################
# Dependencies

include(cmake/dependencies.cmake)

################################################################################
# cuda visualization library

add_subdirectory(src)

################################################################################
# cuda visualization python library

# add_subdirectory(cudaviz)