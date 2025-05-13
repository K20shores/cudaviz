pybind11_add_module(_cudaviz
  binding.cpp
)

target_compile_features(_cudaviz PRIVATE cxx_std_20)

target_link_libraries(_cudaviz
  PRIVATE
    k20shores::cudaviz
)

if (APPLE)
  # set the rpath for the shared library
  set_target_properties(_cudaviz PROPERTIES
    INSTALL_RPATH "@loader_path"
    BUILD_WITH_INSTALL_RPATH TRUE
  )
elseif(UNIX)
  # set the rpath for the shared library
  set_target_properties(_cudaviz PROPERTIES
    INSTALL_RPATH "$ORIGIN"
    BUILD_WITH_INSTALL_RPATH TRUE
  )
endif()

if(WIN32)
  # makefiles on windows don't need the config directory
  if (${CMAKE_GENERATOR} MATCHES "MinGW Makefiles")
      set(PYTHON_MODULE_PATH "${CMAKE_CURRENT_BINARY_DIR}")
  else()
  # but visual studio does
      set(PYTHON_MODULE_PATH "${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>")
  endif()
else()
  set(PYTHON_MODULE_PATH "${CMAKE_CURRENT_BINARY_DIR}")
endif()

install(TARGETS _cudaviz LIBRARY DESTINATION .)