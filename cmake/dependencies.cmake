include(FetchContent)

################################################################################
# pybind11
set(PYBIND11_NEWPYTHON ON)

FetchContent_Declare(pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11
    GIT_TAG        v2.12.0
)

FetchContent_MakeAvailable(pybind11)