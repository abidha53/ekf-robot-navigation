cmake_minimum_required(VERSION 3.15)
project(robot_sim_hybrid_ekf CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
    GIT_SHALLOW TRUE
)

FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.13.6
    GIT_SHALLOW TRUE
)

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.15.2
    GIT_SHALLOW TRUE
)

set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(eigen pybind11 googletest)

set(EIGEN_INCLUDE_DIR ${eigen_SOURCE_DIR})

add_library(ekf_lib STATIC
    cpp/src/ekf.cpp
)
target_include_directories(ekf_lib PUBLIC
    cpp/include
    ${EIGEN_INCLUDE_DIR}
)

pybind11_add_module(ekf_cpp cpp/bindings.cpp)
target_link_libraries(ekf_cpp PRIVATE ekf_lib)
target_include_directories(ekf_cpp PRIVATE cpp/include)

set_target_properties(ekf_cpp PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/python
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/python
)

enable_testing()

add_executable(test_ekf
    cpp/tests/test_ekf.cpp
)
target_link_libraries(test_ekf PRIVATE
    ekf_lib
    gtest_main
)
target_include_directories(test_ekf PRIVATE cpp/include)

include(GoogleTest)
gtest_discover_tests(test_ekf)
