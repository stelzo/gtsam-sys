cmake_minimum_required(VERSION 3.29)
project(gtsam_imu)

find_package(gtsam 4.2 QUIET)

set(CMAKE_POLICY_VERSION_MINIMUM 3.5)


if(gtsam_FOUND)
    message(STATUS "Found GTSAM version ${gtsam_VERSION}")
    add_library(gtsam_imu gtsam_imu.cpp)
    add_dependencies(gtsam_imu gtsam)
    target_link_libraries(gtsam_imu gtsam)
    target_include_directories(gtsam_imu PUBLIC ${gtsam_INCLUDE_DIRS})
else()
    message(STATUS "GTSAM package not found. Building from source.")

    # override with cache and force
    set(GTSAM_USE_SYSTEM_EIGEN TRUE CACHE BOOL "Use Eigen3 from system")
    set(GTSAM_ENABLE_BOOST_SERIALIZATION ON CACHE BOOL "Enable Boost Serialization")
    set(GTSAM_USE_BOOST_FEATURES OFF CACHE BOOL "Use Boost features")
    set(GTSAM_BUILD_UNSTABLE OFF CACHE BOOL "Build unstable features")
    set(GTSAM_UNSTABLE_BUILD_PYTHON OFF CACHE BOOL "Build python bindings for unstable features")
    set(GTSAM_UNSTABLE_INSTALL_MATLAB_TOOLBOX OFF CACHE BOOL "Install MATLAB toolbox for unstable features")
    set(GTSAM_BUILD_EXAMPLES OFF CACHE BOOL "Build examples")
    set(GTSAM_BUILD_TESTS OFF CACHE BOOL "Build tests")
    set(GTSAM_BUILD_EXAMPLES_ALWAYS OFF CACHE BOOL "Build examples")
    set(GTSAM_BUILD_TIMING_ALWAYS OFF CACHE BOOL "Build timing")
    set(GTSAM_BUILD_DOCS OFF CACHE BOOL "Build documentation")
    set(GTSAM_BUILD_PYTHON OFF CACHE BOOL "Build Python bindings")
    set(GTSAM_INSTALL_MATLAB_TOOLBOX OFF CACHE BOOL "Build MATLAB bindings")
    set(GTSAM_TANGENT_PREINTEGRATION OFF CACHE BOOL "Use tangent preintegration")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")

    add_subdirectory(gtsam)

    set(CMAKE_CXX_STANDARD 17)

    add_library(gtsam_imu gtsam_imu.cpp)
    add_dependencies(gtsam_imu gtsam)
    target_link_libraries(gtsam_imu gtsam)
    target_include_directories(gtsam_imu PUBLIC ${GTSAM_INCLUDE_DIRS})
endif()


