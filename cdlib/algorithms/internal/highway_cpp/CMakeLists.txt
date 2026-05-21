cmake_minimum_required(VERSION 3.16)
project(highway LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenMP)

add_executable(highway
    src/main.cpp
    src/graph_io.cpp
    src/csr.cpp
    src/highway.cpp
    src/anchors.cpp
    src/propagation.cpp
    src/refinement.cpp
    src/decode.cpp
)

target_include_directories(highway PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(highway PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(highway PRIVATE HIGHWAY_USE_OPENMP)
endif()

if(MSVC)
    target_compile_options(highway PRIVATE /O2)
else()
    target_compile_options(highway PRIVATE -O3)
endif()