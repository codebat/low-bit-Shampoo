# for linux with GCC
# cmake -B build . && cmake --build build

# for windows with MSVC
# cmake -B build .
# cmake --build build --config Release

cmake_minimum_required(VERSION 3.22.1)

project(qtensor LANGUAGES CXX)

# Define included source files
set(CPP_FILES csrc/pythonInterface.cpp)
set(CUDA_FILES csrc/ops.cu csrc/kernels.cu)

list(APPEND SRC_FILES ${CPP_FILES})
list(APPEND SRC_FILES ${CUDA_FILES})

set(EX_OUTPUT_NAME "qtensor")

enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

# Convert the CUDA version from X.Y.z to XY. There's probably a shorter way of doing this
string(REGEX MATCH "^[0-9]+.[0-9]+" _CUDA_VERSION_FIRST_TWO "${CMAKE_CUDA_COMPILER_VERSION}")
string(REPLACE "." "" CUDA_VERSION_SHORT "${_CUDA_VERSION_FIRST_TWO}")

message(STATUS "CUDA Version: ${CUDA_VERSION_SHORT} (${CMAKE_CUDA_COMPILER_VERSION})")
message(STATUS "CUDA Compiler: ${CMAKE_CUDA_COMPILER}")

string(APPEND EX_OUTPUT_NAME "_cuda${CUDA_VERSION_SHORT}")

if(CMAKE_HOST_SYSTEM_NAME MATCHES "Windows")
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX2 /fp:fast")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /arch:AVX2 /fp:fast")
endif()

set_source_files_properties(${CPP_FILES} PROPERTIES LANGUAGE CXX)
add_library(qtensor SHARED ${SRC_FILES})
target_compile_features(qtensor PUBLIC cxx_std_14)
target_include_directories(qtensor PUBLIC csrc include)
target_include_directories(qtensor PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

set_target_properties(qtensor
    PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)

if(CMAKE_HOST_SYSTEM_NAME MATCHES "Windows")
    set_target_properties(qtensor PROPERTIES PREFIX "lib")
    set_target_properties(qtensor PROPERTIES LIBRARY_OUTPUT_DIRECTORY_RELEASE qtensor)
    set_target_properties(qtensor PROPERTIES LIBRARY_OUTPUT_DIRECTORY_DEBUG qtensor)
    set_target_properties(qtensor PROPERTIES RUNTIME_OUTPUT_DIRECTORY_RELEASE qtensor)
    set_target_properties(qtensor PROPERTIES RUNTIME_OUTPUT_DIRECTORY_DEBUG qtensor)
endif()

set_target_properties(qtensor PROPERTIES OUTPUT_NAME ${EX_OUTPUT_NAME})
set_target_properties(qtensor PROPERTIES LIBRARY_OUTPUT_DIRECTORY qtensor)
