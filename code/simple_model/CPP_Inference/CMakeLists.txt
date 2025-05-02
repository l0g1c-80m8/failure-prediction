cmake_minimum_required(VERSION 3.12)
project(robot_risk_inference)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find ONNX Runtime package
# You'll need to set ONNXRUNTIME_ROOT to your ONNX Runtime installation directory
# e.g., -DONNXRUNTIME_ROOT=/path/to/onnxruntime
if(DEFINED ENV{ONNXRUNTIME_ROOT})
    set(ONNXRUNTIME_ROOT $ENV{ONNXRUNTIME_ROOT})
endif()

if(NOT DEFINED ONNXRUNTIME_ROOT)
    message(FATAL_ERROR "ONNXRUNTIME_ROOT is not defined. Please set it to your ONNX Runtime installation directory.")
endif()

# Check for different include directory structures
if(EXISTS "${ONNXRUNTIME_ROOT}/include/onnxruntime")
    include_directories(${ONNXRUNTIME_ROOT}/include)
elseif(EXISTS "${ONNXRUNTIME_ROOT}/include")
    include_directories(${ONNXRUNTIME_ROOT}/include)
else()
    # Try to find include directory by looking for the C API header
    file(GLOB_RECURSE ORT_HEADER_FILE
      "${ONNXRUNTIME_ROOT}/**/onnxruntime_c_api.h"
    )
    if(ORT_HEADER_FILE)
      get_filename_component(ORT_INCLUDE_DIR "${ORT_HEADER_FILE}" DIRECTORY)
      get_filename_component(ORT_INCLUDE_DIR "${ORT_INCLUDE_DIR}" DIRECTORY)
      get_filename_component(ORT_INCLUDE_DIR "${ORT_INCLUDE_DIR}" DIRECTORY)
      include_directories(${ORT_INCLUDE_DIR})
      message(STATUS "Found ONNX Runtime include directory: ${ORT_INCLUDE_DIR}")
    else()
      message(FATAL_ERROR "Could not find onnxruntime_c_api.h in ${ONNXRUNTIME_ROOT}. Please check your ONNX Runtime installation.")
    endif()
endif()

# Find ONNX Runtime library
if(WIN32)
    set(ONNXRUNTIME_LIB "${ONNXRUNTIME_ROOT}/lib/onnxruntime.lib")
else()
    # Try different possible library paths and names
    find_library(ONNXRUNTIME_LIB 
        NAMES onnxruntime libonnxruntime onnxruntime_providers_shared
        PATHS 
        "${ONNXRUNTIME_ROOT}/lib"
        "${ONNXRUNTIME_ROOT}/lib64"
        "${ONNXRUNTIME_ROOT}"
        NO_DEFAULT_PATH
    )
    
    if(NOT ONNXRUNTIME_LIB)
        # If not found directly, try to find it using glob
        file(GLOB_RECURSE ORT_LIB_FILES
          "${ONNXRUNTIME_ROOT}/**/*onnxruntime*.so"
          "${ONNXRUNTIME_ROOT}/**/*onnxruntime*.dylib"
        )
        
        if(ORT_LIB_FILES)
            list(GET ORT_LIB_FILES 0 ONNXRUNTIME_LIB)
            message(STATUS "Found ONNX Runtime library: ${ONNXRUNTIME_LIB}")
        else()
            message(FATAL_ERROR "Could not find ONNX Runtime library in ${ONNXRUNTIME_ROOT}. Please check your installation.")
        endif()
    endif()
endif()

# Add the executable
add_executable(risk_inference risk_inference.cpp)

# Link against ONNX Runtime
target_link_libraries(risk_inference ${ONNXRUNTIME_LIB})

# If using CUDA, you might need additional libraries
option(USE_CUDA "Enable CUDA support" OFF)
if(USE_CUDA)
    find_package(CUDA REQUIRED)
    include_directories(${CUDA_INCLUDE_DIRS})
    target_link_libraries(risk_inference ${CUDA_LIBRARIES})
endif()

# Installation
install(TARGETS risk_inference DESTINATION bin)