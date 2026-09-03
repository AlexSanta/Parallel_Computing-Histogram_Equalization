cmake_minimum_required(VERSION 3.15)
project(Progetto_Histogram_Equalization)

set(CMAKE_CXX_STANDARD 14)

if(APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xpreprocessor -fopenmp")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lomp")
    # Get Homebrew home directory
    execute_process(
            COMMAND brew --prefix
            OUTPUT_VARIABLE HOMEBREW_PREFIX
            OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    include_directories(${HOMEBREW_PREFIX}/opt/libomp/include)
    link_directories(${HOMEBREW_PREFIX}/opt/libomp/lib)
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
endif()

add_executable(Progetto_Histogram_Equalization main.cpp aos_equalization.cpp aos_equalization.h soa_equalization.cpp soa_equalization.h STBImage.cpp STBImage.h equalization_utils.cpp equalization_utils.h)