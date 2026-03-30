cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(CS4700 LANGUAGES CXX CUDA)

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)


set(PROJECT_SOURCES 
    main.cpp
    src/test.cu
)

add_executable(${PROJECT_NAME} ${PROJECT_SOURCES})