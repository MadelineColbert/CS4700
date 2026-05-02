cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(CS4700 LANGUAGES CXX CUDA)

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

find_package(CUDA REQUIRED)

<<<<<<< HEAD
set(PROJECT_SOURCES 
=======
set(PROJECT_SOURCES
>>>>>>> main
    main.cu
)

add_executable(${PROJECT_NAME} ${PROJECT_SOURCES})


target_link_libraries(${PROJECT_NAME} ${CUDA_CUBLAS_LIBRARIES})