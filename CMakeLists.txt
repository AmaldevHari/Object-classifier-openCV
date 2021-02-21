cmake_minimum_required(VERSION 3.16)

project(main)

find_package(OpenCV 4.0.0 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LINK_DIRS})
add_executable(main main2.cpp object_CV.cpp)
target_link_libraries(main ${OpenCV_LIBS})

