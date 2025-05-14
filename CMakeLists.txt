cmake_minimum_required(VERSION 3.20)
project(digit_recognition)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(COMPILE_DIR src/c)

set(SOURCE_FILES ${COMPILE_DIR}/main.c)

include_directories(src/include linear-algebra-lib/src/include)

add_executable(out ${SOURCE_FILES})

target_link_libraries(out ${CMAKE_SOURCE_DIR}/libs/liblinear-algebra.a)
