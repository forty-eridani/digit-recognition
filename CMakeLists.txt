cmake_minimum_required(VERSION 3.20)
project(digit_recognition)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_BUILD_TYPE Debug)
set(COMPILE_DIR src/c)

set(SOURCE_FILES ${COMPILE_DIR}/main.c
				 ${COMPILE_DIR}/network.c
			 	 ${COMPILE_DIR}/mnist.c)
set(EXECUTABLE_NAME out)

include_directories(src/include linear-algebra-lib/src/include input)

add_executable(${EXECUTABLE_NAME} ${SOURCE_FILES})

target_link_libraries(${EXECUTABLE_NAME} ${CMAKE_SOURCE_DIR}/libs/liblinear-algebra.a m)
