cmake_minimum_required(VERSION 2.8)
project(computer_vision)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package( OpenCV REQUIRED )
 
include_directories(include ${OpenCV_INCLUDE_DIRS})
 
#Can manually add the sources using the set command as follows:
#set(SOURCES src/main.cpp)
 
#However, the file(GLOB...) allows for wildcard additions:
file(GLOB SOURCES "src/*.cpp")
 
add_executable( ${PROJECT_NAME} ${SOURCES} )
target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS} )

# copy files in the data folder into build directory
file(COPY data DESTINATION ${CMAKE_BINARY_DIR})