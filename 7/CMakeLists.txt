cmake_minimum_required(VERSION 3.16)
project(lab7_boolean_search CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(core
  src/common.cpp
  src/tokenizer.cpp
  src/stemmer_ru.cpp
)
target_include_directories(core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)

add_executable(lab7_cli src/lab7_cli.cpp)
target_link_libraries(lab7_cli core)

add_executable(lab7_server src/lab7_server.cpp)
target_link_libraries(lab7_server core)
