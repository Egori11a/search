cmake_minimum_required(VERSION 3.16)
project(lab6_boolean_index CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(core
  src/common.cpp
  src/tokenizer.cpp
  src/stemmer_ru.cpp
)

target_include_directories(core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)

add_executable(lab6_build_index src/lab6_build_index.cpp)
target_link_libraries(lab6_build_index core)

add_executable(lab6_bool_search src/lab6_bool_search.cpp)
target_link_libraries(lab6_bool_search core)
