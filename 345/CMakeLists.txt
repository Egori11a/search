cmake_minimum_required(VERSION 3.16)
project(search_labs_345 CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(core
  src/common.cpp
  src/tokenizer.cpp
  src/stemmer_ru.cpp
)

add_executable(lab3_token_stats src/lab3_token_stats.cpp)
target_link_libraries(lab3_token_stats core)

add_executable(lab4_zipf src/lab4_zipf.cpp)
target_link_libraries(lab4_zipf core)

add_executable(lab5_search src/lab5_search.cpp)
target_link_libraries(lab5_search core)
