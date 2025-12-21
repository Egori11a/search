cmake_minimum_required(VERSION 3.16)
project(search_labs_345 CXX)

find_package(PkgConfig REQUIRED)
pkg_check_modules(MONGOC REQUIRED libmongoc-1.0)
pkg_check_modules(BSON REQUIRED libbson-1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(core
  src/common.cpp
  src/tokenizer.cpp
  src/stemmer_ru.cpp
  src/mongo_corpus.cpp
)

target_include_directories(core PRIVATE ${MONGOC_INCLUDE_DIRS} ${BSON_INCLUDE_DIRS})
target_compile_options(core PRIVATE ${MONGOC_CFLAGS_OTHER} ${BSON_CFLAGS_OTHER})
target_link_libraries(core PRIVATE ${MONGOC_LIBRARIES} ${BSON_LIBRARIES})

add_executable(lab3_token_stats src/lab3_token_stats.cpp)
target_link_libraries(lab3_token_stats core)

add_executable(lab4_zipf src/lab4_zipf.cpp)
target_link_libraries(lab4_zipf core)

add_executable(lab5_search src/lab5_search.cpp)
target_link_libraries(lab5_search core)
