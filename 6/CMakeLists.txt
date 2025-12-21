cmake_minimum_required(VERSION 3.16)
project(lab6_boolean_index CXX)

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

target_include_directories(core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src ${MONGOC_INCLUDE_DIRS} ${BSON_INCLUDE_DIRS})
target_link_directories(core PUBLIC ${MONGOC_LIBRARY_DIRS} ${BSON_LIBRARY_DIRS})
target_compile_options(core PUBLIC ${MONGOC_CFLAGS_OTHER} ${BSON_CFLAGS_OTHER})
target_link_libraries(core PUBLIC ${MONGOC_LIBRARIES} ${BSON_LIBRARIES})

target_include_directories(core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)

target_include_directories(core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)

add_executable(lab6_build_index src/lab6_build_index.cpp)
target_link_libraries(lab6_build_index core)

add_executable(lab6_bool_search src/lab6_bool_search.cpp)
target_link_libraries(lab6_bool_search core)
