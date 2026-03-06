cmake_minimum_required(VERSION 3.16)
project(SortSearchDemo LANGUAGES CXX)

# 启用 C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 把所有 cpp 文件编译进一个可执行文件
add_executable(cmake_run
    main.cpp
    sort_numlist.cpp
    find_numlist.cpp
)

# 因为使用了头文件 要使用下面代码
target_include_directories(cmake_run PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/algo
)