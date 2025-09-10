cmake_minimum_required(VERSION 3.16)
project(phantom_go_project)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add the current directory to include path
include_directories(.)
include_directories(minizero)
include_directories(minizero/base)
include_directories(minizero/config)
include_directories(minizero/utils)

# Find required packages
find_package(Boost REQUIRED COMPONENTS iostreams system)

# Find all necessary source files for the minizero library
file(GLOB_RECURSE MINIZERO_SOURCES 
    "minizero/go/*.cpp"
    "minizero/utils/*.cpp"
    "minizero/config/*.cpp"
    "minizero/base/*.cpp"
)

# Create the executable
add_executable(main 
    src/main.cpp
    ${MINIZERO_SOURCES}
)

# Link libraries
target_link_libraries(main ${Boost_LIBRARIES})