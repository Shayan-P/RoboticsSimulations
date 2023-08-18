project("casadi_test")

find_package(casadi REQUIRED)
find_package(pinocchio)

add_executable(main main.cpp)
target_link_libraries(main casadi pinocchio)
