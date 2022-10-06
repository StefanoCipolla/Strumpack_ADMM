cmake_minimum_required(VERSION 3.13)
project(myapp LANGUAGES CXX)
 
find_package(STRUMPACK REQUIRED)

add_executable(SVM_ADMM KernelSVM_final.cpp)
target_link_libraries(SVM_ADMM PRIVATE STRUMPACK::strumpack)

