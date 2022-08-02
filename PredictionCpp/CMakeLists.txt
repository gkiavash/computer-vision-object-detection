cmake_minimum_required(VERSION 2.8)
project( FinalProjectV2 )
include_directories("library/")
find_package( OpenCV REQUIRED )
add_executable( FinalProjectV2 main.cpp library/MeanShift.cpp library/MeanShift.h)
target_link_libraries( FinalProjectV2 ${OpenCV_LIBS} )

