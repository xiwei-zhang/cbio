cmake_minimum_required(VERSION 2.8)

project( TeleOphta )
# find_package( OpenCV REQUIRED )
INCLUDE_DIRECTORIES(/usr/local/include/opencv)
INCLUDE_DIRECTORIES(/usr/local/include/opencv2)

add_executable( TeleOphta TeleOphta )
# target_link_libraries( DisplayImage ${OpenCV_LIBS} )
#TARGET_LINK_LIBRARIES(TeleOphta /usr/local/lib/libopencv_core.so.2.4.3)
#TARGET_LINK_LIBRARIES(TeleOphta /usr/local/lib/libopencv_highgui.so.2.4.3)
#TARGET_LINK_LIBRARIES(TeleOphta /usr/local/lib/libopencv_imgproc.so.2.4.3)
TARGET_LINK_LIBRARIES(TeleOphta /usr/local/lib/libopencv_core.so.3.0.0)
TARGET_LINK_LIBRARIES(TeleOphta /usr/local/lib/libopencv_highgui.so.3.0.0)
TARGET_LINK_LIBRARIES(TeleOphta /usr/local/lib/libopencv_imgproc.so.3.0.0)


set(CMAKE_CXX_FLAGS "-O3")
