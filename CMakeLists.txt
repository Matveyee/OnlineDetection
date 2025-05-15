
project(rtsp)
cmake_minimum_required(VERSION 3.22)

include_directories("/usr/local/include/BasicUsageEnvironment")
include_directories("/usr/local/include/groupsock")
include_directories("/usr/local/include/liveMedia")
include_directories("/usr/local/include/opencv4")
include_directories("/usr/local/include/UsageEnvironment")
include_directories(${PROJECT_SOURCE_DIR}/include)

set(STAITC_PATH "/usr/local/lib")

link_directories("/usr/local/lib")
link_directories("/usr/lib/x86_64-linux-gnu")

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR})

add_executable(main main1.cpp CamFramedSource.cpp)

target_link_libraries(main  ${STAITC_PATH}/libliveMedia.a ${STAITC_PATH}/libgroupsock.a ${STAITC_PATH}/libBasicUsageEnvironment.a ${STAITC_PATH}/libUsageEnvironment.a -lopencv_core -lopencv_videoio -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc -lssl -lcrypto)
