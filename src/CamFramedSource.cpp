


#include <liveMedia.hh>
#include <BasicUsageEnvironment.hh>
#include <GroupsockHelper.hh>
#include <iostream>
#include <vector>
#include <mutex>
#include "opencv2/opencv.hpp"
#include "CamFramedSource.hpp"


std::mutex frameMutex;

    CustomFramedSource* CustomFramedSource::createNew(UsageEnvironment& env) {
        return new CustomFramedSource(env);
    }


    CustomFramedSource::CustomFramedSource(UsageEnvironment& env) :  JPEGVideoSource(env), cap("/dev/video0") {
        if (!cap.isOpened()) {
            env << "Failed to open camera\n";
        } else {
            cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
            cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
            cap.set(cv::CAP_PROP_FPS, 30);
            env << "Camera opened successfully\n";
        }
   		std::cout << "Restart Interval : " << restartInterval() << std::endl;
	}

    void* CustomFramedSource::getImage(size_t& size) {

        
        cv::Mat frame;
        cap >> frame;
        // Сжимаем изображение в формат JPEG
        std::vector<uchar> buf;
        std::vector<int> compression_params = {cv::IMWRITE_JPEG_QUALITY, 90};  // Качество сжатия 90
        cv::imencode(".jpg", frame, buf, compression_params);

        fFrameSize = std::min(buf.size(), static_cast<size_t>(fMaxSize));
        fNumTruncatedBytes = buf.size() - fFrameSize;

        // Возвращаем указатель на данные JPEG и их размер
        size = buf.size();
        uchar* jpegData = new uchar[size];
        std::memcpy(jpegData, buf.data(), size);
        
        return jpegData;
    }



   u_int8_t const* CustomFramedSource::quantizationTables(u_int8_t& precision, u_int16_t& length) {
    static u_int8_t qt[128]; // Таблица для Y + Cb/Cr (2×64)
    memcpy(qt, std_luminance_qt, 64);
    memcpy(qt + 64, std_chrominance_qt, 64);
    
    precision = 0;  // 0 = 8-битные таблицы
    length = sizeof(qt);
    return qt;
}

    
    void CustomFramedSource::doGetNextFrame() {
	std::cout << "[DEBUG] doGetNextFrame() entered" << std::endl;
        std::lock_guard<std::mutex> lock(frameMutex);
        size_t jpegSize = 0;
        
        void* jpegData = getImage(jpegSize);
	    std::cout << "May be here" << std::endl;
        if (!jpegData || jpegSize == 0) {
	    std::cout << "Problem point" << std::endl;
           envir().taskScheduler().scheduleDelayedTask(10000, (TaskFunc*)FramedSource::afterGetting, this);
            return;
        }

        if (jpegSize > fMaxSize) {
            jpegSize = fMaxSize; // Обрезаем, если буфер не помещается
        }

        memcpy(fTo, jpegData, fFrameSize);
        fPresentationTime.tv_usec += 33333;
        afterGetting(this);
    }