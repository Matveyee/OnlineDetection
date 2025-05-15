


#include <liveMedia.hh>
#include <BasicUsageEnvironment.hh>
#include <GroupsockHelper.hh>
#include <iostream>
#include <vector>
#include <mutex>
#include "opencv2/opencv.hpp"


class CustomFramedSource :  public JPEGVideoSource{
public:

    static CustomFramedSource* createNew(UsageEnvironment& env);
    Boolean isJPEGVideoSource() const override {return True;};
private:


    CustomFramedSource(UsageEnvironment& env);
    cv::VideoCapture cap;

public:

    static constexpr u_int8_t std_luminance_qt[64] = {
     16,  11,  10,  16,  24,  40,  51,  61,
     12,  12,  14,  19,  26,  58,  60,  55,
     14,  13,  16,  24,  40,  57,  69,  56,
     14,  17,  22,  29,  51,  87,  80,  62,
     18,  22,  37,  56,  68, 109, 103,  77,
     24,  35,  55, 64,  81, 104, 113,  92,
     49,  64,  78, 87, 103, 121, 120, 101,
     72,  92, 95, 98, 112, 100, 103, 99
    };
    static constexpr u_int8_t std_chrominance_qt[64] = {
     17,  18,  24,  47,  99,  99,  99,  99,
     18,  21,  26,  66,  99,  99,  99,  99,
     24,  26,  56,  99,  99,  99,  99,  99,
     47,  66,  99,  99,  99,  99,  99,  99,
     99,  99,  99,  99,  99,  99,  99,  99,
     99,  99,  99,  99,  99,  99,  99,  99,
     99,  99,  99,  99,  99,  99,  99,  99,
     99,  99,  99,  99,  99,  99,  99,  99
    };

    void* getImage(size_t& size);

    u_int8_t const* quantizationTables(u_int8_t& precision, u_int16_t& length) override;

   u_int8_t type() override {return 1;}
    u_int8_t qFactor() override {return 128;}
    u_int8_t width() override {return 80;}
    u_int8_t height() override {return 60;}
    void doGetNextFrame() override;

};