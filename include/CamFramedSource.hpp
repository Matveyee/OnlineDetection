


#include <liveMedia.hh>
#include <BasicUsageEnvironment.hh>
#include <GroupsockHelper.hh>
#include <iostream>
#include <vector>
#include <mutex>
#include "opencv2/opencv.hpp"
#include "hailo/hailort.hpp"


class CustomFramedSource :  public JPEGVideoSource{
public:

    static CustomFramedSource* createNew(UsageEnvironment& env, std::string source, std::string hef, int delay);
    Boolean isJPEGVideoSource() const override {return True;};
private:


    CustomFramedSource(UsageEnvironment& env, std::string source, std::string hef, int delay);
    
    static u_int8_t std_luminance_qt[64] ;
    static u_int8_t std_chrominance_qt[64];
    cv::VideoCapture cap;

public:

    
    std::unique_ptr<hailort::VDevice> vdevice;
    std::shared_ptr<hailort::InferModel> infer_model;
    hailort::ConfiguredInferModel configured_infer_model;
    int DELAY;

    std::string HEF_FILE;

    void getImage(size_t& size, std::vector<uchar>* data_to_write);

    u_int8_t const* quantizationTables(u_int8_t& precision, u_int16_t& length) override;

   u_int8_t type() override {return 1;}
    u_int8_t qFactor() override {return 128;}
    u_int8_t width() override {return 80;}
    u_int8_t height() override {return 60;}
    void doGetNextFrame() override;

};