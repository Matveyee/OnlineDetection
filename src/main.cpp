


#include <liveMedia.hh>
#include <BasicUsageEnvironment.hh>
#include <GroupsockHelper.hh>
#include <iostream>
#include <vector>
#include <mutex>
#include "opencv2/opencv.hpp"
#include "CamFramedSource.hpp"

// Параметры потока
#define RTSP_PORT 8554
#define STREAM_NAME "live"
UsageEnvironment* env;
RTSPServer* rtspServer;

std::vector<uchar> jpegBuffer;
using namespace std;

// ==== Подсессия RTSP ====
class CustomVideoServerMediaSubsession : public OnDemandServerMediaSubsession {
public:
    static CustomVideoServerMediaSubsession* createNew(UsageEnvironment& env) {
        return new CustomVideoServerMediaSubsession(env);
    }

protected:
    CustomVideoServerMediaSubsession(UsageEnvironment& env)
        : OnDemandServerMediaSubsession(env, true) {
	std::cout << "Did we get here customVSMS" << std::endl;
	}

    JPEGVideoSource* createNewStreamSource(unsigned /*clientSessionId*/, unsigned& estBitrate) override {
	std::cout << "Did we get here FrmdSrce" << std::endl;
        estBitrate = 500; // Оценка битрейта (500 кбит/с)
        return CustomFramedSource::createNew(envir());
    }

    RTPSink* createNewRTPSink(Groupsock* rtpGroupsock, unsigned char rtpPayloadTypeIfDynamic, FramedSource* inputSource) override {
        std::cout << "Did we get here rtsplink" << std::endl;
	return JPEGVideoRTPSink::createNew(envir(), rtpGroupsock);
    }
};

// ==== Запуск RTSP-сервера ====
void setupRTSPServer() {
    TaskScheduler* scheduler = BasicTaskScheduler::createNew();
    env = BasicUsageEnvironment::createNew(*scheduler);

    rtspServer = RTSPServer::createNew(*env, RTSP_PORT);
    if (!rtspServer) {
        *env << "Ошибка запуска RTSP-сервера: " << env->getResultMsg() << "\n";
        exit(1);
    }

    ServerMediaSession* sms = ServerMediaSession::createNew(*env, STREAM_NAME, "Live Stream", "RTSP Custom stream");
    sms->addSubsession(CustomVideoServerMediaSubsession::createNew(*env));
    rtspServer->addServerMediaSession(sms);

    char* url = rtspServer->rtspURL(sms);
    std::cout << "RTSP Stream доступен по: " << url << std::endl;
    delete[] url;
}


int main(int argc, char* argv[]) {

    setupRTSPServer();
    env->taskScheduler().doEventLoop();
    return 0;
}
