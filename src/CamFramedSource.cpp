


#include <liveMedia.hh>
#include <BasicUsageEnvironment.hh>
#include <GroupsockHelper.hh>
#include <iostream>
#include <vector>
#include <mutex>
#include "opencv2/opencv.hpp"
#include "CamFramedSource.hpp"
#include <unistd.h> // для Unix систем

using namespace hailort;
std::mutex frameMutex;

#if defined(__unix__)
#include <sys/mman.h>
#endif

static std::shared_ptr<uint8_t> page_aligned_alloc(size_t size)
{
#if defined(__unix__)
    auto addr = mmap(NULL, size, PROT_WRITE | PROT_READ, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    if (MAP_FAILED == addr) throw std::bad_alloc();
    return std::shared_ptr<uint8_t>(reinterpret_cast<uint8_t*>(addr), [size](void *addr) { munmap(addr, size); });
#elif defined(_MSC_VER)
    auto addr = VirtualAlloc(NULL, size, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    if (!addr) throw std::bad_alloc();
    return std::shared_ptr<uint8_t>(reinterpret_cast<uint8_t*>(addr), [](void *addr){ VirtualFree(addr, 0, MEM_RELEASE); });
#else
#pragma error("Aligned alloc not supported")
#endif
}
struct NamedBbox {
    hailo_bbox_float32_t bbox;
    size_t class_id;
};

std::vector<NamedBbox> parse_nms_data(uint8_t* data, size_t max_class_count) {
    std::vector<NamedBbox> bboxes;
    size_t offset = 0;

    for (size_t class_id = 0; class_id < max_class_count; class_id++) {
        auto det_count = static_cast<uint32_t>(*reinterpret_cast<float32_t*>(data + offset));
        offset += sizeof(float32_t);

        for (size_t j = 0; j < det_count; j++) {
            hailo_bbox_float32_t bbox_data = *reinterpret_cast<hailo_bbox_float32_t*>(data + offset);
            offset += sizeof(hailo_bbox_float32_t);

            NamedBbox named_bbox;
            named_bbox.bbox = bbox_data;
            named_bbox.class_id = class_id + 1;
            bboxes.push_back(named_bbox);
        }
    }
    return bboxes;
}
std::string get_coco_name_from_int(int cls)
{
    std::string result = "N/A";
    switch(cls) {
        case 0:  result = "__background__";   break;
        case 1:  result = "person";           break;
        case 2:  result = "bicycle";          break;
        case 3:  result = "car";              break;
        case 4:  result = "motorcycle";       break;
        case 5:  result = "airplane";         break;
        case 6:  result = "bus";              break;
        case 7:  result = "train";            break;
        case 8:  result = "truck";            break;
        case 9:  result = "boat";             break;
        case 10: result = "traffic light";    break;
        case 11: result = "fire hydrant";     break;
        case 12: result = "stop sign";        break;
        case 13: result = "parking meter";    break;
        case 14: result = "bench";            break;
        case 15: result = "bird";             break;
        case 16: result = "cat";              break;
        case 17: result = "dog";              break;
        case 18: result = "horse";            break;
        case 19: result = "sheep";            break;
        case 20: result = "cow";              break;
        case 21: result = "elephant";         break;
        case 22: result = "bear";             break;
        case 23: result = "zebra";            break;
        case 24: result = "giraffe";          break;
        case 25: result = "backpack";         break;
        case 26: result = "umbrella";         break;
        case 27: result = "handbag";          break;
        case 28: result = "tie";              break;
        case 29: result = "suitcase";         break;
        case 30: result = "frisbee";          break;
        case 31: result = "skis";             break;
        case 32: result = "snowboard";        break;
        case 33: result = "sports ball";      break;
        case 34: result = "kite";             break;
        case 35: result = "baseball bat";     break;
        case 36: result = "baseball glove";   break;
        case 37: result = "skateboard";       break;
        case 38: result = "surfboard";        break;
        case 39: result = "tennis racket";    break;
        case 40: result = "bottle";           break;
        case 41: result = "wine glass";       break;
        case 42: result = "cup";              break;
        case 43: result = "fork";             break;
        case 44: result = "knife";            break;
        case 45: result = "spoon";            break;
        case 46: result = "bowl";             break;
        case 47: result = "banana";           break;
        case 48: result = "apple";            break;
        case 49: result = "sandwich";         break;
        case 50: result = "orange";           break;
        case 51: result = "broccoli";         break;
        case 52: result = "carrot";           break;
        case 53: result = "hot dog";          break;
        case 54: result = "pizza";            break;
        case 55: result = "donut";            break;
        case 56: result = "cake";             break;
        case 57: result = "chair";            break;
        case 58: result = "couch";            break;
        case 59: result = "potted plant";     break;
        case 60: result = "bed";              break;
        case 61: result = "dining table";     break;
        case 62: result = "toilet";           break;
        case 63: result = "tv";               break;
        case 64: result = "laptop";           break;
        case 65: result = "mouse";            break;
        case 66: result = "remote";           break;
        case 67: result = "keyboard";         break;
        case 68: result = "cell phone";       break;
        case 69: result = "microwave";        break;
        case 70: result = "oven";             break;
        case 71: result = "toaster";          break;
        case 72: result = "sink";             break;
        case 73: result = "refrigerator";     break;
        case 74: result = "book";             break;
        case 75: result = "clock";            break;
        case 76: result = "vase";             break;
        case 77: result = "scissors";         break;
        case 78: result = "teddy bear";       break;
        case 79: result = "hair drier";       break;
        case 80: result = "toothbrush";       break;
    }
    return result;
}
std::vector<cv::Scalar> COLORS = {
    cv::Scalar(255,   0,   0),  // Red
    cv::Scalar(  0, 255,   0),  // Green
    cv::Scalar(  0,   0, 255),  // Blue
    cv::Scalar(255, 255,   0),  // Cyan
    cv::Scalar(255,   0, 255),  // Magenta
    cv::Scalar(  0, 255, 255),  // Yellow
    cv::Scalar(255, 128,   0),  // Orange
    cv::Scalar(128,   0, 128),  // Purple
    cv::Scalar(128, 128,   0),  // Olive
    cv::Scalar(128,   0, 255),  // Violet
    cv::Scalar(  0, 128, 255),  // Sky Blue
    cv::Scalar(255,   0, 128),  // Pink
    cv::Scalar(  0, 128,   0),  // Dark Green
    cv::Scalar(128, 128, 128),  // Gray
    cv::Scalar(255, 255, 255)   // White
};
cv::Rect get_bbox_coordinates(const hailo_bbox_float32_t& bbox, int frame_width, int frame_height) {
    int x1 = static_cast<int>(bbox.x_min * frame_width);
    int y1 = static_cast<int>(bbox.y_min * frame_height);
    int x2 = static_cast<int>(bbox.x_max * frame_width);
    int y2 = static_cast<int>(bbox.y_max * frame_height);
    return cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2));
}
void initialize_class_colors(std::unordered_map<int, cv::Scalar>& class_colors) {
    for (int cls = 0; cls <= 80; ++cls) {
        class_colors[cls] = COLORS[cls % COLORS.size()]; 
    }
}
void draw_label(cv::Mat& frame, const std::string& label, const cv::Point& top_left, const cv::Scalar& color) {
    int baseLine = 0;
    cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    int top = std::max(top_left.y, label_size.height);
    cv::rectangle(frame, cv::Point(top_left.x, top + baseLine), 
                  cv::Point(top_left.x + label_size.width, top - label_size.height), color, cv::FILLED);
    cv::putText(frame, label, cv::Point(top_left.x, top), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
}

void draw_single_bbox(cv::Mat& frame, const NamedBbox& named_bbox, const cv::Scalar& color) {
    auto bbox_rect = get_bbox_coordinates(named_bbox.bbox, frame.cols, frame.rows);
    cv::rectangle(frame, bbox_rect, color, 2);

    std::string score_str = std::to_string(named_bbox.bbox.score * 100).substr(0, 4) + "%";
    std::string label = get_coco_name_from_int(static_cast<int>(named_bbox.class_id)) + " " + score_str;
    draw_label(frame, label, bbox_rect.tl(), color);
}

void draw_bounding_boxes(cv::Mat& frame, const std::vector<NamedBbox>& bboxes) {
    std::unordered_map<int, cv::Scalar> class_colors;
    initialize_class_colors(class_colors);
    for (const auto& named_bbox : bboxes) {
        const auto& color = class_colors[named_bbox.class_id];
        draw_single_bbox(frame, named_bbox, color);
    }
}

    CustomFramedSource* CustomFramedSource::createNew(UsageEnvironment& env, std::string source, std::string hef, int delay) {
        return new CustomFramedSource(env, source, hef,delay);
    }

    u_int8_t CustomFramedSource::std_luminance_qt[64] = {
     16,  11,  10,  16,  24,  40,  51,  61,
     12,  12,  14,  19,  26,  58,  60,  55,
     14,  13,  16,  24,  40,  57,  69,  56,
     14,  17,  22,  29,  51,  87,  80,  62,
     18,  22,  37,  56,  68, 109, 103,  77,
     24,  35,  55, 64,  81, 104, 113,  92,
     49,  64,  78, 87, 103, 121, 120, 101,
     72,  92, 95, 98, 112, 100, 103, 99
    };
    u_int8_t CustomFramedSource::std_chrominance_qt[64] = {
     17,  18,  24,  47,  99,  99,  99,  99,
     18,  21,  26,  66,  99,  99,  99,  99,
     24,  26,  56,  99,  99,  99,  99,  99,
     47,  66,  99,  99,  99,  99,  99,  99,
     99,  99,  99,  99,  99,  99,  99,  99,
     99,  99,  99,  99,  99,  99,  99,  99,
     99,  99,  99,  99,  99,  99,  99,  99,
     99,  99,  99,  99,  99,  99,  99,  99
    };


    CustomFramedSource::CustomFramedSource(UsageEnvironment& env, std::string source, std::string hef, int delay) :  JPEGVideoSource(env),DELAY(delay), cap(source), HEF_FILE(hef) {
        if (!cap.isOpened()) {
            env << "Failed to open camera\n";
        } else {
            // cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
            // cap.set(cv::CAP_PROP_FRAME_HEIGHT, 640);
            // cap.set(cv::CAP_PROP_FPS, 30);
            env << "Camera opened successfully\n";
        }
        this->vdevice = VDevice::create().expect("Failed create vdevice");
        this->infer_model = vdevice->create_infer_model(HEF_FILE).expect("Failed to create infer model");
        this->configured_infer_model = infer_model->configure().expect("Failed to create configured infer model");
   		std::cout << "Restart Interval : " << restartInterval() << std::endl;
	}

    void CustomFramedSource::getImage(size_t& size, std::vector<uchar>* data_to_write) {

        
        cv::Mat original;
        cap >> original;
        cv::Mat frame_rgb;
        cv::cvtColor(original, frame_rgb, cv::COLOR_BGR2RGB);
        cv::Mat frame;
        cv::resize(frame_rgb, frame, cv::Size(640, 640));
        std::vector<uint8_t> data(frame.data, frame.data + frame.total() * frame.elemSize());

        std::shared_ptr<uint8_t> input_buffer;
        auto bindings = configured_infer_model.create_bindings().expect("Failed to create infer bindings");
        for (const auto &input_name : infer_model->get_input_names()) {
            size_t input_frame_size = infer_model->input(input_name)->get_frame_size();
            input_buffer = page_aligned_alloc(input_frame_size);
            auto status = bindings.input(input_name)->set_buffer(MemoryView(input_buffer.get(), input_frame_size));
            if (HAILO_SUCCESS != status) {
                throw hailort_error(status, "Failed to set infer input buffer");
            }

        }
        memcpy( input_buffer.get(), data.data() , data.size());

        std::shared_ptr<uint8_t> output_buffer;
        for (const auto &output_name : infer_model->get_output_names()) {
            size_t output_frame_size = infer_model->output(output_name)->get_frame_size();
            output_buffer = page_aligned_alloc(output_frame_size);
            auto status = bindings.output(output_name)->set_buffer(MemoryView(output_buffer.get(), output_frame_size));
            if (HAILO_SUCCESS != status) {
                throw hailort_error(status, "Failed to set infer output buffer");
            }

        }

        std::cout << "Running inference..." << std::endl;
        // Run the async infer job
        auto job = configured_infer_model.run_async(bindings).expect("Failed to start async infer job");
        auto status = job.wait(std::chrono::milliseconds(DELAY));
        if (HAILO_SUCCESS != status) {
            throw hailort_error(status, "Failed to wait for infer to finish");
        }else {
            std::cout << "Inference succeed" << std::endl;

        }
        auto bboxes = parse_nms_data(output_buffer.get(), 80);
        draw_bounding_boxes(original, bboxes);
        cv::imencode(".jpg", original, *data_to_write);
        size = data_to_write->size();
        this->fFrameSize = std::min(data_to_write->size(), static_cast<size_t>(fMaxSize));
        this->fNumTruncatedBytes = data_to_write->size() - fFrameSize;
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
        std::vector<uchar> jpegData;
        getImage(jpegSize, &jpegData);
        // usleep(33333);
	    std::cout << "May be here" << std::endl;
        if (jpegSize == 0) {
	    std::cout << "Problem point" << std::endl;
           envir().taskScheduler().scheduleDelayedTask(10000, (TaskFunc*)FramedSource::afterGetting, this);
            return;
        }

        if (jpegSize > fMaxSize) {
            jpegSize = fMaxSize; // Обрезаем, если буфер не помещается
        }

        memcpy(fTo, jpegData.data(), fFrameSize);
        fPresentationTime.tv_usec += DELAY*1000;
        afterGetting(this);
    }