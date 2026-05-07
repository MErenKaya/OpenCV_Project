#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <fstream>

using namespace cv;
using namespace dnn;
using namespace std;

int main()
{
    vector<string> classes;

    ifstream ifs("coco.names");
    string line;

    while (getline(ifs, line))
        classes.push_back(line);

    Net net = readNet("yolov8n.onnx");

    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(DNN_TARGET_CPU);

    VideoCapture cap(0);

    if (!cap.isOpened())
    {
        cerr << "Kamera acilamadi!" << endl;
        return -1;
    }

    const float confThreshold = 0.45f;
    const float nmsThreshold = 0.40f;
    const int inputSize = 640;

    Mat frame;

    while (true) 
    {
        cap >> frame;
        if (frame.empty()) break;

        Mat blob;

        blobFromImage(frame, blob, 1.0 / 255.0, Size(inputSize, inputSize), Scalar(), true, false);

        net.setInput(blob);

        vector<Mat> outputs;
        net.forward(outputs, net.getUnconnectedOutLayersNames());

        Mat output = outputs[0];


        Mat rows84 = Mat(output.size[1], output.size[2], CV_32F, output.ptr<float>());
        Mat outMat = rows84.t();

        vector<int> classIds;
        vector<float> confidences;
        vector<Rect> boxes;

        float x_factor = (float)frame.cols / inputSize;
        float y_factor = (float)frame.rows / inputSize;

        for (int i = 0; i < outMat.rows; i++)
        {
            float* data = (float*)outMat.row(i).data;

            Mat scores(1, classes.size(), CV_32F, data + 4);
            Point classIdPoint;
            double maxClassScore;

            minMaxLoc(scores, 0, &maxClassScore, 0, &classIdPoint);

            if (maxClassScore > confThreshold)
            {
                float x = data[0];
                float y = data[1];
                float w = data[2];
                float h = data[3];

                int left = (int)((x - 0.5 * w) * x_factor);
                int top = (int)((y - 0.5 * h) * y_factor);
                int width = (int)(w * x_factor);
                int height = (int)(h * y_factor);

                classIds.push_back(classIdPoint.x);
                confidences.push_back((float)maxClassScore);
                boxes.push_back(Rect(left, top, width, height));
            }
        }

        vector<int> indices;

        NMSBoxes(boxes, confidences, confThreshold,
            nmsThreshold, indices);

        for (int i : indices)
        {
            Rect box = boxes[i];

            box &= Rect(0, 0, frame.cols, frame.rows);

            rectangle(frame, box, Scalar(0, 255, 0), 2);

            string label = classes[classIds[i]] + " " +
                to_string((int)(confidences[i] * 100)) + "%";

            putText(frame, label,
                Point(box.x, box.y),
                FONT_HERSHEY_SIMPLEX,
                0.5,
                Scalar(0, 0, 0), 1);
        }

        imshow("YOLOv8 + OpenCV DNN", frame);

        if (waitKey(1) == 27) break;
    }

    cap.release();
    destroyAllWindows();

    return 0;
}