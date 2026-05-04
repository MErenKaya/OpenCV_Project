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

    const float confThreshold = 0.45f; // güven eşiği (zayıf tahminleri sil)
    const float nmsThreshold = 0.40f;  // çakışan kutuları temizleme oranı
    const int inputSize = 640;         // YOLO giriş görüntü boyutu

    Mat frame;

    while (true) 
    {
        cap >> frame;
        if (frame.empty()) break;

        Mat blob;

        blobFromImage(frame, blob, 1.0 / 255.0, Size(inputSize, inputSize), Scalar(), true, false); // görüntüyü YOLO formatına çevir

        net.setInput(blob);          // modele görüntüyü ver

        vector<Mat> outputs;         // model çıktıları 100*100 lük bir fotoyu 2*2 yapmak gibi
        net.forward(outputs, net.getUnconnectedOutLayersNames()); // forward pass hadi hesapla

        Mat output = outputs[0];     // YOLO çıktısı (raw tensor)

        // YOLOv8 çıktısını tabloya çevir
        Mat rows84 = Mat(output.size[1], output.size[2], CV_32F, output.ptr<float>());
        Mat outMat = rows84.t();     // transpose (8400 x 84 hale getir)

        vector<int> classIds;        // tespit edilen sınıf ID'leri
        vector<float> confidences;   // güven skorları
        vector<Rect> boxes;          // bounding box'lar

        float x_factor = (float)frame.cols / inputSize; // genişlik ölçek
        float y_factor = (float)frame.rows / inputSize; // yükseklik ölçek

        for (int i = 0; i < outMat.rows; i++) // işe yararmı yaramaz mı
        {
            float* data = (float*)outMat.row(i).data; //tek satır verisi

            Mat scores(1, classes.size(), CV_32F, data + 4); //classların skorları
            Point classIdPoint;
            double maxClassScore;

            minMaxLoc(scores, 0, &maxClassScore, 0, &classIdPoint); // en iyi sınıf

            if (maxClassScore > confThreshold) // güven yeterliyse
            {
                float x = data[0]; // merkez x
                float y = data[1]; // merkez y
                float w = data[2]; // genişlik
                float h = data[3]; // yükseklik

                int left = (int)((x - 0.5 * w) * x_factor); // sol üst x
                int top = (int)((y - 0.5 * h) * y_factor);  // sol üst y
                int width = (int)(w * x_factor);            // gerçek genişlik
                int height = (int)(h * y_factor);           // gerçek yükseklik

                classIds.push_back(classIdPoint.x);        // sınıf ID kaydet
                confidences.push_back((float)maxClassScore); // güven kaydet
                boxes.push_back(Rect(left, top, width, height)); // kutu ekle
            }
        }

        vector<int> indices; // NMS sonucu indexler

        NMSBoxes(boxes, confidences, confThreshold,
            nmsThreshold, indices); // fazla kutuları sil

        for (int i : indices) // kalan kutular
        {
            Rect box = boxes[i]; // kutu al

            box &= Rect(0, 0, frame.cols, frame.rows); // ekran dışına taşmayı engelle

            rectangle(frame, box, Scalar(0, 255, 0), 2); // yeşil kutu çiz

            string label = classes[classIds[i]] + " " +
                to_string((int)(confidences[i] * 100)) + "%"; // etiket

            putText(frame, label,
                Point(box.x, box.y),
                FONT_HERSHEY_SIMPLEX,
                0.5,
                Scalar(0, 0, 0), 1); // yazıyı çiz
        }

        imshow("YOLOv8 + OpenCV DNN", frame); // görüntüyü göster

        if (waitKey(1) == 27) break;
    }

    cap.release();
    destroyAllWindows();

    return 0;
}