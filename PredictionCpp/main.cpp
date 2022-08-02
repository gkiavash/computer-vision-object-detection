#include <string>
#include <vector>
#include <iterator>
#include <regex>
#include <fstream>

#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2/dnn.hpp>

#include "MeanShift.h"


using namespace cv;
using namespace std;
using namespace cv::dnn;

const std::regex comma(",");

vector<string> class_names = {"boat", "no_boat"};


class MyNN{
    public:
        String nn_path;
        Net model;
        vector<string> class_names;
        int size_;

        MyNN(String nn_path, vector<string> class_names, int size_){
            this->nn_path = nn_path;
            this->size_ = size_;
            this->model = readNetFromTensorflow(this->nn_path);
            // this->model = readNet(this->nn_path);
            printf("\nModel loaded successfully\n");

            this->class_names = class_names;
        }

        MyNN(){}

        vector<Mat> make_proposals(Mat image, vector<Rect> rects, bool preview=false){
            vector<Mat> proposals;
            for(int i=0;i<rects.size();i++){
                Mat image_sub = image(rects[i]);
                Mat image_resize;
                resize(image_sub, image_resize, Size(this->size_, this->size_), 0, 0, cv::INTER_AREA);
                proposals.push_back(image_resize);

                if(preview==true){
                    imshow("make_proposals", image_resize);
                    waitKey();
                }
            }
            return proposals;
        }

        vector<Rect> predict(Mat image, vector<Rect> rects, float threshold, bool preview=false, bool save=false, string save_full_path=""){
            vector<Rect> rects_predicted;

            vector<Mat> proposals = this->make_proposals(image, rects, false);
            Mat outputs = this->predict_(proposals);
            for(int i=0;i<outputs.size().height;i++){
                if(outputs.at<float>(i, 0)>=outputs.at<float>(i, 1)){
                    if(outputs.at<float>(i, 0) >= threshold){
                        rects_predicted.push_back(rects[i]);
                        cout<<"\n"<<outputs.at<float>(i, 0);

                        // for preview and save:
                        string out_text = format("%s, %.3f", (class_names[0].c_str()), outputs.at<float>(i, 0));
                        putText(image, out_text, Point(rects[i].x, rects[i].y), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);
                        rectangle(image, rects[i], 255, 2);
                    }
                }
            }
            cout<<"\nNumber of positives: "<<rects_predicted.size();

            if(preview==true){
                Mat image_resize;
                resize(image, image_resize, cv::Size(600, 600), 0, 0, cv::INTER_AREA);
                imshow("predict", image_resize);
                waitKey();
            }
            if(save==true){
                cout<<"\nsaving... to: "<<save_full_path;
                Mat image_resize;
                resize(image, image_resize, cv::Size(600, 600), 0, 0, cv::INTER_AREA);
                imwrite(save_full_path, image_resize);
            }
            return rects_predicted;
        }

        Mat predict_(vector<Mat> proposals){
            Mat blob = blobFromImages(proposals, 0.1, Size(this->size_, this->size_), Scalar(127, 127, 127));
            
            Vec3b intensity = proposals[0].at<Vec3b>(0, 0);
            Vec4b intensity2 = blob.at<Vec4b>(0, 0, 0);

            cout<<"\nM: "<<intensity;
            cout<<"\nM: "<<intensity2;


            this->model.setInput(blob);
            Mat outputs = this->model.forward(); 
            cout<<"\noutputs"<<outputs;
            return outputs;

            // for(int i=0;i<proposals.size();i++){
            //     Mat blob = blobFromImage(proposals[i], 1, Size(224, 224), Scalar(225, 225, 225));
            //     this->model.setInput(blob);
            //     Mat outputs = this->model.forward(); 
            //     cout<<"\noutputs"<<outputs;
            //     Point classIdPoint;
            //     double final_prob;
            //     minMaxLoc(outputs.reshape(1, 1), 0, &final_prob, 0, &classIdPoint);
            //     int label_id = classIdPoint.x; 
            //     string out_text = format("%s, %.3f", (class_names[label_id].c_str()), final_prob);
            //     // put the class name text on top of the image
            //     putText(proposals[i], out_text, Point(25, 50), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);
            //     // string label = gender + ", " + age; // label
            //     // putText(frameFace, label, Point(it->at(0), it->at(1) -20), cv::FONT_HERSHEY_SIMPLEX, 0.9, Scalar(0, 255, 255), 2, cv::LINE_AA);
            //     // imshow("Frame", frameFace);
            //     if(label_id == 0){
            //         imshow("Image", proposals[i]);
            //         waitKey();
            //     }
            // }
        }

        float intersection_over_union(Rect rect1, Rect rect2){
            int rect1_x_min = rect1.x;
            int rect1_x_max = rect1.x + rect1.width;
            int rect1_y_min = rect1.y;
            int rect1_y_max = rect1.y + rect1.height;

            int rect2_x_min = rect2.x;
            int rect2_x_max = rect2.x + rect2.width;
            int rect2_y_min = rect2.y;
            int rect2_y_max = rect2.y + rect2.height;

            if(rect1_x_max < rect2_x_min || rect2_x_max < rect1_x_min || rect1_y_max < rect2_y_min || rect2_y_max < rect1_y_min){
                return 0;
            }else{
                float x1_inner = max(rect1_x_min, rect2_x_min);
                float y1_inner = max(rect1_y_min, rect2_y_min);
                float x2_inner = min(rect1_x_max, rect2_x_max);
                float y2_inner = min(rect1_y_max, rect2_y_max);

                float inner_area = (y2_inner - y1_inner) * (x2_inner - x1_inner);

                float rect1_area = (rect1_y_max - rect1_y_min) * (rect1_x_max - rect1_x_min);
                float rect2_area = (rect2_y_max - rect2_y_min) * (rect2_x_max - rect2_x_min);
                float union_area = rect1_area + rect2_area - inner_area;
                // cout<<"rect1_area:"<<rect1_area<<"rect2_area:"<<rect2_area;
                // cout<<"inner_area:"<<inner_area<<"union_area:"<<union_area;
                return inner_area / union_area;
            }
        }

        float evaluate(vector<Rect> rects_ground_truth, vector<Rect> rects_predicted){
            float eval_sum = 0;
            float eval_num = 0;

            for(int i=0;i<rects_predicted.size();i++){
                float eval_max = 0;
                for(int i=0;i<rects_predicted.size();i++){
                    float eval = this->intersection_over_union(rects_ground_truth[i], rects_predicted[i]);
                    if(eval>eval_max){
                        eval_max = eval;
                    }
                }
                eval_sum += eval_max;
                eval_num += 1;
            }
            return eval_sum/eval_num;
        }

};

class BoatDetector{
    public:
        String image_path;
        Mat image;
        MyNN mynn;

        BoatDetector(string image_path,  MyNN mynn){
            this->image_path = image_path;
            this->mynn = mynn;
            this->image = imread(this->image_path);
        }

        Mat segment_meanshift(Mat image, bool preview=false){
            cvtColor(image, image, COLOR_RGB2Lab);

            MeanShift MSProc(8, 16);
            // MSProc.MSFiltering(image);
            // Segmentation Process include Filtering Process (Region Growing)
            MSProc.MSSegmentation(image);
            
            cout<<"\nthe Spatial Bandwith is "<<MSProc.hs;
            cout<<"\nthe Color Bandwith is "<<MSProc.hr;

            cvtColor(image, image, COLOR_Lab2RGB);
            if(preview==true){
                imshow("MS Picture", image);
                waitKey();
            }
            return image;
        }

        vector<Rect> find_countours(Mat image, bool preview=false){
            vector<Rect> rects;

            if(image.depth() != CV_8U){
                Mat image_;
                image.convertTo(image_, CV_8U);
                image = image_;
            }

            RNG rng(12345);
            vector<vector<Point>> contours;
            vector<Vec4i> hierarchy;
            findContours(image, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
            
            cout<<"\ncontours size(): "<<contours.size();
            
            Mat drawing = Mat::zeros(image.size(), CV_8UC3);
            for(size_t i = 0; i< contours.size(); i++){
                Scalar color = Scalar(rng.uniform(0, 256), rng.uniform(0,256), rng.uniform(0,256));
                drawContours(drawing, contours, (int)i, color, 2, LINE_8, hierarchy, 0);

                if(contours[i].size()>40){

                    int x_min=image.size().width, y_min=image.size().height, x_max=0, y_max=0;
                    for(int j=0;j<contours[i].size();j++){
                        if(contours[i][j].x <= x_min) x_min = contours[i][j].x;
                        if(contours[i][j].x >= x_max) x_max = contours[i][j].x;
                        if(contours[i][j].y <= y_min) y_min = contours[i][j].y;
                        if(contours[i][j].y >= y_max) y_max = contours[i][j].y;
                    }
                    Point pt1(x_min, y_min);
                    Point pt2(x_max, y_max);
                    rects.push_back(Rect(x_min, y_min, x_max-x_min, y_max-y_min));
                    rectangle(drawing, pt1, pt2, cv::Scalar(0, 255, 0));
                }
                // cout<<contours[i];
                // cout<<"each: "<<contours[i].size();
            }
            if(preview==true){
                imshow("Contours", drawing);
                waitKey(0);
            }
            
            return rects;
        }

        Mat edge_canny(Mat image_main, bool preview=false){
            Mat image_canny;
            Canny(image_main, image_canny, 100, 200,3);

            Mat dist;
            distanceTransform(image_canny, dist, DIST_L2, 3);
            normalize(dist, dist, 0, 1.0, NORM_MINMAX);
            threshold(dist, dist, 0.4, 1.0, THRESH_BINARY);
            Mat kernel1 = Mat::ones(3, 3, CV_8U);

            dilate(dist, dist, kernel1);
            if(preview==true){
                imshow("Image_canny", image_canny);
                imshow("dilate", dist);
                waitKey();
            }
            return image_canny;
        }

};

void test_intersection_over_union(){
    MyNN mynn = MyNN();
    float t1 = mynn.intersection_over_union(Rect(0,0,2,2), Rect(1,1,2,2));
    float t2 = mynn.intersection_over_union(Rect(0,0,2,2), Rect(3,3,4,4));
    cout<<"t1: "<<t1<<" t2: "<<t2; // assert 0.1428, 0
}

void test_make_proposals(){
    cout<<"\ntest_make_proposals\n";
    MyNN mynn = MyNN();
    BoatDetector b = BoatDetector("", mynn);
    vector<Rect> rects;
    rects.push_back(Rect(0,0,20,20));
    rects.push_back(Rect(10,10,50,90));

    Mat image = imread("images/20130305_135514_86270.jpg");

    vector<Mat> t1 = mynn.make_proposals(image, rects);
    for (int i=0 ; i < t1.size(); i++) {
        imshow("img", t1[i]);
        waitKey();
    }
}

void test_predict(){
    cout<<"\test_predict\n";
    String nn_test_path = "frozen_models/simple_frozen_graph.pb";
    MyNN mynn = MyNN(nn_test_path, class_names, 224);
    BoatDetector b = BoatDetector("", mynn);
    vector<Rect> rects;
    rects.push_back(Rect(232, 60, 313, 80));
    rects.push_back(Rect(10,10,50,90));

    Mat image = imread("images/20130305_135514_86270.jpg");

    vector<Mat> proposals = mynn.make_proposals(image ,rects);
    mynn.predict_(proposals);
}


vector<vector<string>> read_record(string input_path_csv){
    std::ifstream mesh(input_path_csv);

    vector<vector<string>> point_coordinates;
    string line{};

    while (mesh && getline(mesh, line)) {
        vector<string> row{ sregex_token_iterator(line.begin(),line.end(),comma,-1), sregex_token_iterator() };
        point_coordinates.push_back(row);
    }
    return point_coordinates;
}


vector<Rect> get_rects_ground_truth(String annotation_path){
    vector<Rect> rects_ground_truth;

    vector<vector<string>> data = read_record(annotation_path);
    cout<<"\ndata.size(): "<<data.size();
    for(int i=1;i<data.size();i++){
        // cout<<"lksdjflfshgflkjl:  "<<data[i][4];
        rects_ground_truth.push_back(Rect(
            stoi(data[i][4]), 
            stoi(data[i][5]), 
            stoi(data[i][6]), 
            stoi(data[i][7])
        ));
    }
    cout<<"\nects_ground_truth.size(): "<<rects_ground_truth[0];
    return rects_ground_truth;
}


int main(int argc, char** argv ) {
    // test_intersection_over_union();
    // test_make_proposals();
    // test_predict();

    vector<String> annotations_paths = {
        // "annotations/20130305_135514_86270.csv",
        "annotations/07.csv"
        "annotations/06.csv"
        "annotations/05.csv"
        "annotations/20130305_085139_39961.csv"
    };

    // vector<String> image_paths = {
    //     // "images/20130305_135514_86270.jpg"
    //     "images/07.png",
    //     "images/06.png",
    //     "images/05.png",
    //     "images/20130305_085139_39961.jpg"
    // };

    string final_nn_path = "frozen_models/model_cnn.pb";
    MyNN mynn = MyNN(final_nn_path, class_names, 224);

    std::string folder("images/");
    std::vector<cv::String> image_paths;
    cv::glob(folder, image_paths, false);
    for (int i=0 ; i < image_paths.size(); i++) {

    // for (int i=0 ; i < image_paths.size(); i++) {
        cout<<"Loading "<<image_paths[i];
        BoatDetector boat_detector = BoatDetector(image_paths[i], mynn);

        // Mat image_blurred;
        // image_blurred = boat_detector.image;
        // GaussianBlur(boat_detector.image, image_blurred, Size(13, 13), 2.4);
        // Mat image_segment = boat_detector.segment_meanshift(image_blurred, true);


        vector<Rect> rects_;
        vector<int> kernels = {31, 45};
        for(int j=0;j<kernels.size();j++){
            Mat image_blurred;
            GaussianBlur(boat_detector.image, image_blurred, Size(kernels[j], kernels[j]), 2.4);
            
            // Mat image_canny = boat_detector.edge_canny(image_blurred, false);
            // vector<Rect> rects = boat_detector.find_countours(image_canny, true);
            // rects_.insert(rects_.end(), rects.begin(), rects.end());

            Mat image_segment = boat_detector.segment_meanshift(image_blurred, false);
            Mat image_canny = boat_detector.edge_canny(image_segment, false);

            // cvtColor(image_segment, image_segment, COLOR_RGB2GRAY);

            // imshow("image_segment", image_segment);
            // waitKey();
            vector<Rect> rects = boat_detector.find_countours(image_canny, false);
            rects_.insert(rects_.end(), rects.begin(), rects.end());
            
            cout<<"\nrects.size(): "<<rects.size();
            cout<<"\nrects_.size(): "<<rects_.size();

        }
        vector<Rect> rects_predict = boat_detector.mynn.predict(boat_detector.image, rects_, 0.97, false, true, "results/"+image_paths[i]);
        cout<<"\nrects_predict.size(): "<<rects_predict.size();

        // vector<Rect> rects_ground_truth = get_rects_ground_truth(annotations_paths[i]);
        // cout<<"\nrects_ground_truth.size(): "<<rects_ground_truth.size();

        // float res = boat_detector.mynn.evaluate(rects_ground_truth, rects_predict);
        // cout<<"\neval: "<<res;
    }

    return 0;
}
