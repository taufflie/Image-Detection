
#include "opencv2/core.hpp"                // OpenCV core routines
#include "opencv2/imgcodecs.hpp"
#include <opencv2/videoio.hpp>             // OpenCV video routines
#include "opencv2/highgui.hpp"             // OpenCV GUI routines
#include "opencv2/imgproc.hpp"             // OpenCV Image Processing routines
#include "opencv2/features2d.hpp"
#include "opencv2/calib3d.hpp"
#include <iostream>
#include <string>

using namespace std;
using namespace cv;


//---- Global variables
Mat frame, frame_gray, frame_out;
Mat *the_frame;
Mat img, img1, img2, warped_image;
Mat H;
bool verbosity = true;

// Camera video stream
VideoCapture cap;
bool isGrayCamera = false;

// Number of features
const char* trackbarNumberFeatures_name = "N° of features";  // Trackbar name
int trackbarNumberFeatures_max_value = 500;          // Trackbar max value
int trackbarNumberFeatures_min_value = 10;           // Trackbar min value
int trackbarNumberFeatures_value;                   // Trackbar current value
int numberFeatures;                               // Number of features
int numberFeatures_default_value = 250;

// Ratio threshold
const char* trackbarRatioThresh_name = "100*Ratio Threshold";  // Trackbar name
int trackbarRatioThresh_max_value = 100;          // Trackbar max value
int trackbarRatioThresh_min_value = 1;           // Trackbar min value
int trackbarRatioThresh_value;                   // Trackbar current value
double ratio_thresh;                               // ratio threshold
double ratio_thresh_default_value = 70.0;

// Threshold type
const char* trackbarType_name = "Global Approch";    // Trackbar name
int trackbarType_max_value = 1;             // = THRESH_TOZERO (Upper threshold)
int trackbarType_min_value = 0;      // Trackbar min value
int trackbarType_value;                    // Trackbar current value
bool type;                 // Vector norm for contrast                                    
bool type_default_value = true;

// Number of Neighbours
const char* trackbarNumberNeighbours_name = "N° of neighbours";  // Trackbar name
int trackbarNumberNeighbours_max_value = 20;          // Trackbar max value
int trackbarNumberNeighbours_min_value = 1;           // Trackbar min value
int trackbarNumberNeighbours_value;                   // Trackbar current value
int numberNeighbours;                               // Number of neighbours
int numberNeighbours_default_value = 2;

// Windows
String window_out_name;

void create_GUI();
void grab_preprocess();
void initialize_stream();
void process_display();
void process_display_callback(int, void*);
int  usage(char*);

int main(int argc, char* argv[])
{
    int   i;
    String filename1;
    String filename2 ="";

    // Command line parsing
    if (argc > 1)
        for (i = 1; i < argc; i++) {
            if (string(argv[i]) == "-i") {          // Input image
                if (++i > argc)
                    usage(argv[0]);
                filename1 = filename2;
                filename2 = string(argv[i]);
                
            }
            else {
                cout << "! Invalid program option " << argv[i] << endl;
                usage(argv[0]);
            }
        }
    else {
        cout << "! Missing argument" << endl;
        usage(argv[0]);
    }

    img = imread(filename1);
    cvtColor(img, img1, COLOR_BGR2GRAY);
    img2 = imread(filename2);

    // Open camera & gets its features
    initialize_stream();

    // Grab & preprocess frame
    grab_preprocess();

    // Create GUI
    create_GUI();

    // Invoke callback routine to initialize and show transformed image
    process_display_callback(trackbarNumberFeatures_value, 0);

    // Perform features matching from video stream
    for (;;) {
        // Grab & preprocess frame
        grab_preprocess();

        // Processing & Visualization
        process_display();

        // Listen to next event - Exit if key pressed
        if (waitKey(5) >= 0)
            break;
    }

    // Destroy windows
    destroyAllWindows();

    // The camera will be released automatically in VideoCapture destructor
    return 0;
}

void create_GUI()
{
    String window_name_prefix = "OpenCV | Features extraction ";

    // Create windows for :
    // - original and transformed images
    window_out_name = window_name_prefix;
    namedWindow(window_out_name, WINDOW_AUTOSIZE);

    // Create trackbars 
    // - for number of features
    trackbarNumberFeatures_value = (int)numberFeatures;
    createTrackbar(trackbarNumberFeatures_name, window_out_name,
        nullptr, trackbarNumberFeatures_max_value,
        (TrackbarCallback)process_display_callback);
    if (1 == 2) { // For OpenCV 4.5.3, setTrackbarMin() blocks the trackbar to its minimal value  
        setTrackbarMin(trackbarNumberFeatures_name, window_out_name,
            trackbarNumberFeatures_min_value);
    }

    // - for ratio threshold
    trackbarRatioThresh_value = (int)ratio_thresh;
    createTrackbar(trackbarRatioThresh_name, window_out_name,
        nullptr, trackbarRatioThresh_max_value,
        (TrackbarCallback)process_display_callback);
    if (1 == 2) { // For OpenCV 4.5.3, setTrackbarMin() blocks the trackbar to its minimal value  
        setTrackbarMin(trackbarRatioThresh_name, window_out_name,
            trackbarRatioThresh_min_value);
    }

    // - for threshold type
    createTrackbar(trackbarType_name, window_out_name,
        nullptr, trackbarType_max_value,
        (TrackbarCallback)process_display_callback);
    if (1 == 2) { // For OpenCV 4.5.3, setTrackbarMin() blocks the trackbar to its minimal value  
        setTrackbarMin(trackbarType_name, window_out_name,
            trackbarType_min_value);
    }

    // - for number of Neighbours
    trackbarNumberNeighbours_value = (int)numberNeighbours;
    createTrackbar(trackbarNumberNeighbours_name, window_out_name,
        nullptr, trackbarNumberNeighbours_max_value,
        (TrackbarCallback)process_display_callback);
    if (1 == 2) { // For OpenCV 4.5.3, setTrackbarMin() blocks the trackbar to its minimal value  
        setTrackbarMin(trackbarNumberNeighbours_name, window_out_name,
            trackbarNumberNeighbours_min_value);
    }
    // Set trackbars default positions

    trackbarNumberFeatures_value = numberFeatures_default_value;
    setTrackbarPos(trackbarNumberFeatures_name, window_out_name,
        trackbarNumberFeatures_value);

    trackbarRatioThresh_value = ratio_thresh_default_value;
    setTrackbarPos(trackbarRatioThresh_name, window_out_name,
        trackbarRatioThresh_value);

    trackbarType_value = type_default_value;
    setTrackbarPos(trackbarType_name, window_out_name,
        trackbarType_value);

    trackbarNumberNeighbours_value = numberNeighbours_default_value;
    setTrackbarPos(trackbarNumberNeighbours_name, window_out_name,
        trackbarNumberNeighbours_value);
}

void initialize_stream()
{
    // Open default camera
    cap.open(0);

    // Check if camera opening is successful
    if (!cap.isOpened())
        exit(EXIT_FAILURE);

    // Get camera properties and initialize various variables
    // - Check for color camera
    cap >> frame;                                 // Get frame
    if (frame.channels() == 1)
        isGrayCamera = true;
}

void grab_preprocess()
{
    // Get a new frame from camera
    cap >> frame;

    // Convert frame to graylevel if appropriate
    if (isGrayCamera)
        the_frame = &frame;
    else {
        cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
        the_frame = &frame_gray;
    }
}

void process_display_callback(int value1, void* userdata)
{
    // Get trackbars positions
    trackbarNumberFeatures_value =
        getTrackbarPos(trackbarNumberFeatures_name, window_out_name);

    trackbarRatioThresh_value =
        getTrackbarPos(trackbarRatioThresh_name, window_out_name);

    trackbarType_value =
        getTrackbarPos(trackbarType_name, window_out_name);

    trackbarNumberNeighbours_value =
        getTrackbarPos(trackbarNumberNeighbours_name, window_out_name);

    // Safety check for OpenCV 4.5.3 which does not manage trackbar min value properly
    if (CV_VERSION == "4.5.3") {
        if (trackbarNumberFeatures_value < trackbarNumberFeatures_min_value) {
            trackbarNumberFeatures_value = trackbarNumberFeatures_min_value;
            setTrackbarPos(trackbarNumberFeatures_name, window_out_name,
                trackbarNumberFeatures_value);
        }

        if (trackbarRatioThresh_value < trackbarRatioThresh_min_value) {
            trackbarRatioThresh_value = trackbarRatioThresh_min_value;
            setTrackbarPos(trackbarRatioThresh_name, window_out_name,
                trackbarRatioThresh_value);
        }

        if (trackbarType_value < trackbarType_min_value) {
            trackbarType_value = trackbarType_min_value;
            setTrackbarPos(trackbarType_name, window_out_name,
                trackbarType_value);
        }

        if (trackbarNumberNeighbours_value < trackbarNumberNeighbours_min_value) {
            trackbarNumberNeighbours_value = trackbarNumberNeighbours_min_value;
            setTrackbarPos(trackbarNumberNeighbours_name, window_out_name,
                trackbarNumberNeighbours_value);
        }
    }

    // Set hyperparameter values from trackbars
    numberFeatures = (int)trackbarNumberFeatures_value;
    ratio_thresh = 0.01*trackbarRatioThresh_value;
    type = ((trackbarType_value == 1) ? true : false);
    numberNeighbours = (int)trackbarNumberNeighbours_value;

    if (verbosity == true) {
        cout << "Number of Features = " << numberFeatures;
        if (type)
            cout << " | Global Approach";
        else
            cout << " | Local Approach";
        cout << " | Number of Neighbours = " << numberNeighbours;
        cout << " | Ratio Threshold = " << ratio_thresh << "\n";
    }

    // Processing & Visualization
    process_display();
}

void process_display()
{
    //-- Step 1: Detect the keypoints using SIFT Detector, compute the descriptors

    Ptr<SIFT> detector = SIFT::create(numberFeatures);
    std::vector<KeyPoint> keypoints1, keypoints2;
    Mat descriptors1, descriptors2;
    detector->detectAndCompute(img1, noArray(), keypoints1, descriptors1);
    detector->detectAndCompute(*the_frame, noArray(), keypoints2, descriptors2);

    //-- Step 2: Matching descriptor vectors with a knn matcher

    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE);
    std::vector< std::vector<DMatch> > knn_matches;
    matcher->knnMatch(descriptors1, descriptors2, knn_matches, numberNeighbours);

    //-- Step 3: Match filtering using the Lowe's ratio test
    std::vector<DMatch> good_matches;

    if (type)
    {
        for (size_t i = 0; i < knn_matches.size(); i++)
        {
            float distance_max = 0;
            for (size_t j = 0; j < knn_matches[i].size(); j++)
            {
                if (distance_max < knn_matches[i][j].distance)
                {
                    distance_max = knn_matches[i][j].distance;
                }
            }

            if (knn_matches[i][0].distance < ratio_thresh * distance_max)
            {
                good_matches.push_back(knn_matches[i][0]);
            }
        }
    }
    
    else
    {
        for (size_t i = 0; i < knn_matches.size(); i++)
        {
            if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
            {
                good_matches.push_back(knn_matches[i][0]);
            }
        }
    }

    //-- Draw matches

    Mat img_matches;
    drawMatches(img, keypoints1, frame, keypoints2, good_matches, img_matches, Scalar::all(-1),
        Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    
    //-- Step 4: Alignment
    std::vector<Point2f> poster;
    std::vector<Point2f> picture;

    for (size_t i = 0; i < good_matches.size(); i++)
    {
        //-- Get the keypoints from the good matches
        poster.push_back(keypoints1[good_matches[i].queryIdx].pt);
        picture.push_back(keypoints2[good_matches[i].trainIdx].pt);
    }

    if (poster.size() >= 4 && picture.size() >= 4)
    {
        Mat H = findHomography(poster, picture, RANSAC);

        //-- Get the corners from the image_1 ( the object to be "detected" )
        std::vector<Point2f> poster_corners(4);
        poster_corners[0] = Point2f(0, 0);
        poster_corners[1] = Point2f((float)img1.cols, 0);
        poster_corners[2] = Point2f((float)img1.cols, (float)img1.rows);
        poster_corners[3] = Point2f(0, (float)img1.rows);

        std::vector<Point2f> new_corners(4);
        new_corners[0] = Point2f(0, 0);
        new_corners[1] = Point2f((float)img2.cols, 0);
        new_corners[2] = Point2f((float)img2.cols, (float)img2.rows);
        new_corners[3] = Point2f(0, (float)img2.rows);

        std::vector<Point2f> scene_corners(4);
        if (!H.empty()) 
        {
            perspectiveTransform(poster_corners, scene_corners, H);

            //-- Draw lines between the corners (the mapped object in the scene - image_2 )
            line(img_matches, scene_corners[0] + Point2f((float)img1.cols, 0),
                scene_corners[1] + Point2f((float)img1.cols, 0), Scalar(0, 255, 0), 4);
            line(img_matches, scene_corners[1] + Point2f((float)img1.cols, 0),
                scene_corners[2] + Point2f((float)img1.cols, 0), Scalar(0, 255, 0), 4);
            line(img_matches, scene_corners[2] + Point2f((float)img1.cols, 0),
                scene_corners[3] + Point2f((float)img1.cols, 0), Scalar(0, 255, 0), 4);
            line(img_matches, scene_corners[3] + Point2f((float)img1.cols, 0),
                scene_corners[0] + Point2f((float)img1.cols, 0), Scalar(0, 255, 0), 4);

            Size size = Size(cvRound(scene_corners[2].x), cvRound(scene_corners[2].y));
            Mat M = getPerspectiveTransform(new_corners, scene_corners);

            warpPerspective(img2, warped_image, H, size ); // do perspective transformation  
            imshow("Warped Image", warped_image);
        }

    }   
    
    //-- Show detected matches

    imshow(window_out_name, img_matches);

}

int usage(char* prgname)
{
    cout << "Usage: " << prgname << " ";
    cout << "[-i {image file}]" << endl;

    exit(-1);
}

