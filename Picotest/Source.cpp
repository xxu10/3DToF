#include <fstream>
#include <iostream>
#include <cstdlib>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/shape_utils.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "PicoZense_api.h"

using namespace std;
using namespace cv;
using namespace cv::dnn;

const size_t inWidth = 300;
const size_t inHeight = 300;
const float WHRatio = inWidth / (float)inHeight;
const float inScaleFactor = 0.007843f;
const float meanVal = 127.5;
const char* classNames[] = { "background",
							 "aeroplane", "bicycle", "bird", "boat",
							 "bottle", "bus", "car", "cat", "chair",
							 "cow", "diningtable", "dog", "horse",
							 "motorbike", "person", "pottedplant",
							 "sheep", "sofa", "train", "tvmonitor" };



int main(int argc, char* argv[])
{
	Net net = readNetFromCaffe("MobileNetSSD_deploy.prototxt", "MobileNetSSD_deploy.caffemodel");
	if (net.empty())
	{
		cerr << "Can't load network by using the following files: " << endl;
		exit(-1);
	}
    
	PsReturnStatus status;
	int32_t deviceIndex = 0;
	int32_t deviceCount = 0;
	uint32_t slope = 1450;
	uint32_t wdrSlope = 4400;
	PsDepthRange depthRange = PsMidRange;
	int32_t dataMode = PsDepthAndIR_30;

	status = PsInitialize();
	if (status != PsReturnStatus::PsRetOK)
	{
		cout << "PsInitialize failed!" << endl;
		system("pause");
		return -1;
	}

	status = PsGetDeviceCount(&deviceCount);
	if (status != PsReturnStatus::PsRetOK)
	{
		cout << "PsGetDeviceCount failed!" << endl;
		system("pause");
		return -1;
	}
	cout << "Get device count: " << deviceCount << endl;

	//Set the Depth Range to Near through PsSetDepthRange interface
	status = PsSetDepthRange(deviceIndex, depthRange);
	if (status != PsReturnStatus::PsRetOK)
		cout << "PsSetDepthRange failed!" << endl;
	else
		cout << "Set Depth Range to Far" << endl;

	status = PsOpenDevice(deviceIndex);
	if (status != PsReturnStatus::PsRetOK)
	{
		cout << "OpenDevice failed!" << endl;
		system("pause");
		return -1;
	}

	//Set PixelFormat as PsPixelFormatBGR888 for opencv display
	PsSetColorPixelFormat(deviceIndex, PsPixelFormatBGR888);

	//Set to DepthAndRGB_30 mode
	PsSetDataMode(deviceIndex, (PsDataMode)dataMode);

	PsFrameMode depthFrameMode;
	PsFrameMode irFrameMode;
	status = PsGetFrameMode(deviceIndex, PsDepthFrame, &depthFrameMode);
	cout << "Get Depth Frame mode status: " << status << endl;
	cout << "depthFrameMode.pixelFormat: " << depthFrameMode.pixelFormat << endl;
	cout << "depthFrameMode.resolutionWidth: " << depthFrameMode.resolutionWidth << endl;
	cout << "depthFrameMode.resolutionHeight: " << depthFrameMode.resolutionHeight << endl;
	cout << "depthFrameMode.fps: " << depthFrameMode.fps << endl;

	status = PsGetFrameMode(deviceIndex, PsIRFrame, &irFrameMode);
	cout << "Get IR Frame mode status: " << status << endl;
	cout << "irFrameMode.pixelFormat: " << irFrameMode.pixelFormat << endl;
	cout << "irFrameMode.resolutionWidth: " << irFrameMode.resolutionWidth << endl;
	cout << "irFrameMode.resolutionHeight: " << irFrameMode.resolutionHeight << endl;
	cout << "irFrameMode.fps: " << irFrameMode.fps << endl;

	PsCameraParameters cameraParameters;
	status = PsGetCameraParameters(deviceIndex, PsDepthSensor, &cameraParameters);

	cout << "Get PsGetCameraParameters status: " << status << endl;
	cout << "Depth Camera Intinsic: " << endl;
	cout << "Fx: " << cameraParameters.fx << endl;
	cout << "Cx: " << cameraParameters.cx << endl;
	cout << "Fy: " << cameraParameters.fy << endl;
	cout << "Cy: " << cameraParameters.cy << endl;
	cout << "Depth Distortion Coefficient: " << endl;
	cout << "K1: " << cameraParameters.k1 << endl;
	cout << "K2: " << cameraParameters.k2 << endl;
	cout << "P1: " << cameraParameters.p1 << endl;
	cout << "P2: " << cameraParameters.p2 << endl;
	cout << "K3: " << cameraParameters.k3 << endl;
	cout << "K4: " << cameraParameters.k4 << endl;
	cout << "K5: " << cameraParameters.k5 << endl;
	cout << "K6: " << cameraParameters.k6 << endl;

	status = PsGetCameraParameters(deviceIndex, PsRgbSensor, &cameraParameters);

	cout << "Get PsGetCameraParameters status: " << status << endl;
	cout << "RGB Camera Intinsic: " << endl;
	cout << "Fx: " << cameraParameters.fx << endl;
	cout << "Cx: " << cameraParameters.cx << endl;
	cout << "Fy: " << cameraParameters.fy << endl;
	cout << "Cy: " << cameraParameters.cy << endl;
	cout << "RGB Distortion Coefficient: " << endl;
	cout << "K1: " << cameraParameters.k1 << endl;
	cout << "K2: " << cameraParameters.k2 << endl;
	cout << "K3: " << cameraParameters.k3 << endl;
	cout << "P1: " << cameraParameters.p1 << endl;
	cout << "P2: " << cameraParameters.p2 << endl;

	PsCameraExtrinsicParameters CameraExtrinsicParameters;
	status = PsGetCameraExtrinsicParameters(deviceIndex, &CameraExtrinsicParameters);

	cout << "Get PsGetCameraExtrinsicParameters status: " << status << endl;
	cout << "Camera rotation: " << endl;
	cout << CameraExtrinsicParameters.rotation[0] << " "
		<< CameraExtrinsicParameters.rotation[1] << " "
		<< CameraExtrinsicParameters.rotation[2] << " "
		<< CameraExtrinsicParameters.rotation[3] << " "
		<< CameraExtrinsicParameters.rotation[4] << " "
		<< CameraExtrinsicParameters.rotation[5] << " "
		<< CameraExtrinsicParameters.rotation[6] << " "
		<< CameraExtrinsicParameters.rotation[7] << " "
		<< CameraExtrinsicParameters.rotation[8] << " "
		<< endl;

	cout << "Camera transfer: " << endl;
	cout << CameraExtrinsicParameters.translation[0] << " "
		<< CameraExtrinsicParameters.translation[1] << " "
		<< CameraExtrinsicParameters.translation[2] << " " << endl;

	cv::Mat imageMat;
	const string irImageWindow = "IR Image";
	const string rgbImageWindow = "RGB Image";
	const string depthImageWindow = "Depth Image";
	const string mappedRgbImageWindow = "MappedRGB Image";
	const string wdrDepthImageWindow = "WDR Depth Image";

	bool f_bDistortionCorrection = false;
	bool f_bFilter = false;
	bool f_bMappedRGB = true;
	bool f_bWDRMode = false;
	bool f_bInvalidDepth2Zero = false;

	Size cropSize;
	if ((depthFrameMode.resolutionWidth / (float)depthFrameMode.resolutionHeight) > WHRatio)
	{
		cropSize = Size(static_cast<int>(depthFrameMode.resolutionHeight * WHRatio),
			depthFrameMode.resolutionHeight);
	}
	else
	{
		cropSize = Size(depthFrameMode.resolutionWidth,
			static_cast<int>(depthFrameMode.resolutionWidth / WHRatio));
	}

	Rect crop(Point((depthFrameMode.resolutionWidth - cropSize.width) / 2,
		(depthFrameMode.resolutionHeight - cropSize.height) / 2),
		cropSize);

	//const auto window_name = "Display Image";
	//namedWindow(window_name, WINDOW_AUTOSIZE);

	PsFrame depthFrame = { 0 };
	PsFrame irFrame = { 0 };

	//Ptr<BackgroundSubtractor> pBackSub;
	//pBackSub = createBackgroundSubtractorKNN();
	//Mat fgMask;

	/*while (true) {
		status = PsReadNextFrame(deviceIndex);
		PsGetFrame(deviceIndex, PsDepthFrame, &depthFrame);
		if (depthFrame.pFrameData != NULL)
		{
			auto depth_mat = cv::Mat(depthFrameMode.resolutionHeight, depthFrameMode.resolutionWidth, CV_16UC1, depthFrame.pFrameData);
			pBackSub->apply(depth_mat, fgMask);
			//imshow("Frame", depth_mat);
			imshow("FG Mask", fgMask);
		}
		
		int keyboard = waitKey(10);
			
	}
	*/

	ofstream out("./log", ios::app);
	if (out.fail()) {
		cout << "error\n";
	}


	for (;;)
	{

		// Read one frame before call PsGetFrame
		status = PsReadNextFrame(deviceIndex);
		PsGetFrame(deviceIndex, PsDepthFrame, &depthFrame);
		/*if (depthFrame.pFrameData != NULL) {
			out << "Frame " << i << endl;
			PsDepthPixel* DepthFrameData = (PsDepthPixel*)depthFrame.pFrameData;
			for (int k = 0; k < (depthFrameMode.resolutionHeight);  k++){
				for (int j = 0; j < (depthFrameMode.resolutionWidth); j++) {
					out << DepthFrameData[k*j + j]<<" ";
				}
				out << endl;
			}	

		}*/

		PsGetFrame(deviceIndex, PsIRFrame, &irFrame);

		//static int last_frame_number = 0;
		//if (rgbFrame.frameIndex == last_frame_number)
			//continue;
		//last_frame_number = rgbFrame.frameIndex;

		if (irFrame.pFrameData == NULL || depthFrame.pFrameData == NULL) {
			continue;

		}

		auto color_mat = cv::Mat(irFrameMode.resolutionHeight, irFrameMode.resolutionWidth, CV_16UC1, irFrame.pFrameData);
		auto depth_mat = cv::Mat(depthFrameMode.resolutionHeight, depthFrameMode.resolutionWidth, CV_16UC1, depthFrame.pFrameData);

		double distance_scale = 255.0 / 3000;
		color_mat.convertTo(color_mat, CV_8U, distance_scale);
		applyColorMap(color_mat, color_mat, cv::COLORMAP_RAINBOW);
		Mat inputBlob = blobFromImage(color_mat, inScaleFactor, Size(inWidth, inHeight), meanVal, false); 
		net.setInput(inputBlob, "data");
		Mat detection = net.forward("detection_out");
		Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());
		color_mat = color_mat(crop);
		depth_mat = depth_mat(crop);
        
		float confidenceThreshold = 0.8f;

		for (int i = 0; i < detectionMat.rows; i++)
		{
			float confidence = detectionMat.at<float>(i, 2);
			cout << "confidence is" << confidence << endl;

			if (confidence > confidenceThreshold)
			{
				size_t objectClass = (size_t)(detectionMat.at<float>(i, 1));

				int xLeftBottom = static_cast<int>(detectionMat.at<float>(i, 3) * color_mat.cols);
				int yLeftBottom = static_cast<int>(detectionMat.at<float>(i, 4) * color_mat.rows);
				int xRightTop = static_cast<int>(detectionMat.at<float>(i, 5) * color_mat.cols);
				int yRightTop = static_cast<int>(detectionMat.at<float>(i, 6) * color_mat.rows);

				Rect object((int)xLeftBottom, (int)yLeftBottom,
					(int)(xRightTop - xLeftBottom),
					(int)(yRightTop - yLeftBottom));

				object = object & Rect(0, 0, depth_mat.cols, depth_mat.rows);

				// Calculate mean depth inside the detection region
				// This is a very naive way to estimate objects depth
				// but it is intended to demonstrate how one might 
				// use depht data in general
				Scalar m = mean(depth_mat(object));

				std::ostringstream ss;
				ss << classNames[objectClass] << " ";
				ss << std::setprecision(2) << m[0] << " millimeters away";
				cout << std::setprecision(2) << m[0] << " millimeters away";
				String conf(ss.str());

				rectangle(color_mat, object, Scalar(0, 255, 0));
				int baseLine = 0;
				Size labelSize = getTextSize(ss.str(), FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

				auto center = (object.br() + object.tl()) * 0.5;
				center.x = center.x - labelSize.width / 2;

				rectangle(color_mat, Rect(Point(center.x, center.y - labelSize.height),
					Size(labelSize.width, labelSize.height + baseLine)),
					Scalar(255, 255, 255), FILLED);
				putText(color_mat, ss.str(), center,
					FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0));
			}
		}

		imshow("depth camera", color_mat);
		unsigned char key = waitKey(10);

	}


	status = PsCloseDevice(deviceIndex);
	cout << "CloseDevice status: " << status << endl;

	status = PsShutdown();
	cout << "Shutdown status: " << status << endl;

	return 0;
}