
#include <iostream>
#include "ToFApp.h"
#include "PicoZense_api.h"
#include <vector>
#include <deque>
#include <cstdint>
#include <unistd.h>

#define Queue_len (10)

using namespace std;
using namespace cv;

static deque<PsFrame> depthFrames;
static deque<PsFrame> irFrames;
static pthread_mutex_t gmtx = PTHREAD_MUTEX_INITIALIZER;

static void Opencv_Depth(uint32_t slope, PsFrameMode depthFrameMode, uint8_t* pData, cv::Mat& dispImg)
{
	dispImg = cv::Mat(depthFrameMode.resolutionHeight, depthFrameMode.resolutionWidth, CV_16UC1, pData);
	Point2d pointxy(depthFrameMode.resolutionWidth / 2, depthFrameMode.resolutionHeight / 2);
	int val = dispImg.at<ushort>(pointxy);
	char text[20];
#ifdef _WIN32
	//sprintf_s(text, "%d", val);
#else
	snprintf(text, sizeof(text), "%d", val);
#endif
	dispImg.convertTo(dispImg, CV_8U, 255.0 / slope);
	applyColorMap(dispImg, dispImg, cv::COLORMAP_RAINBOW);
	int color;
	if (val > 2500)
		color = 0;
	else
		color = 4096;
	circle(dispImg, pointxy, 4, Scalar(color, color, color), -1, 8, 0);
	putText(dispImg, text, pointxy, FONT_HERSHEY_DUPLEX, 2, Scalar(color, color, color));
}

void getframes(int32_t deviceIndex, size_t number, vector<Mat>& depth_frames, vector<Mat>& ir_frames) {
	for (int i = 0; i < number; i++) {
		PsFrame depthFrame, irFrame;
		PsReadNextFrame(deviceIndex);
		PsGetFrame(deviceIndex, PsDepthFrame, &depthFrame);
		PsGetFrame(deviceIndex, PsIRFrame, &irFrame);
		if (depthFrame.pFrameData != NULL && irFrame.pFrameData !=NULL )
		{

			Mat depth_data = Mat(480, 640, CV_16UC1, depthFrame.pFrameData);
			Mat ir_data = Mat(480, 640, CV_16UC1, irFrame.pFrameData);
			depth_frames.push_back (depth_data.clone());
			ir_frames.push_back(ir_data.clone());

		}
		else {
			i--;
			continue;
		}
		unsigned char key = waitKey(1);
		cout << i << endl;

	}

}

void *eventLoop(void* arg){

	//sleep(5000);
	bool empty;
	bool done = false;
	ToFApp app;
	while(!done){
		pthread_mutex_lock(&gmtx);
		empty = depthFrames.empty()|| irFrames.empty();
		pthread_mutex_unlock(&gmtx);
		if(!empty){
			cout<<"second get here"<<endl;
			pthread_mutex_lock(&gmtx);
			PsFrame depthFrm = depthFrames.front();
			PsFrame irFrm = irFrames.front();
			pthread_mutex_unlock(&gmtx);

			app.update(depthFrm.pFrameData, irFrm.pFrameData);

			pthread_mutex_lock(&gmtx);
			depthFrames.pop_front();
			irFrames.pop_front();
			pthread_mutex_unlock(&gmtx);

			waitKey(1);
		}
	}

	pthread_exit(NULL);
}


int main(int argc, char* argv[]) {

	PsReturnStatus status;
	int32_t deviceIndex = 0;
	int32_t deviceCount = 0;
	uint32_t slope = 1450;
	uint32_t wdrSlope = 4400;
	PsDepthRange depthRange = PsXFarRange;
	int32_t dataMode = PsDepthAndIR_30;

	status = PsInitialize();
	if (status != PsReturnStatus::PsRetOK)
	{
		cout << "PsInitialize failed!" << endl;
		//system("pause");
		return -1;
	}

	status = PsGetDeviceCount(&deviceCount);
	if (status != PsReturnStatus::PsRetOK)
	{
		cout << "PsGetDeviceCount failed!" << endl;
		//system("pause");
		return -1;
	}
	cout << "Get device count: " << deviceCount << endl;

	//Set the Depth Range to Near through PsSetDepthRange interface
	status = PsSetDepthRange(deviceIndex, depthRange);
	if (status != PsReturnStatus::PsRetOK)
		cout << "PsSetDepthRange failed!" << endl;
	else
		cout << "Set Depth Range to Near" << endl;

	status = PsOpenDevice(deviceIndex);
	if (status != PsReturnStatus::PsRetOK)
	{
		cout << "OpenDevice failed!" << endl;
		//system("pause");
		return -1;
	}

	//Set PixelFormat as PsPixelFormatBGR888 for opencv display
	PsSetColorPixelFormat(deviceIndex, PsPixelFormatBGR888);

	//Set to DepthAndRGB_30 mode
	PsSetDataMode(deviceIndex, (PsDataMode)dataMode);

	PsFrameMode depthFrameMode;
	PsFrameMode irFrameMode;
	PsFrameMode rgbFrameMode;
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

	status = PsGetFrameMode(deviceIndex, PsRGBFrame, &rgbFrameMode);
	cout << "Get RGB Frame mode status: " << status << endl;
	cout << "rgbFrameMode.pixelFormat: " << rgbFrameMode.pixelFormat << endl;
	cout << "rgbFrameMode.resolutionWidth: " << rgbFrameMode.resolutionWidth << endl;
	cout << "rgbFrameMode.resolutionHeight: " << rgbFrameMode.resolutionHeight << endl;
	cout << "rgbFrameMode.fps: " << rgbFrameMode.fps << endl;

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

	const string irImageWindow = "IR Image";
	const string rgbImageWindow = "RGB Image";
	const string depthImageWindow = "Depth Image";
	const string mappedRgbImageWindow = "MappedRGB Image";
	const string wdrDepthImageWindow = "WDR Depth Image";

	ofstream PointCloudWriter;
	PsDepthVector3 DepthVector = { 0, 0, 0 };
	PsVector3f WorldVector = { 0.0f };

	bool f_bDistortionCorrection = false;
	bool f_bFilter = false;
	bool f_bMappedRGB = true;
	bool f_bWDRMode = false;
	bool f_bInvalidDepth2Zero = false;

	cout << "\n--------------------------------------------------------------------" << endl;
	cout << "--------------------------------------------------------------------" << endl;
	cout << "Press following key to set corresponding feature:" << endl;
	cout << "M/m: Change data mode: input corresponding index in terminal:" << endl;
	cout << "                    0: Output Depth and RGB in 30 fps" << endl;
	cout << "                    1: Output IR and RGB in 30 fps" << endl;
	cout << "                    2: Output Depth and IR in 30 fps" << endl;
	cout << "                    3: Output Single Depth in 60 fps" << endl;
	cout << "                    4: Output Single IR in 60 fps" << endl;
	cout << "                    5: Output Depth and IR and RGB in 30 fps" << endl;
	cout << "                    6: Output Depth/IR frames alternatively in 15fps, and RGB in 30fps" << endl;
	cout << "                    7: Output WDR_Depth and RGB in 30 fps" << endl;
	cout << "0/1/2...: Change depth range Near/Middle/Far..." << endl;
	cout << "R/r: Change the RGB resolution: input corresponding index in terminal:" << endl;
	cout << "                             0: 1920*1080" << endl;
	cout << "                             1: 1280*720" << endl;
	cout << "                             2: 640*480" << endl;
	cout << "                             3: 640*360" << endl;
	cout << "P/p: Save point cloud data into PointCloud.txt in current directory" << endl;
	cout << "T/t: Change background filter threshold value" << endl;
	cout << "U/u: Enable or disable the distortion correction feature" << endl;
	cout << "F/f: Enable or disable the smoothing filter feature" << endl;
	cout << "Q/q: Enable or disable the mapped RGB in Depth space" << endl;
	cout << "V/v: Enable or disable the WDR depth fusion feature " << endl;
	cout << "Esc: Program quit " << endl;
	cout << "--------------------------------------------------------------------" << endl;
	cout << "--------------------------------------------------------------------\n" << endl;

	bool done = false;

	ToFApp app;
	cv::Mat imageMat;

	PsFrame depthFrame = { 0 };
	PsFrame irFrame = { 0 };

	pthread_mutex_init(&gmtx,NULL);

	int key = 0, ret = 0, count = 0;
	pthread_t tid;
	ret = pthread_create(&tid,NULL,eventLoop,NULL);
	if(ret!=0){
		cout<< "Multi-threading failed";
		return 0;
	}

	/*for (int i = 0; i < 100; i++) {
		PsReadNextFrame(deviceIndex);
		PsGetFrame(deviceIndex, PsDepthFrame, &depthFrame);
		PsGetFrame(deviceIndex, PsIRFrame, &irFrame);
		if (depthFrame.pFrameData != NULL && irFrame.pFrameData != NULL)
		{

			Mat depth_data = Mat(depthFrameMode.resolutionHeight, depthFrameMode.resolutionWidth, CV_16UC1, depthFrame.pFrameData);
			Mat ir_data = Mat(irFrameMode.resolutionHeight, irFrameMode.resolutionWidth, CV_16UC1, irFrame.pFrameData);
			frames[i] = depth_data.clone();
			ir[i] = ir_data.clone();
			
		}
		else {
			i--;
			continue;
		}
		unsigned char key = waitKey(1);
		cout << i << endl;

	}


	for (int i = 0; i < 100; i++) {
		frames[i].convertTo(frames[i], CV_8U, 255.0 / slope);
		applyColorMap(frames[i], frames[i], cv::COLORMAP_RAINBOW);
		cv::imshow(depthImageWindow, frames[i]);
		cout << "print " << i << endl;
		unsigned char key = waitKey(1);
	}

	for (int i = 0; i < 100; i++) {
		ir[i].convertTo(ir[i], CV_8U, 255.0 / 40);
		applyColorMap(ir[i], ir[i], cv::COLORMAP_RAINBOW);
		cv::imshow("ir image", ir[i]);
	}

	*/

	while (true) {
		PsReadNextFrame(deviceIndex);
		PsGetFrame(deviceIndex, PsDepthFrame, &depthFrame);
		PsGetFrame(deviceIndex, PsIRFrame, &irFrame);

		if (depthFrame.pFrameData != NULL && irFrame.pFrameData != NULL)
		{
			cout<<"gethere"<<endl;
			count++;
			if (count > 5) {
				cout<<"capture"<<endl;
				pthread_mutex_lock(&gmtx);
				if(depthFrames.size()< Queue_len){
					depthFrames.push_back(depthFrame);
				}
				if(irFrames.size() < Queue_len){
					irFrames.push_back(irFrame);
				}
				//app.update(depthFrame.pFrameData, irFrame.pFrameData, depthFrameMode, irFrameMode);
				pthread_mutex_unlock(&gmtx);
				cout<<"first here"<<endl;
				count = 6;
			}
			
			//Opencv_Depth(slope, depthFrameMode, depthFrame.pFrameData, imageMat);
			//cv::imshow(depthImageWindow, imageMat);
			if (key == 'b')
				app.resetBackground();
		}
	
		unsigned char key = waitKey(1);
	}
	
	pthread_join(tid,NULL);

	status = PsCloseDevice(deviceIndex);
	status = PsShutdown();
	return 0;

}
