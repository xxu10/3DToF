
#include <iostream>
//#include "ToFApp.h"
#include "PicoZense_api.h"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <deque>
#include <sys/types.h>       // For data types
#include <sys/socket.h>      // For socket(), connect(), send(), and recv()
#include <netinet/in.h>      // For sockaddr_in
#include <netdb.h>           // For gethostbyname()
#include <arpa/inet.h>       // For inet_addr()
#include <unistd.h>          // For close()
#include <vector>
#include <string.h>
#include <errno.h>
//#include <stdlib.h>
#include <cstdlib>

using namespace std;
using namespace cv;

#define FRAME_HEIGHT 480
#define FRAME_WIDTH 640
#define FRAME_INTERVAL (1000/30)
#define PACK_SIZE 4096 //udp pack size; note that OSX limits < 8100 bytes
#define PNG_COMPRESSION 9

static deque<PsFrame*> qFrame;

static void Opencv_Depth(uint32_t slope, int resolutionHeight, int resolutionWidth, cv::Mat& dispImg)
{
        
        Point2d pointxy(resolutionWidth / 2, resolutionHeight / 2);
        int val = dispImg.at<ushort>(pointxy);
        char text[20];
#ifdef _WIN32
        sprintf_s(text, "%d", val);
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

int main(int argc, char* argv[]) {

	if ((argc < 3) || (argc > 3)) { // Test for correct number of arguments
        cerr << "Usage: " << argv[0] << " <Server> <Server Port>\n";
        exit(1);
    }
	const char * DEST_IP_ADDRESS = argv[1];
	int DEST_PORT = atoi(argv[2]);

	cout<<"The dest Ip is : "<<DEST_IP_ADDRESS<<" : "<<DEST_PORT<<endl;

	PsReturnStatus status;
	int32_t deviceIndex = 0;
	int32_t deviceCount = 0;
	uint32_t slope = 1450;
	uint32_t wdrSlope = 4400;
	PsDepthRange depthRange = PsNearRange;
	int32_t dataMode = PsDepthAndRGB_30;

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
	status = PsSetDepthRange(deviceIndex, PsNearRange);
	if (status != PsReturnStatus::PsRetOK)
		cout << "PsSetDepthRange failed!" << endl;
	else
		cout << "Set Depth Range to Near" << endl;

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
	
	status = PsGetFrameMode(deviceIndex, PsDepthFrame, &depthFrameMode);
	cout << "Get Depth Frame mode status: " << status << endl;
	cout << "depthFrameMode.pixelFormat: " << depthFrameMode.pixelFormat << endl;
	cout << "depthFrameMode.resolutionWidth: " << depthFrameMode.resolutionWidth << endl;
	cout << "depthFrameMode.resolutionHeight: " << depthFrameMode.resolutionHeight << endl;
	cout << "depthFrameMode.fps: " << depthFrameMode.fps << endl;

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

	bool done = false;

	PsFrame depthFrame = { 0 };
	int png_cps =  PNG_COMPRESSION; // Compression Parameter
    Mat img;
    vector < uchar > encoded;

    int sock_fd;  
    sock_fd = socket(AF_INET, SOCK_DGRAM, 0);  
    if(sock_fd < 0)  
    {  
           perror("socket");  
           exit(1);  
    }  
    int send_num = 0;
    struct sockaddr_in addr_serv;    
    memset(&addr_serv, 0, sizeof(addr_serv));  
    addr_serv.sin_family = AF_INET;  
    addr_serv.sin_addr.s_addr = inet_addr(DEST_IP_ADDRESS);  
    addr_serv.sin_port = htons(DEST_PORT); 

	clock_t last_cycle = clock();
        
	if(connect(sock_fd,(struct sockaddr*)&addr_serv,sizeof(addr_serv))< 0 ){
          printf("Connect socket error: %s (errno : %d)\n",strerror(errno),errno);
          return 0;
        }	
        
	while (!done) {
	   PsReadNextFrame(deviceIndex);
	   PsGetFrame(deviceIndex, PsDepthFrame, &depthFrame);
	   if (depthFrame.pFrameData != NULL)
	   {
	       img = Mat(depthFrameMode.resolutionHeight, depthFrameMode.resolutionWidth, CV_16UC1, depthFrame.pFrameData);
	       vector < int > compression_params;
               compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
               compression_params.push_back(png_cps); 
	       imencode(".png", img, encoded, compression_params);
		   Opencv_Depth(slope, FRAME_HEIGHT,FRAME_WIDTH, img);
		   cv::imshow("depthImage_send", img);
	       int total_pack = 1 + (encoded.size() - 1) / PACK_SIZE;
               int ibuf[1];
               ibuf[0] = total_pack;
               if(sendto(sock_fd,ibuf,sizeof(int),0,(struct sockaddr *)&addr_serv, sizeof(addr_serv)) != sizeof(int)){
	          perror("sendto error:");  
                  exit(1); 
	       }
	       cout<<"total pack is : "<<total_pack<<endl;
	       for(int i=0;i<total_pack;i++){
		   if(sendto(sock_fd,& encoded[i * PACK_SIZE],PACK_SIZE,0,(struct sockaddr *)&addr_serv, sizeof(addr_serv)) != PACK_SIZE){
		      perror("sendto error:");  
                      exit(1); 
		   }
	       }
	       waitKey(FRAME_INTERVAL);
	       clock_t next_cycle = clock();
               double duration = (next_cycle - last_cycle) / (double) CLOCKS_PER_SEC;
               cout << "\teffective FPS:" << (1 / duration) << " \tkbps:" << (PACK_SIZE * total_pack / duration / 1024 * 8) << endl;
               cout << next_cycle - last_cycle;
               last_cycle = next_cycle;

	   }
		unsigned char key = waitKey(10);
	}
	close(sock_fd);  
	return 0;
}
