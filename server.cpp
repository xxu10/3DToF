#include <iostream>          // For cout and cerr
#include <cstdlib>           // For atoi()
#include <stdio.h>   
#include <sys/types.h>   
#include <sys/socket.h>   
#include <netinet/in.h>   
#include <unistd.h>   
#include <errno.h>   
#include <string.h>   
#include <stdlib.h>   
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include "PicoZense_api.h"

using namespace std;
using namespace cv;

#define BUF_LEN 65540
#define PACK_SIZE 4096
#define resolutionHeight_pre 640
#define resolutionWidth_pre 480
#define slope_pre 1450

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


int main(int argc, char* argv[])  
{  

    if (argc != 2) { // Test for correct number of parameters
        cerr << "Usage: " << argv[0] << " <Server Port>" << endl;
        exit(1);
    }
    unsigned int SERV_PORT = atoi(argv[1]);
    char buffer [BUF_LEN];
    int recvMsgSize;
    clock_t last_cycle = clock();

    int sock_fd = socket(AF_INET, SOCK_DGRAM, 0);
    int len;

    if(sock_fd < 0)  
    {  
       perror("socket");  
       exit(1);  
    }  
  
    struct sockaddr_in addr_serv;   
    memset(&addr_serv, 0, sizeof(struct sockaddr_in));
    addr_serv.sin_family = AF_INET;              
    addr_serv.sin_port = htons(SERV_PORT);            
    addr_serv.sin_addr.s_addr = htonl(INADDR_ANY);
    len = sizeof(addr_serv);
    
    if(bind(sock_fd, (struct sockaddr *)&addr_serv, sizeof(addr_serv)) < 0)  
    {  
       perror("bind error:");  
       exit(1);  
    }  
  
    struct sockaddr_in addr_client;  
    cv::Mat imageMat;

    while(1)  
    {    
       do{
           recvMsgSize = recvfrom(sock_fd,buffer,BUF_LEN,0, (struct sockaddr *)&addr_client, (socklen_t *)&len);
       }while(recvMsgSize > sizeof(int));

       int total_pack = ((int * ) buffer)[0];
       cout << "expecting length of packs:" << total_pack << endl;
       char *longbuf = new char[PACK_SIZE * total_pack];
       for (int i = 0; i < total_pack; i++) {
                recvMsgSize = recvfrom(sock_fd,buffer,BUF_LEN,0, (struct sockaddr *)&addr_client, (socklen_t *)&len);
                if (recvMsgSize != PACK_SIZE) {
                    cerr << "Received unexpected size pack:" << recvMsgSize << endl;
                    continue;
                }
              memcpy( & longbuf[i * PACK_SIZE], buffer, PACK_SIZE);
        }

        Mat img = Mat(1,PACK_SIZE * total_pack, CV_16UC1, longbuf);
        Mat imageMat = imdecode(img, CV_LOAD_IMAGE_ANYDEPTH);
	//cout<<"image pixel size is :"<<imageMat.elemSize()<<endl;
        if (imageMat.size().width == 0) {
                cerr << "decode failure!" << endl;
                continue;
        }
        Opencv_Depth(slope_pre,resolutionHeight_pre,resolutionWidth_pre,imageMat);
        cv::imshow("depthdata",imageMat);
        free(longbuf);

        waitKey(1);
        clock_t next_cycle = clock();
        double duration = (next_cycle - last_cycle) / (double) CLOCKS_PER_SEC;
        cout << "\teffective FPS:" << (1 / duration) << " \tkbps:" << (PACK_SIZE * total_pack / duration / 1024 * 8) << endl;
        cout << next_cycle - last_cycle;
        last_cycle = next_cycle;
     }  
    
  close(sock_fd);   
  return 0;  
}
