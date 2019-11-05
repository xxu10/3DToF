#include <deque>
#include <string>
#include <pthread.h>
#include <math.h>
#include <opencv2/opencv.hpp>
#include "PicoZense_api.h"

#ifndef __TOFAPP_H__
#define __TOFAPP_H__

using namespace std;
using namespace cv;

class ToFApp{
public:
  ToFApp();
  void update(uint8_t* depthpData, uint8_t* irpData);
  void resetBackground();

public:
  Mat _dMat, _iMat, _bMat, _bkgndMat, _irbkgnd;
  bool _setBackground;
  int _depthThresh;
  int _ampGain;
  int _ampThresh;
  int _minContourArea;
  int _aspectRatio;
  int32_t deviceIndex;
  Ptr<BackgroundSubtractor> pBackSub;

private:
  pthread_t _thread;
  pthread_mutex_t _mtx;
  bool isPerson(vector<cv::Point>&contour, Mat dmat);
  Mat clipBackground(float dThr, float iThr);
  void getPCA(vector<cv::Point>&contour, float &center, float &angle);
};

#endif
