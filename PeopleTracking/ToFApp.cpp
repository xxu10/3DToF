#include "ToFApp.h"
#include <climits>
#include <algorithm>

ToFApp::ToFApp(){
  _setBackground = false;
  _ampGain = 100;
  _ampThresh = 100;
  _depthThresh = 100;
  _minContourArea = 300;
  _aspectRatio = 100;

}

Mat ToFApp::clipBackground( float dThr, float iThr){
  Mat dMat = Mat::zeros( _dMat.size(), CV_32FC1);
  Mat fMat = Mat::zeros(_dMat.size(), CV_32FC1);
  Mat irMat = Mat::zeros(_iMat.size(), CV_32FC1);
  //absdiff(_dMat,_bkgndMat, fMat);
  //cout << "central bias" << fMat.at<float>(320, 240) << endl;
  for(int i=0;i<_dMat.rows;i++){
    for(int j=0;j<_dMat.cols;j++){
		if (_iMat.at<float>(i, j) - _irbkgnd.at<float>(i, j) < 0.0f) {
			irMat.at<float>(i, j) = 0.0f;
		}
		else {
			irMat.at<float>(i, j) = _iMat.at<float>(i, j) - _irbkgnd.at<float>(i, j);
		}
		if ((_bkgndMat.at<float>(i, j) - _dMat.at<float>(i, j)) < 0.0f) {
			//cout << "background [(" << i << "," << j << ")] is " << _bkgndMat.at<float>(i, j)<<endl;
			//cout << "reading [(" << i << "," << j << ")] is " << _dMat.at<float>(i, j);
			fMat.at<float>(i, j) = 0.0f;
		}	
		else {
			fMat.at<float>(i, j) = (_bkgndMat.at<float>(i, j) - _dMat.at<float>(i, j));
		}
			
      dMat.at<float>(i,j) = (fMat.at<float>(i,j)>dThr && irMat.at<float>(i,j)>iThr)? 255.0:0.0;
    }
  }
  return dMat; 
}

void ToFApp::resetBackground(){
  _setBackground = false;
}

void ToFApp::getPCA(vector<cv::Point>&contour, float &center, float &angle){
  int sz = static_cast<int>(contour.size());
  Mat data_pts = Mat(sz,2,CV_32FC1);
  for(int i=0;i<data_pts.rows;++i){
    data_pts.at<float>(i,0) = contour[i].x;
    data_pts.at<float>(i,1) = contour[i].y;
  }

  PCA pca_analysis(data_pts,Mat(),CV_PCA_DATA_AS_ROW);

  cv::Point cntr = cv::Point(static_cast<int>(pca_analysis.mean.at<float>(0,0)),
                   static_cast<int>(pca_analysis.mean.at<float>(0,1)));
  vector<cv::Point2d> eigen_vecs(2);
  vector<float> eigen_vals(2);
  for(int i=0;i<2;++i){
    eigen_vecs[i] = Point2d(pca_analysis.eigenvectors.at<float>(i,0),
                    pca_analysis.eigenvectors.at<float>(i,1));
    eigen_vals[i] = pca_analysis.eigenvalues.at<float>(0,i);
  }

  angle = atan2(eigen_vecs[0].y,eigen_vecs[0].x);

}

bool ToFApp::isPerson(vector<cv::Point>& contour, Mat dMat) {
	bool rc = false;
	int area = 0;
	long sumX = 0, sumY = 0;
	int minX = INT_MAX, minY = INT_MAX;
	int maxX = 0, maxY = 0;
	int dx, dy;

	for (int i = 0; i < contour.size(); i++) {
		minX = std::min(minX, contour[i].x);
		minY = std::min(minY, contour[i].y);
		maxX = std::max(maxX, contour[i].x);
		maxY = std::max(maxY, contour[i].y);
		sumX += contour[i].x;
		sumY += contour[i].y;
	}
	dx = maxX - minX;
	dy = maxY - minY;

	if (contourArea(contour) > _minContourArea) {
		if (dx > 0) {
			float ratio = (float)dy / (float)dx;
			cout << "ratio = " << ratio << endl;
			if (ratio > (float)_aspectRatio / 100.0) {
				rc = true;
			}
		}
	}

	return rc;
}

void ToFApp::update(uint8_t *depthpData, uint8_t *irpData) {

	vector<vector<cv::Point>> contours;
	vector<Vec4i> hierarchy;
	RNG rng(12345);

	Mat _iMat_int;
	Mat _dMat_int;

	if (depthpData != NULL) {
		_dMat_int = Mat(480, 640, CV_16UC1, depthpData);
	}
	else {
		_dMat_int = Mat(480, 640, CV_16UC1);
	}

	if (irpData != NULL) {
		_iMat_int = Mat(480, 640, CV_16UC1, irpData);
	}
	else {
		_iMat_int = Mat(480, 640, CV_16UC1);
	}
    //_iMat = Mat(depthFrameMode.resolutionHeight, depthFrameMode.resolutionWidth, CV_16UC1, depthpData);
	//_dMat = Mat(irFrameMode.resolutionHeight, irFrameMode.resolutionWidth, CV_16UC1, irpData);

	//addWeighted(_iMat_int, 0.4, _dMat_int, 0.6, 0, _dMat_int);
	_iMat_int.convertTo(_iMat, CV_32FC1);
	//_dMat_int.copyTo(_dMat);
	_dMat_int.convertTo(_dMat, CV_32FC1);
	
	//_iMat = (float)_ampGain*_iMat;

	if (!_setBackground) {
		_dMat.copyTo(_bkgndMat);
		_setBackground = true;
		_iMat.copyTo(_irbkgnd);
		for (int i = 0; i<_bkgndMat.rows; i++) {
			for (int j = 0; j<_bkgndMat.cols; j++) {
				if (_bkgndMat.at<float>(i, j) == 0.0f) {
					_bkgndMat.at<float>(i, j) = 6200.0f;
				}
			}
		}
		
		cout << endl << "Updated background" << endl;
	}


	//pBackSub->apply(_dMat, _bMat, 0);

	Mat fMat = clipBackground((float)_depthThresh, (float)_ampThresh);
	fMat.convertTo(_bMat, CV_8U, 255.0);
	Mat morphMat = _bMat.clone();
	Mat element = getStructuringElement(0, Size(5, 5), cv::Point(1, 1));
	morphologyEx(_bMat, morphMat, 2, element);

	Mat drawing = Mat::zeros(_dMat.size(), CV_8UC3);
	Mat im_with_keypoints = Mat::zeros(_iMat.size(), CV_8UC3);
	//cvtColor(_iMat_int * 2, drawing, CV_GRAY2RGB);

	int peopleCount = 0;

	findContours(morphMat, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
	for (int i = 0; i < contours.size(); i++) {
		drawContours(drawing, contours, i, Scalar(0, 0, 255), 2, 8, vector<Vec4i>(), 0, cv::Point());
		if (isPerson(contours[i], _dMat)) {
			peopleCount++;
			
		}
	}

	putText(drawing, "Count = " + to_string(peopleCount), cv::Point(200, 50), FONT_HERSHEY_PLAIN, 1, Scalar(255, 255, 255));

	imshow("Binary", _bMat);
	//imshow("Amplitude", _iMat);
	imshow("Draw", drawing);
	imshow("Morph", morphMat);

	uchar key = waitKey(10);

}
