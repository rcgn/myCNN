#ifndef _CNNFF_H_
#define _CNNFF_H_
#include "cnnCom.h"


void ActivationFunction(float* net, float* output ,int length, int actFuncFlag);

int GetConvnOutImage(CNNCONVNLAYER* layer);

void GetSingleConvn(IMAGESIZE* inputImage, float***w, float b, float* output, int scale);

float* CnnFf(CNN* cnn, float* x);

float GetSamplingNum(float** inputImage, int samplingFlag, int inputImageWidth, int inputImageHeight, int scale, int yIndex, int xIndex);

void GetSingleSampling(float** inputImage, float** outImage, int samplingFlag, int inputImageWidth, int inputImageHeight, int outImageWidth, int outImageHeight, int scale);

void GetSamplingOutImage(CNNSAMPLINGLAYER* layer);

float* GetAllConnectNetResult(ALLCONNECTNET* layer);
#endif