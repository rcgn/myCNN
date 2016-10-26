#ifndef _CNN_SETUP_H_
#define _CNN_SETUP_H_
#include "cnnCom.h"


int SetCnn(CNN* cnn, int layerNum, unsigned char types[], int scales[], int layerImageNum[], 
	int imageNum, int imageWidth, int imageHeight,
	char actFuncFlag[], char samplingFlag, char ImageFlag);

int SetCnnLayer(char type, void** layer, int scale, char actFuncFlag, char samplingFlag, char ImageFlag, 
	int layerImageNum, IMAGESIZE *inImage, IMAGESIZE **outImage, int layerIndex, int inputsize,
	float* input, float** output, float* dinput, float** doutput);

int SetConvnLayer(CNNCONVNLAYER** layer, int scale, char actFuncFlag, char ImageFlag, int layerImageNum, 
	IMAGESIZE *inImage, IMAGESIZE **outImage, int layerIndex);

int SetSamplingLayer(CNNSAMPLINGLAYER** layer, int scale, char samplingFlag, char ImageFlag, 
	IMAGESIZE *inImage, IMAGESIZE **outImage);

//int SetAllConnectNet(ALLCONNECTNET** layer, int outsize, IMAGESIZE *inImage, int layerIndex);
int SetAllConnectNet(ALLCONNECTNET** layer, int outsize, IMAGESIZE *inImage, int layerIndex, 
	int inputsize, float* input, float** output, float* dinput, float** doutput);
int SetSoftmaxLayer(CNNSOFTMAXLAYER** layer, int outsize, IMAGESIZE *inImage, int layerIndex, 
	int inputsize, float* input, float** output, float* dinput, float** doutput);

void DleteCnn(CNN* cnn);
#endif