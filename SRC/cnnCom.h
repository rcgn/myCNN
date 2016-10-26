#ifndef _CNN_COM_H_
#define _CNN_COM_H_

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <memory.h>
#include "cnnData.h"

#define TRAIN


//卷积计算方式
#define FULL		1
#define VALID		2

//网络可支持最大层数
#define MAX_LAYER	30

//损失函数类型，与最后一层的结构相关
#define BP_NN		1//直接BP网络输出
#define SOFTMAX		2//softmax回归输出

#define SOFTMAX_LAMBDA   0.0001f
extern float gSoftmaxLambda;

//激活函数类型
#define SIGMOD		1
#define RELU		2
#define LEAKY_RELU	3

//激活函数参数
#define LEAKY_RELU_RATIO 0.1f
//图像是否填充标志
#define FILLER		1
#define NOFILLER	0

//sampling层采样类型
#define MAX_TYPE	1
#define MEAN_TYPE	2

//每一层的类型标志
#define CONVNLAYER		1
#define SAMPLINGLAYER	2
#define ALLCONNECTLAYER	3
#define SOFTMAXLAYER	4

//错误码
#define TOO_MANY_LAYERS								2
#define TOO_FEW_LAYERS								3
#define OUTPUT_LAYER_ERR							4
#define EMPTY_SAMPLE								5
#define EMPTY_CNN									6
#define EMPTY_OUTPUT								7
#define ERR_LAYER_TYPE								8
#define ERR_LAST_LAYER_TYPE							9

#define MOLLOC_ERR_IN_UPDATE_COVN_LAYER_DINPUT		10
#define MOLLOC_ERR_IN_UPDATE_CONVN_LAYER_DK			11
#define MOLLOC_ERR_IN_GET_CONVN_OUT_IMAGE			12
#define MOLLOC_IN_IMAGE_ERR_IN_SET_CNN				13
#define MOLLOC_ERR_IN_SET_CONVN_LAYER				14
#define MOLLOC_ERR_IN_SET_SAMPLING_LAYER			15
#define MOLLOC_ERR_IN_SET_ALL_CONNECT_NET			16
#define MOLLOC_ERR_IN_SET_SOFTMAX_LAYER				17
#define MOLLOC_ERR_IN_CNN_TRAIN						18
#define MATRIX_TRANSPOSE_ERR						19
#define MATRIX_ROTATION_180_ERR						20

typedef struct tag_ImageSize
{
	int ImageNum;
	int width;
	int height;
	float*** Image;
	float*** dImage;
	float*** dNet;
}IMAGESIZE;


typedef struct tag_CnnConvnLayer
{
	float ****k;			//参数指针
	float ****dk;			//参数偏导
	float *b;				//偏置指针
	float *db;
	char actFuncFlag;		//激活函数类型 
	char ImageFlag;			//图像填充标志 0为不填充，其他值为填充
	int kernelSize;			//核尺寸
	IMAGESIZE* inputImage;	//输入图像
	IMAGESIZE* outputImage;	//输出图像
}CNNCONVNLAYER;


typedef struct tag_CnnSamplingLayer
{
	char samplingFlag;		//抽样类型标志
	char ImageFlag;			//图像填充标志 0为不填充，其他值为填充
	int scale;				//抽样尺寸
	IMAGESIZE* inputImage;	//输入图像
	IMAGESIZE* outputImage;	//输出图像
}CNNSAMPLINGLAYER;

typedef struct tag_AllConnectNet
{
	float** k;
	float** dk;
	float* b;
	float* db;
	float* input;
	float* dinput;
	IMAGESIZE* inputImage;	//输入图像
	float* net;
	float* output;
	float* doutput;
	int inputSize;
	int outputSize;
}ALLCONNECTNET;

typedef struct tag_SoftMaxLayer
{
	float** k;
	float** dk;
	float* b;
	float* db;
	float* input;
	float* dinput;
	IMAGESIZE* inputImage;	//输入图像
	float* net;
	float* output;
	float* doutput;
	int inputSize;
	int outputSize;
}CNNSOFTMAXLAYER;


typedef struct tag_CNN
{
	int inputImageNum;
	int inputImageHeight;
	int inputImageWidth;
	int outSize;

	int depth;
	char lossType;
	void* layer[30];
	char type[30];
}CNN;

typedef struct tag_TrainPara
{
	int batchSize;
	int numepochs;
	float alpha;
}TRANPARA;

float*** MallocImageBlock(int outImageNum, int height, int width);
void DeleteImageBlock(float*** block, int outImageNum, int height, int width);

float**** MallocParaBlock(int outImageNum, int inputImageNum, int height, int width, float* w);
void DeleteParaBlock(float**** block, int outImageNum, int inputImageNum, int height, int width);

void Sigmod(float* net, float* output ,int length);
void Softmax(float* net, float* output ,int length);
void LeakyRelu(float* net, float* output ,int length);
void Relu(float* net, float* output ,int length);
void GetNet(float** w, float* b, float* input, float* net, int inputSize, int outputSize);

void Convn(float** inputImage, float** w, float* output, int imageHeight, int imageWidth, int scale);
void Convn2(float** matrix1, float** matrix2, float* output, int height1, int width1, int height2, int width2, char flag);
void initBlock(char* block, int size);
float** MallocSingleImage(int height, int width);
void DeleteSingleImage(float**matrix);
int MatrixRotation180(float** matrix, int height, int width, float** resultMatrix);
int MatrixTranspose(float** matrix, int height, int width, float** resultMatrix);
#endif