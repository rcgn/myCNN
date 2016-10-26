#include <stdlib.h>
#include <time.h>
#include "cnnSetup.h"

//int SetCnn(CNN* cnn, int layerNum, unsigned char types[], int scales[], int layerImageNum[], IMAGESIZE *inputImage,
//	char actFuncFlag, char samplingFlag, char ImageFlag)
int SetCnn(CNN* cnn, int layerNum, unsigned char types[], int scales[], int layerImageNum[], 
	int imageNum, int imageWidth, int imageHeight,
	char actFuncFlag[], char samplingFlag, char ImageFlag)
{
	int i;
	int result=0;
	IMAGESIZE *outImage = NULL;
	IMAGESIZE *inImage = NULL;
	float* output= NULL;
	float* input = NULL;
	float* doutput= NULL;
	float* dinput = NULL;
	//cnn网络层数设置
	if(layerNum >= MAX_LAYER)
	{
		return TOO_MANY_LAYERS;//
	}
	else if(layerNum < 1)
	{
		return TOO_FEW_LAYERS;//
	}
	else
	{
		cnn->depth = layerNum;
	}

	//cnn网络损失函数类型设置，与最后一层的结构有关，目前只支持两类：1.BP网络直接输出，2.softmax回归输出
	if(types[cnn->depth - 1] == ALLCONNECTLAYER)
	{
		cnn->lossType = BP_NN;
	}
	else if(types[cnn->depth - 1] == SOFTMAXLAYER)
	{
		cnn->lossType = SOFTMAX;
	}
	else 
	{
		return OUTPUT_LAYER_ERR;
	}

	cnn->inputImageNum = imageNum;
	cnn->inputImageHeight = imageHeight;
	cnn->inputImageWidth = imageWidth;
	cnn->outSize = layerImageNum[layerNum-1];

	//第一层input图像空间分配及设置
	inImage = (IMAGESIZE *)malloc(sizeof(IMAGESIZE));
	if(inImage == NULL)
	{
		return MOLLOC_IN_IMAGE_ERR_IN_SET_CNN;
	}
	else
	{
		inImage->ImageNum = imageNum;
		inImage->height = imageHeight;
		inImage->width = imageWidth;

		inImage->Image = MallocImageBlock(imageNum,imageHeight,imageWidth);
		inImage->dImage = MallocImageBlock(imageNum,imageHeight,imageWidth);
		if(inImage->Image == NULL || inImage->dImage == NULL)
		{
			if(inImage->Image != NULL ){free(inImage->Image);}
			if(inImage->dImage != NULL ){free(inImage->dImage);}
			free(inImage);
			return MOLLOC_IN_IMAGE_ERR_IN_SET_CNN;
		}
	}

	//各层网络结构设置
	for(i=0; i<cnn->depth; i++)
	{
		int inputsize = 0;//全连接层使用
		if(i>0)
		{
			inputsize = layerImageNum[i-1];
		}
		cnn->type[i] = types[i];
		result = SetCnnLayer(cnn->type[i], &(cnn->layer[i]), scales[i], actFuncFlag[i], samplingFlag, ImageFlag,
			layerImageNum[i], inImage, &outImage, i, inputsize, input, &output,dinput, &doutput);
		inImage = outImage;//卷积及抽样层使用
		input = output;//全连接层使用
		dinput = doutput;//全连接层使用
		if(result != 0)
		{
			DleteCnn(cnn);
			break;
		}
	}

	return result;
}


//	输入：
//	type:				当前层类型										
//	layer：				当前层指针   
//	scale：				当前层模板尺寸（卷积层为模板尺寸，抽样层为抽样尺寸）
//	actFuncFlag:		卷积层激活函数类型
//  samplingFlag：		抽样层抽样方式
//	ImageFlag：			图像是否填充为原大小
//	layerImageNum：		当前层输出图像数目
//	inImage:			当前层输入图像
//	outImage:			当前层输出图像
int SetCnnLayer(char type, void** layer, int scale, char actFuncFlag, char samplingFlag, char ImageFlag, 
	int layerImageNum, IMAGESIZE *inImage, IMAGESIZE **outImage, int layerIndex,  int inputsize, float* input, float** output,
	float* dinput, float** doutput)
{
	int result = 0;
	if(type == CONVNLAYER)
	{
		result = SetConvnLayer((CNNCONVNLAYER**)layer, scale, actFuncFlag, ImageFlag, layerImageNum, inImage, outImage, layerIndex);
		*output = NULL;
	}
	else if(type == SAMPLINGLAYER)
	{
		result = SetSamplingLayer((CNNSAMPLINGLAYER**)layer, scale, samplingFlag, ImageFlag, inImage, outImage);
		*output = NULL;
	}
	else if(type == ALLCONNECTLAYER)
	{
		result = SetAllConnectNet((ALLCONNECTNET**)layer, layerImageNum, inImage, layerIndex, inputsize, input, output, dinput, doutput);
		*outImage = NULL;
	}
	else if(type == SOFTMAXLAYER)
	{
		result = SetSoftmaxLayer((CNNSOFTMAXLAYER**)layer, layerImageNum, inImage, layerIndex, inputsize, input, output, dinput, doutput);
		*outImage = NULL;
	}
	else
	{
		result = ERR_LAYER_TYPE;
		*outImage = NULL;
		*output = NULL;
	}
	return result;
}



int SetConvnLayer(CNNCONVNLAYER** layer, int scale, char actFuncFlag, char ImageFlag, int layerImageNum, 
	IMAGESIZE *inImage, IMAGESIZE **outImage, int layerIndex)
{
	int i;
	CNNCONVNLAYER* convnLayer = (CNNCONVNLAYER*)malloc(sizeof(CNNCONVNLAYER));

	if(convnLayer == NULL)
	{
		return MOLLOC_ERR_IN_SET_CONVN_LAYER;
	}
#ifdef TRAIN
	W[layerIndex] = (float*)malloc(sizeof(float) * layerImageNum * inImage->ImageNum * scale * scale);
	B[layerIndex] = (float*)malloc(sizeof(float) * layerImageNum);

	if(W[layerIndex] == NULL || B[layerIndex] == NULL)
	{
		//free(convnLayer);
		return MOLLOC_ERR_IN_SET_ALL_CONNECT_NET;
	}

	srand((unsigned int) time(NULL));
	for(i=0; i<layerImageNum * inImage->ImageNum * scale * scale; i++)
	{
		W[layerIndex][i] = (float)((2.0f*rand()/RAND_MAX-1.0f)/(sqrt(inImage->ImageNum)*scale));//产生(-1,1)的随机数
	}
	for(i=0; i<layerImageNum; i++)
	{
		B[layerIndex][i] = 0;
	}
#endif
	dW[layerIndex] = (float*)malloc(sizeof(float) * layerImageNum * inImage->ImageNum * scale * scale);
	dB[layerIndex] = (float*)malloc(sizeof(float) * layerImageNum);

	if(dW[layerIndex] == NULL || dB[layerIndex] == NULL)
	{
		return MOLLOC_ERR_IN_SET_CONVN_LAYER;
	}

	for(i=0; i<layerImageNum * inImage->ImageNum * scale * scale; i++)
	{
		dW[layerIndex][i] = 0.0f;
	}
	for(i=0; i<layerImageNum; i++)
	{
		dB[layerIndex][i] = 0.0f;
	}



	convnLayer->actFuncFlag = actFuncFlag;
	convnLayer->kernelSize = scale;
	convnLayer->ImageFlag = ImageFlag;
	convnLayer->inputImage = inImage;
	convnLayer->b = B[layerIndex];
	convnLayer->db = dB[layerIndex];

	convnLayer->outputImage = (IMAGESIZE*)malloc(sizeof(IMAGESIZE));
	if(convnLayer->outputImage == NULL)
	{
		return MOLLOC_ERR_IN_SET_CONVN_LAYER;
	}
	else
	{
		convnLayer->outputImage->ImageNum = layerImageNum;
		if(convnLayer->ImageFlag == FILLER)
		{
			convnLayer->outputImage->height = convnLayer->inputImage->height;
			convnLayer->outputImage->width = convnLayer->inputImage->width;
		}
		else
		{
			convnLayer->outputImage->height = convnLayer->inputImage->height - convnLayer->kernelSize/2*2;
			convnLayer->outputImage->width = convnLayer->inputImage->width - convnLayer->kernelSize/2*2;
		}
		convnLayer->outputImage->Image = MallocImageBlock(convnLayer->outputImage->ImageNum, convnLayer->outputImage->height, convnLayer->outputImage->width);
		convnLayer->outputImage->dImage = MallocImageBlock(convnLayer->outputImage->ImageNum, convnLayer->outputImage->height, convnLayer->outputImage->width);
		convnLayer->outputImage->dNet = MallocImageBlock(convnLayer->outputImage->ImageNum, convnLayer->outputImage->height, convnLayer->outputImage->width);
		if(convnLayer->outputImage->Image == NULL || convnLayer->outputImage->dImage == NULL || convnLayer->outputImage->dNet == NULL)
		{
			return MOLLOC_ERR_IN_SET_CONVN_LAYER;
		}
	}

	convnLayer->k = MallocParaBlock(convnLayer->outputImage->ImageNum, convnLayer->inputImage->ImageNum, convnLayer->kernelSize, convnLayer->kernelSize, W[layerIndex]);
	convnLayer->dk = MallocParaBlock(convnLayer->outputImage->ImageNum, convnLayer->inputImage->ImageNum, convnLayer->kernelSize, convnLayer->kernelSize, dW[layerIndex]);
	if(convnLayer->k == NULL )
	{
		return MOLLOC_ERR_IN_SET_CONVN_LAYER;
	}

	*outImage = convnLayer->outputImage;
	*layer = convnLayer;
	return 0;
}

int SetSamplingLayer(CNNSAMPLINGLAYER** layer, int scale, char samplingFlag, char ImageFlag, IMAGESIZE *inImage, IMAGESIZE **outImage)
{
	CNNSAMPLINGLAYER* samplingLayer = (CNNSAMPLINGLAYER*)malloc(sizeof(CNNSAMPLINGLAYER));

	if(samplingLayer == NULL)
	{
		return MOLLOC_ERR_IN_SET_SAMPLING_LAYER;
	}

	samplingLayer->ImageFlag = ImageFlag;
	samplingLayer->inputImage = inImage;
	samplingLayer->samplingFlag = samplingFlag;
	samplingLayer->scale = scale;
	samplingLayer->outputImage = (IMAGESIZE*)malloc(sizeof(IMAGESIZE));

	if(samplingLayer->outputImage == NULL)
	{
		return MOLLOC_ERR_IN_SET_SAMPLING_LAYER;
	}
	else
	{
		samplingLayer->outputImage->ImageNum = inImage->ImageNum;
		if(samplingFlag == FILLER)
		{
			samplingLayer->outputImage->height = (samplingLayer->inputImage->height + scale - 1)/scale;
			samplingLayer->outputImage->width = (samplingLayer->inputImage->width + scale - 1)/scale;
		}
		else
		{
			samplingLayer->outputImage->height = (samplingLayer->inputImage->height)/scale;
			samplingLayer->outputImage->width = (samplingLayer->inputImage->width)/scale;
		}

		samplingLayer->outputImage->Image = MallocImageBlock(samplingLayer->outputImage->ImageNum, samplingLayer->outputImage->height, samplingLayer->outputImage->width);
		samplingLayer->outputImage->dImage = MallocImageBlock(samplingLayer->outputImage->ImageNum, samplingLayer->outputImage->height, samplingLayer->outputImage->width);
		samplingLayer->outputImage->dNet = MallocImageBlock(samplingLayer->outputImage->ImageNum, samplingLayer->outputImage->height, samplingLayer->outputImage->width);

		if(samplingLayer->outputImage->Image == NULL || samplingLayer->outputImage->dNet== NULL || samplingLayer->outputImage->dImage == NULL)
		{
			return MOLLOC_ERR_IN_SET_SAMPLING_LAYER;
		}
	}
	*outImage = samplingLayer->outputImage;
	*layer = samplingLayer;
	return 0;
}

int SetAllConnectNet(ALLCONNECTNET** layer, int outsize, IMAGESIZE *inImage, int layerIndex, int inputsize, float* input, float** output, float* dinput, float** doutput)
{
	int i;
	ALLCONNECTNET* allConnectNet = (ALLCONNECTNET*)malloc(sizeof(ALLCONNECTNET));

	if(allConnectNet == NULL)
	{
		return MOLLOC_ERR_IN_SET_ALL_CONNECT_NET;
	}

	if(inImage != NULL)
	{
		allConnectNet->inputSize = inImage->ImageNum * inImage->width * inImage->height;
		allConnectNet->input = (float*)malloc(sizeof(float) * allConnectNet->inputSize);
		allConnectNet->dinput = (float*)malloc(sizeof(float) * allConnectNet->inputSize);
	}
	else if(input != NULL)
	{
		allConnectNet->inputSize = inputsize;
		allConnectNet->input = input;	
		allConnectNet->dinput = dinput;	
	}
	else
	{
		return MOLLOC_ERR_IN_SET_ALL_CONNECT_NET;
	}
	allConnectNet->inputImage = inImage;
	allConnectNet->outputSize= outsize;
	allConnectNet->output = (float*)malloc(sizeof(float) * allConnectNet->outputSize);
	allConnectNet->doutput = (float*)malloc(sizeof(float) * allConnectNet->outputSize);
	allConnectNet->net = (float*)malloc(sizeof(float) * allConnectNet->outputSize);

#ifdef TRAIN
	W[layerIndex] = (float*)malloc(sizeof(float) * allConnectNet->outputSize * allConnectNet->inputSize);
	B[layerIndex] = (float*)malloc(sizeof(float) * allConnectNet->outputSize);
	if(W[layerIndex] == NULL || B[layerIndex] == NULL)
	{
		//free(convnLayer);
		return MOLLOC_ERR_IN_SET_ALL_CONNECT_NET;
	}
	srand((unsigned int) time(NULL));
	for(i=0; i<allConnectNet->outputSize * allConnectNet->inputSize; i++)
	{
		W[layerIndex][i] = 1.0f*rand()/RAND_MAX - 0.5f;//产生(0,1)的随机数
	}
	for(i=0; i<allConnectNet->outputSize; i++)
	{
		B[layerIndex][i] = 0;
	}
#endif
	dW[layerIndex] = (float*)malloc(sizeof(float) * allConnectNet->outputSize * allConnectNet->inputSize);
	dB[layerIndex] = (float*)malloc(sizeof(float) * allConnectNet->outputSize);

	if(dW[layerIndex] == NULL || dB[layerIndex] == NULL)
	{
		return MOLLOC_ERR_IN_SET_ALL_CONNECT_NET;
	}

	for(i=0; i<allConnectNet->outputSize * allConnectNet->inputSize; i++)
	{
		dW[layerIndex][i] = 0.0f;
	}
	for(i=0; i<allConnectNet->outputSize; i++)
	{
		dB[layerIndex][i] = 0.0f;
	}


	allConnectNet->k = (float**)malloc(sizeof(float*) * allConnectNet->outputSize);
	allConnectNet->dk = (float**)malloc(sizeof(float*) * allConnectNet->outputSize);
	allConnectNet->b = B[layerIndex];
	allConnectNet->db = dB[layerIndex];



	if(allConnectNet->input == NULL || allConnectNet->output == NULL || allConnectNet->net == NULL 
		|| allConnectNet->k == NULL || allConnectNet->dk == NULL || allConnectNet->doutput == NULL)
	{
		return MOLLOC_ERR_IN_SET_ALL_CONNECT_NET;
	}
	else
	{
		int i=0;
		for(i=0; i<allConnectNet->outputSize; i++)
		{
			allConnectNet->k[i] = W[layerIndex]+i*allConnectNet->inputSize;
			allConnectNet->dk[i] = dW[layerIndex]+i*allConnectNet->inputSize;
		}
		*output = allConnectNet->output;
		*doutput= allConnectNet->doutput;
		*layer = allConnectNet;
		return 0;
	}

}



int SetSoftmaxLayer(CNNSOFTMAXLAYER** layer, int outsize, IMAGESIZE *inImage, int layerIndex, int inputsize, float* input, float** output, float* dinput, float** doutput)
{
	int i;
	CNNSOFTMAXLAYER* softmaxLayer = (CNNSOFTMAXLAYER*)malloc(sizeof(CNNSOFTMAXLAYER));

	if(softmaxLayer == NULL)
	{
		return MOLLOC_ERR_IN_SET_SOFTMAX_LAYER;
	}

	if(inImage != NULL)
	{
		softmaxLayer->inputSize = inImage->ImageNum * inImage->width * inImage->height;
		softmaxLayer->input = (float*)malloc(sizeof(float) * softmaxLayer->inputSize);
		softmaxLayer->dinput = (float*)malloc(sizeof(float) * softmaxLayer->inputSize);
	}
	else if(input != NULL)
	{
		softmaxLayer->inputSize = inputsize;
		softmaxLayer->input = input;	
		softmaxLayer->dinput = dinput;	
	}
	else
	{
		return MOLLOC_ERR_IN_SET_ALL_CONNECT_NET;
	}
	softmaxLayer->inputImage = inImage;
	softmaxLayer->outputSize= outsize;
	softmaxLayer->output = (float*)malloc(sizeof(float) * softmaxLayer->outputSize);
	softmaxLayer->doutput = (float*)malloc(sizeof(float) * softmaxLayer->outputSize);
	softmaxLayer->net = (float*)malloc(sizeof(float) * softmaxLayer->outputSize);

#ifdef TRAIN
	W[layerIndex] = (float*)malloc(sizeof(float) * softmaxLayer->outputSize * softmaxLayer->inputSize);
	B[layerIndex] = (float*)malloc(sizeof(float) * softmaxLayer->outputSize);
	if(W[layerIndex] == NULL || B[layerIndex] == NULL)
	{
		//free(convnLayer);
		return MOLLOC_ERR_IN_SET_SOFTMAX_LAYER;
	}
	srand((unsigned int) time(NULL));
	for(i=0; i<softmaxLayer->outputSize * softmaxLayer->inputSize; i++)
	{
		W[layerIndex][i] = 1.0f*rand()/RAND_MAX - 0.5f;//产生(0,1)的随机数
	}
	for(i=0; i<softmaxLayer->outputSize; i++)
	{
		B[layerIndex][i] = 0;
	}
#endif
	dW[layerIndex] = (float*)malloc(sizeof(float) * softmaxLayer->outputSize * softmaxLayer->inputSize);
	dB[layerIndex] = (float*)malloc(sizeof(float) * softmaxLayer->outputSize);

	if(dW[layerIndex] == NULL || dB[layerIndex] == NULL)
	{
		return MOLLOC_ERR_IN_SET_ALL_CONNECT_NET;
	}

	for(i=0; i<softmaxLayer->outputSize * softmaxLayer->inputSize; i++)
	{
		dW[layerIndex][i] = 0.0f;
	}
	for(i=0; i<softmaxLayer->outputSize; i++)
	{
		dB[layerIndex][i] = 0.0f;
	}


	softmaxLayer->k = (float**)malloc(sizeof(float*) * softmaxLayer->outputSize);
	softmaxLayer->dk = (float**)malloc(sizeof(float*) * softmaxLayer->outputSize);
	softmaxLayer->b = B[layerIndex];
	softmaxLayer->db = dB[layerIndex];



	if(softmaxLayer->input == NULL || softmaxLayer->output == NULL || softmaxLayer->net == NULL 
		|| softmaxLayer->k == NULL || softmaxLayer->dk == NULL || softmaxLayer->doutput == NULL)
	{
		return MOLLOC_ERR_IN_SET_SOFTMAX_LAYER;
	}
	else
	{
		int i=0;
		for(i=0; i<softmaxLayer->outputSize; i++)
		{
			softmaxLayer->k[i] = W[layerIndex]+i*softmaxLayer->inputSize;
			softmaxLayer->dk[i] = dW[layerIndex]+i*softmaxLayer->inputSize;
		}
		*output = softmaxLayer->output;
		*doutput= softmaxLayer->doutput;
		*layer = softmaxLayer;
		return 0;
	}

}


void DeleteConvnLayer(CNNCONVNLAYER* layer)
{
	
	if(layer == NULL)
	{
		return;
	}

	DeleteImageBlock(layer->outputImage->Image,layer->outputImage->ImageNum,layer->outputImage->height,layer->outputImage->width);
	DeleteImageBlock(layer->outputImage->dImage,layer->outputImage->ImageNum,layer->outputImage->height,layer->outputImage->width);
	DeleteImageBlock(layer->outputImage->dNet,layer->outputImage->ImageNum,layer->outputImage->height,layer->outputImage->width);
	free(layer->outputImage);
	DeleteParaBlock(layer->k,layer->outputImage->ImageNum,layer->inputImage->ImageNum,layer->kernelSize,layer->kernelSize);
	DeleteParaBlock(layer->dk,layer->outputImage->ImageNum,layer->inputImage->ImageNum,layer->kernelSize,layer->kernelSize);
	free(layer->db);
	free(layer->b);
	free(layer);
}
void DeleteSamplingnLayer(CNNSAMPLINGLAYER* layer)
{
	if(layer == NULL)
	{
		return;
	}
	DeleteImageBlock(layer->outputImage->Image,layer->outputImage->ImageNum,layer->outputImage->height,layer->outputImage->width);
	DeleteImageBlock(layer->outputImage->dImage,layer->outputImage->ImageNum,layer->outputImage->height,layer->outputImage->width);
	DeleteImageBlock(layer->outputImage->dNet,layer->outputImage->ImageNum,layer->outputImage->height,layer->outputImage->width);
	free(layer->outputImage);
	free(layer);
}
void DeleteAllConnectNet(ALLCONNECTNET* layer)
{
	if(layer == NULL)
	{
		return;
	}
	free(layer->input);
	free(layer->output);
	free(layer->net);
	free(layer->b);
	free(layer->db);
	free(layer->dinput);
	free(layer->doutput);

	free(layer->k);
	free(layer);
}

void DeleteSoftmaxLayer(CNNSOFTMAXLAYER* layer)
{
	if(layer == NULL)
	{
		return;
	}
	free(layer->input);
	free(layer->output);
	free(layer->net);
	free(layer->b);
	free(layer->db);
	free(layer->dinput);
	free(layer->doutput);

	free(layer->k);
	free(layer);
}


void DleteCnn(CNN* cnn)
{
	int i;
	for(i=0;i<cnn->depth;i++)
	{
		if(cnn->type[i] == CONVNLAYER)
		{
			DeleteConvnLayer((CNNCONVNLAYER*) cnn->layer[i]);
		}
		else if(cnn->type[i] == SAMPLINGLAYER)
		{
			DeleteSamplingnLayer((CNNSAMPLINGLAYER*) cnn->layer[i]);
		}
		else if(cnn->type[i] == ALLCONNECTLAYER)
		{
			DeleteAllConnectNet((ALLCONNECTNET*) cnn->layer[i]);
		}
		else if(cnn->type[i] == SOFTMAXLAYER)
		{
			DeleteSoftmaxLayer((CNNSOFTMAXLAYER*) cnn->layer[i]);
		}
		cnn->layer[i] = NULL;
		cnn->type[i] = 0;
#ifdef TRAIN
		if(W[i] != NULL) {free(W[i]);}
		if(B[i] != NULL) {free(B[i]);}
		if(dW[i] != NULL) {free(dW[i]);}
		if(dB[i] != NULL) {free(dB[i]);}
#endif
	}

	cnn->depth = 0;
}
