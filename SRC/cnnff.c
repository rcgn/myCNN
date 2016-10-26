#include "cnnff.h"

void GetSingleConvn(IMAGESIZE* inputImage, float***w, float b, float* output, int scale)
{
	int i;
	int height = inputImage->height;
	int width = inputImage->width;
	int imageNum = inputImage->ImageNum;
	for(i=0; i<height*width; i++)
	{
		output[i] = b;
	}

	for(i=0; i<imageNum; i++)
	{//在原output上累加
		//Convn(inputImage->Image[i], w[i], output, inputImage->height, inputImage->width, scale);
		Convn2(inputImage->Image[i], w[i], output, height, width, scale, scale,VALID);
	}
}

int GetConvnOutImage(CNNCONVNLAYER* layer)
{
	int inputImageWidth = layer->inputImage->width;
	int inputImageHeight = layer->inputImage->height;
	int scale = layer->kernelSize;
	int outLength = (inputImageWidth-scale/2*2) * (inputImageHeight-scale/2*2);//卷积后，有效区域总点数
	int outImageNum = layer->outputImage->ImageNum;//输出图像数
	int inputImageNum = layer->inputImage->ImageNum;//输入图像数
	int i,j,k,p=0;
	int outHeightStart,outHeightEnd,outWidthStart,outWidthEnd;
	float* net = (float*)malloc(sizeof(float)*inputImageWidth*inputImageHeight);//输入图像的加权和
	float* out = (float*)malloc(sizeof(float)*inputImageWidth*inputImageHeight);//经过激活函数变化后的net，即输出图像有效部分。

	if(net==NULL || out == NULL)
	{
		if(net != NULL) { free(net);}
		if(out != NULL) { free(out);}
		return MOLLOC_ERR_IN_GET_CONVN_OUT_IMAGE;
	}

	if(layer->ImageFlag == FILLER)
	{
		outHeightStart = scale/2*2;
		outHeightEnd = layer->outputImage->height - scale/2*2;
		outWidthStart = scale/2*2;
		outWidthEnd = layer->outputImage->width - scale/2*2;
	}
	else
	{
		outHeightStart = 0;
		outHeightEnd = layer->outputImage->height;
		outWidthStart = 0;
		outWidthEnd = layer->outputImage->width;
	}

	for(i=0; i<outImageNum; i++)
	{
		//initBlock((char*)net, sizeof(float)*inputImageWidth*inputImageHeight);
		GetSingleConvn(layer->inputImage, layer->k[i], layer->b[i], net, scale);
		ActivationFunction(net, out, outLength, layer->actFuncFlag);

		for(p=0,j=outHeightStart; j<outHeightEnd; j++)
		{
			for(k=outWidthStart; k<outWidthEnd; k++)
			{
				layer->outputImage->Image[i][j][k] = out[p++];
			}
		}
	}

	free(net);
	free(out);
	return 0;

}

void ActivationFunction(float* net, float* out ,int length, int actFuncFlag)
{
	if(actFuncFlag == SIGMOD)
	{
		Sigmod(net, out, length);
	}
	else if(actFuncFlag == RELU)
	{//暂时以sigmod代替
		Sigmod(net, out, length);
	}
	else if(actFuncFlag == LEAKY_RELU)
	{
		LeakyRelu(net, out, length);
	}
	else if(actFuncFlag == RELU)
	{
		Relu(net, out, length);
	}
	else
	{//默认使用sigmod函数
		Sigmod(net, out, length);
	}
}

float GetSamplingNum(float** inputImage, int samplingFlag, int inputImageWidth, int inputImageHeight, int scale, int yIndex, int xIndex)
{
	int i,j;
	int xStart = xIndex*scale;
	int xEnd = min((inputImageWidth), (xStart+scale));
	int yStart = yIndex*scale;
	int yEnd = min((inputImageHeight), (yStart+scale));
	float meanPoint = 0, maxPoint = 0;
	float result = 0.0f;

	if(xStart == xEnd || yStart == yEnd)
	{
		return 0.0f;
	}

	for(i=yStart; i<yEnd; i++)
	{
		for(j=xStart; j<xEnd; j++)
		{
			meanPoint += inputImage[i][j];
			maxPoint = max(maxPoint, inputImage[i][j]);
		}
	}
	meanPoint /=((yEnd-yStart)*(xEnd-xStart));

	if(samplingFlag == MEAN_TYPE)
	{
		result = meanPoint;
	}
	else if(samplingFlag == MAX_TYPE)
	{
		result = maxPoint; 
	}
	return result;
}

void GetSingleSampling(float** inputImage, float** outImage, int samplingFlag, 
	int inputImageWidth, int inputImageHeight, int outImageWidth, int outImageHeight, int scale)
{
	int i,j;

	for(i=0; i<outImageHeight; i++)
	{
		for(j=0; j<outImageWidth; j++)
		{
			outImage[i][j] = GetSamplingNum(inputImage, samplingFlag, inputImageWidth, inputImageHeight, scale, i, j);
		}
	}
}

void GetSamplingOutImage(CNNSAMPLINGLAYER* layer)
{
	int scale = layer->scale;
	int imageNum = layer->inputImage->ImageNum;
	int outImageHeight = layer->outputImage->height;
	int outImageWidth = layer->outputImage->width;
	int inputImageHeight = layer->inputImage->height;
	int inputImageWidth = layer->inputImage->width;
	int i;

	for(i=0;i<imageNum;i++)
	{
		GetSingleSampling(layer->inputImage->Image[i], layer->outputImage->Image[i], layer->samplingFlag, 
			inputImageWidth, inputImageHeight, outImageWidth, outImageHeight, scale);
	}
}


float* GetAllConnectNetResult(ALLCONNECTNET* layer)
{
	int i, j, k;
	int p=0;


	if(layer->inputImage != NULL)
	{
		int imageNum = layer->inputImage->ImageNum;
		int width = layer->inputImage->width;
		int height = layer->inputImage->height;

		for(i=0; i< imageNum; i++)
		{
			//matlab 先列后行
			for(k=0; k<width; k++)
			{
				for(j=0; j<height; j++)
				{
					layer->input[p++] = layer->inputImage->Image[i][j][k];
				}
			}

			//正常图像
			//for(j=0; j<height; j++)
			//{
			//	for(k=0; k<width; k++)
			//	{
			//		//layer->input[p++] = layer->inputImage->Image[i][j][k];
			//		layer->input[p++] = image[(i*height+j)*width+k];
			//	}
			//}
		}
	}

	GetNet(layer->k, layer->b, layer->input, layer->net, layer->inputSize, layer->outputSize);
	Sigmod(layer->net, layer->output, layer->outputSize);
	return layer->output;
}


float* GetSoftmaxLayerResult(CNNSOFTMAXLAYER* layer)
{
	int i, j, k;
	int p=0;


	if(layer->inputImage != NULL)
	{
		int imageNum = layer->inputImage->ImageNum;
		int width = layer->inputImage->width;
		int height = layer->inputImage->height;
		for(i=0; i< imageNum; i++)
		{
			//matlab 先列后行
			for(k=0; k<width; k++)
			{
				for(j=0; j<height; j++)
				{
					layer->input[p++] = layer->inputImage->Image[i][j][k];
				}
			}

			//正常图像
			//for(j=0; j<height; j++)
			//{
			//	for(k=0; k<width; k++)
			//	{
			//		//layer->input[p++] = layer->inputImage->Image[i][j][k];
			//		layer->input[p++] = image[(i*height+j)*width+k];
			//	}
			//}
		}
	}

	GetNet(layer->k, layer->b, layer->input, layer->net, layer->inputSize, layer->outputSize);
	Softmax(layer->net, layer->output, layer->outputSize);

	return layer->output;
}


float* CnnFf(CNN* cnn, float* x)
{
	int i,j,k;
	int p = 0;
	int result = 0;
	float *out = NULL;
	IMAGESIZE* inputImage = NULL;
	int imageHeight ,imageWidth;// 二维数组转为一维数组，减少算法时间

	if(cnn->type[0] == CONVNLAYER)
	{
		inputImage = ((CNNCONVNLAYER*)(cnn->layer[0]))->inputImage;
	}
	else if(cnn->type[0] == SAMPLINGLAYER)
	{
		inputImage = ((CNNSAMPLINGLAYER*)(cnn->layer[0]))->inputImage;
	}
	else if(cnn->type[0] == ALLCONNECTLAYER)
	{
		inputImage = ((ALLCONNECTNET*)(cnn->layer[0]))->inputImage;
	}
	else if(cnn->type[0] == SOFTMAXLAYER)
	{
		inputImage = ((CNNSOFTMAXLAYER*)(cnn->layer[0]))->inputImage;
	}
	else
	{
		return NULL;
	}

	imageHeight = inputImage->height;
	imageWidth = inputImage->width;
	for(p=0,i=0; i< inputImage->ImageNum; i++)
	{
		//正常图像
		//for(j=0; j< imageHeight; j++)
		//{
		//	for(k=0; k< imageWidth; k++)
		//	{
		//		if(x[p]!= 0)
		//		{
		//			x[p] = x[p] ;
		//		}
		//		inputImage->Image[i][j][k] = x[p++];
		//	}
		//}

		//matlab训练时图像被转置
		for(j=0; j< imageHeight; j++)
		{
			for(k=0; k< imageWidth; k++)
			{
				inputImage->Image[i][k][j] = x[p++];
			}
		}
	}

	for(i=0; i<cnn->depth-1; i++)
	{
		if(cnn->type[i] == CONVNLAYER)
		{
			if(0 != GetConvnOutImage((CNNCONVNLAYER*) cnn->layer[i]))
			{
				return NULL;
			}
		}
		else if(cnn->type[i] == SAMPLINGLAYER)
		{
			GetSamplingOutImage((CNNSAMPLINGLAYER*) cnn->layer[i]);
		}
		else if(cnn->type[i] == ALLCONNECTLAYER)
		{
			GetAllConnectNetResult((ALLCONNECTNET*) cnn->layer[i]);
		}
		else if(cnn->type[i] == SOFTMAXLAYER)
		{
			GetSoftmaxLayerResult((CNNSOFTMAXLAYER*) cnn->layer[i]);
		}
		else
		{
			return NULL;
		}
	}
	if(cnn->type[i] == ALLCONNECTLAYER)
	{
		out = GetAllConnectNetResult((ALLCONNECTNET*) cnn->layer[i]);
		return out;
	}
	else if(cnn->type[i] == SOFTMAXLAYER)
	{
		out = GetSoftmaxLayerResult((CNNSOFTMAXLAYER*) cnn->layer[i]);
		return out;
	}
	else
	{
		return NULL;
	}
}

