#include "cnnBp.h"


float BpLossFunction(int label[], float* out, int outsize)
{
	int i;
	float loss = 0.0;

	for(i=0; i< outsize; i++)
	{
		loss += (out[i]-label[i])*(out[i]-label[i]);
	}
	loss /= (2* outsize);

	return loss;
}

float SoftMaxLossFunction(CNN* cnn, int label[], float* out)
{
	int i, j;
	float loss = 0.0;
	float regularLoss = 0.0;
	CNNSOFTMAXLAYER* softmaxLayer = (CNNSOFTMAXLAYER*)(cnn->layer[cnn->depth-1]);

	for(i=0; i<cnn->outSize; i++)
	{
		if(label[i] == 1)
		{
			loss = -log(out[i]);
		}
	}
	for(i=0; i<cnn->outSize; i++)
	{
		for(j=0; j< softmaxLayer->inputSize; j++)
		{
			regularLoss += softmaxLayer->k[i][j] * softmaxLayer->k[i][j];
		}
	}
	regularLoss *= (0.5*gSoftmaxLambda);
	loss += regularLoss;

	return loss;
}


float LossFunction(CNN* cnn, int label[], float* out)
{
	if(cnn->lossType == BP_NN)
	{
		return BpLossFunction(label, out, cnn->outSize);
	}
	else if(cnn->lossType == SOFTMAX)
	{
		return SoftMaxLossFunction(cnn, label, out);
	}
	else
	{
		return 0.0;
	}
}

int UpdateAllConnectNetdInput(ALLCONNECTNET* layer)
{
	int i,j,k;
	int inputSize = layer->inputSize;
	int outputSize= layer->outputSize;

	float* doutput = layer->doutput;
	float* output = layer->output;
	float* dinput = layer->dinput;
	//float* dImage = layer->inputImage->dImage[0][0];

	for(i=0; i<inputSize; i++)
	{
		dinput[i] = 0.0;
		for(j=0; j<outputSize; j++)
		{
			dinput[i] += doutput[j] * (output[j]*(1-output[j])) * layer->k[j][i];
//////////////////////////////////////////////////test/////////////////////////////////////
			//float test = output[j]*(1-output[j]);
			//if(test < 0.00000000000000001)
			//{
			//	test = 0.01;
			//}
			//dinput[i] +=  doutput[j] *test * layer->k[j][i];
/////////////////////////////////////////////////test end//////////////////////////////////
		}
	}

	if(layer->inputImage != NULL)
	{
		int p=0;
		int width = layer->inputImage->width;
		int height = layer->inputImage->height;
		int imageNum = layer->inputImage->ImageNum;
		for(i=0; i< imageNum; i++)
		{
			//matlab 先列后行
			for(k=0; k<width; k++)
			//for(j=0; j<height; j++)
			{
				for(j=0; j<height; j++)
				//for(k=0; k<width; k++)
				{
					layer->inputImage->dImage[i][j][k] = layer->dinput[p++];
				}
			}
		}
	}
	return 0;
}


int UpdateSoftmaxLayerInput(CNNSOFTMAXLAYER* layer)
{
	int i,j,k;
	int inputSize = layer->inputSize;
	int outputSize= layer->outputSize;

	float* doutput = layer->doutput;
	float* output = layer->output;
	float* dinput = layer->dinput;
	//float* dImage = layer->inputImage->dImage[0][0];
	
	for(i=0; i<inputSize; i++)
	{
		dinput[i] = 0.0;
		for(j=0; j<outputSize; j++)
		{
			dinput[i] += doutput[j] * layer->k[j][i];
			//////////////////////////////////////////////////test/////////////////////////////////////
			//float test = output[j]*(1-output[j]);
			//if(test < 0.00000000000000001)
			//{
			//	test = 0.01;
			//}
			//dinput[i] +=  doutput[j] *test * layer->k[j][i];
			/////////////////////////////////////////////////test end//////////////////////////////////
		}
	}

	if(layer->inputImage != NULL)
	{
		int p=0;
		int width = layer->inputImage->width;
		int height = layer->inputImage->height;
		int imageNum = layer->inputImage->ImageNum;
		for(i=0; i< imageNum; i++)
		{
			//matlab 先列后行
			for(k=0; k<width; k++)
				//for(j=0; j<height; j++)
			{
				for(j=0; j<height; j++)
					//for(k=0; k<width; k++)
				{
					layer->inputImage->dImage[i][j][k] = layer->dinput[p++];
				}
			}
		}
	}
	return 0;
}



int UpdateCovnLayerdInput(CNNCONVNLAYER* layer)
{
	int i,j,k;
	int p,q;
	float*** dNet = layer->outputImage->dNet;
	float*** dImage = layer->outputImage->dImage;
	float*** image = layer->outputImage->Image;
	float*** indImage = layer->inputImage->dImage;
	int outNum = layer->outputImage->ImageNum;
	int outHeight = layer->outputImage->height;
	int outWidth = layer->outputImage->width;
	int inNum = layer->inputImage->ImageNum;
	int inHeight = layer->inputImage->height;
	int inWidth = layer->inputImage->width;
	float* inImageBlock = (float*)malloc(inHeight * inWidth * sizeof(float));
	float** w180 = MallocSingleImage(layer->kernelSize, layer->kernelSize);

	if(inImageBlock == NULL || w180 == NULL)
	{
		if(inImageBlock != NULL) free(inImageBlock);
		if(w180 != NULL) DeleteSingleImage(w180);
		return MOLLOC_ERR_IN_UPDATE_COVN_LAYER_DINPUT;
	}

	//for(i=0; i<layer->outputImage->height; i++)
	//{
	//	dNet[i] = imageBlock + i*layer->outputImage->width;
	//}

	for(i=0; i<outNum; i++)
	{
		for(j=0; j<outHeight; j++)
		{
			for(k=0; k<outWidth; k++)
			{
				switch(layer->actFuncFlag)
				{
				case SIGMOD:
					{
						dNet[i][j][k] = image[i][j][k] * (1 - image[i][j][k]) * dImage[i][j][k];
						break;
					}
				case LEAKY_RELU:
					{
						if(image[i][j][k]>=0)
						{
							dNet[i][j][k] = dImage[i][j][k];
						}
						else
						{
							dNet[i][j][k] = LEAKY_RELU_RATIO*dImage[i][j][k];
						}
						break;
					}
				case RELU:
					{
						if(image[i][j][k]>=0)
						{
							dNet[i][j][k] = dImage[i][j][k];
						}
						else
						{
							dNet[i][j][k] = 0;
						}
						break;
					}
				}
				
			}
		}
	}


	for(i=0; i< inNum ;i++)
	{
		memset(inImageBlock, 0, inHeight * inWidth * sizeof(float));
		for(j=0; j <outNum; j++)
		{
			MatrixRotation180(layer->k[j][i], layer->kernelSize, layer->kernelSize, w180);
			Convn2(dNet[j], w180, inImageBlock, outHeight, outWidth, layer->kernelSize, layer->kernelSize, FULL);
		}			
		for(k=0,p=0; p<inHeight; p++)
		{
			for(q=0; q<inWidth; q++)
			{
				indImage[i][p][q] = inImageBlock[k++];
			}
		}
	}


	free(inImageBlock);
	DeleteSingleImage(w180);
	return 0;
}

int UpdateSamplingLayerdInput(CNNSAMPLINGLAYER* layer)
{
	int i,j,k;
	int p,q;

	//此处看似是一个五层循环，实际上是遍历所有图像，没有重复。
	//池化层（抽样层）的误差传递为直接传递，不做误差分配。
	for(i=0; i<layer->outputImage->ImageNum; i++)
	{
		for(j=0; j<layer->outputImage->height; j++)
		{
			for(k=0; k<layer->outputImage->width; k++)
			{
				for(p = 0; p<layer->scale; p++)
				{
					for(q = 0; q<layer->scale; q++)
					{
						layer->inputImage->dImage[i][j*layer->scale+p][k*layer->scale+q] = layer->outputImage->dImage[i][j][k];
					}
				}
			}
		}
	}
	return 0;
}

int UpdateConvnLayerdK(CNNCONVNLAYER* layer)
{
	int i,j,k;
	int p,q;
	float*** dNet = layer->outputImage->dNet;//dNet已在UpdateConvnLayerdInput函数中计算得到。

	int outNum = layer->outputImage->ImageNum;
	int outHeight = layer->outputImage->height;
	int outWidth = layer->outputImage->width;
	int inNum = layer->inputImage->ImageNum;
	int inHeight = layer->inputImage->height;
	int inWidth = layer->inputImage->width;
	int wHeight = layer->kernelSize;
	int wWidth = layer->kernelSize;
	float*** inImage = layer->inputImage->Image;

	float* dwBlock = (float*)malloc(wHeight * wWidth * sizeof(float));
	float** inImage180 = MallocSingleImage(inHeight, inWidth);

	if(dwBlock == NULL || inImage180 == NULL)
	{
		if(dwBlock != NULL) free(dwBlock);

		return MOLLOC_ERR_IN_UPDATE_CONVN_LAYER_DK;;
	}

	//for(i=0; i<layer->outputImage->height; i++)
	//{
	//	dNet[i] = imageBlock + i*layer->outputImage->width;
	//}

	//for(i=0; i<outNum; i++)
	//{
	//	for(j=0; j<outHeight; j++)
	//	{
	//		for(k=0; k<outWidth; k++)
	//		{
	//			dNet[i][j][k] = image[i][j][k] * (1 - image[i][j][k]) * dImage[i][j][k];
	//		}
	//	}
	//}


	for(i=0; i< outNum ;i++)
	{
		for(j=0; j<inNum; j++)
		{
			memset(dwBlock, 0, wHeight * wWidth * sizeof(float));

			MatrixRotation180(inImage[j], inHeight, inWidth, inImage180);
			Convn2(inImage180, dNet[i], dwBlock, inHeight, inWidth, outHeight, outWidth, VALID);

			for(k=0,p=0; p<wHeight; p++)
			{
				for(q=0; q<wWidth; q++)
				{
					layer->dk[i][j][p][q] += dwBlock[k++];//单张钞票权值梯度先累加，够一个batch后再添加到权值中
				}
			}
		}

		for(p=0; p<outHeight; p++ )
		{
			for(q=0; q<outWidth; q++)
			{
				layer->db[i] += layer->outputImage->dNet[i][p][q];//单张钞票权值梯度先累加，够一个batch后再添加到权值中
			}
		}
	}


	free(dwBlock);
	DeleteSingleImage(inImage180);
	return 0;
}

int UpdateAllConnectNetdK(ALLCONNECTNET* layer)
{
	int i,j;
	int outNum = layer->outputSize;
	int inNum = layer->inputSize;
	float* out = layer->output;
	float* in = layer->input;
	float* dOutput = layer->doutput;
	float** dk = layer->dk;
	float* db = layer->db;
	float dnet;

	for(i=0; i<outNum; i++)
	{
		dnet = dOutput[i]*out[i]*(1-out[i]);
//////////////////////////////////////////////////test/////////////////////////////////////
		//float test = out[i]*(1-out[i]);
		////float test2 = dOutput[i]*out[i]*(1-out[i]);
		//if(test < 0.00000000000000001)
		//{
		//	test = 0.0001;
		//}
		//dnet = dOutput[i] *test;
/////////////////////////////////////////////////test end//////////////////////////////////
		for(j=0; j<inNum; j++)
		{
			dk[i][j] += in[j] * dnet;
		}

		db[i] += dnet;
	}

	return 0;
}


int UpdateSoftmaxLayerdK(CNNSOFTMAXLAYER* layer)
{
	int i,j;
	int outNum = layer->outputSize;
	int inNum = layer->inputSize;
	float* out = layer->output;
	float* in = layer->input;
	float* dOutput = layer->doutput;
	float** dk = layer->dk;
	float* db = layer->db;
	//float dnet;

	for(i=0; i<outNum; i++)
	{
		//////////////////////////////////////////////////test/////////////////////////////////////
		//float test = out[i]*(1-out[i]);
		////float test2 = dOutput[i]*out[i]*(1-out[i]);
		//if(test < 0.00000000000000001)
		//{
		//	test = 0.0001;
		//}
		//dnet = dOutput[i] *test;
		/////////////////////////////////////////////////test end//////////////////////////////////
		for(j=0; j<inNum; j++)
		{
			dk[i][j] += in[j] * dOutput[i];
		}

		db[i] += dOutput[i];
	}

	return 0;
}


int UpdateInput(CNN* cnn)
{
	int i=0;
	int rtn = 0;

	for(i=cnn->depth-1 ;i>=0; i--)
	{
		if(cnn->type[i] == CONVNLAYER)
		{
			rtn = UpdateCovnLayerdInput((CNNCONVNLAYER*)cnn->layer[i]);
			if(rtn != 0)
			{
				break;
			}
		}
		else if(cnn->type[i] == ALLCONNECTLAYER)
		{
			rtn = UpdateAllConnectNetdInput((ALLCONNECTNET*)cnn->layer[i]);
			if(rtn != 0)
			{
				break;
			}
		}
		else if (cnn->type[i] == SAMPLINGLAYER)
		{
			rtn = UpdateSamplingLayerdInput((CNNSAMPLINGLAYER*)cnn->layer[i]);
			if(rtn != 0)
			{
				break;
			}
		}
		else if(cnn->type[i] == SOFTMAXLAYER)
		{
			rtn = UpdateSoftmaxLayerInput((CNNSOFTMAXLAYER*)cnn->layer[i]);
			if(rtn != 0)
			{
				break;
			}
		}
		else
		{
			return ERR_LAYER_TYPE;
		}
	}
	return rtn;
}


int UpdatedK(CNN* cnn)
{
	int i=0;
	int rtn = 0;

	for(i=cnn->depth-1 ;i>=0; i--)
	{
		if(cnn->type[i] == CONVNLAYER)
		{
			rtn = UpdateConvnLayerdK((CNNCONVNLAYER*)cnn->layer[i]);
			if(rtn != 0)
			{
				break;
			}
		}
		else if(cnn->type[i] == ALLCONNECTLAYER)
		{
			rtn = UpdateAllConnectNetdK((ALLCONNECTNET*)cnn->layer[i]);
			if(rtn != 0)
			{
				break;
			}
		}
		else if (cnn->type[i] == SAMPLINGLAYER)
		{
			continue;
		}
		else if(cnn->type[i] == SOFTMAXLAYER)
		{
			rtn = UpdateSoftmaxLayerdK((CNNSOFTMAXLAYER*)cnn->layer[i]);
			if(rtn != 0)
			{
				break;
			}
		}
		else
		{
			return ERR_LAYER_TYPE;
		}
	}

	return rtn;
}
