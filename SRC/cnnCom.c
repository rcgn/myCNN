#include "cnnCom.h"
float gSoftmaxLambda = SOFTMAX_LAMBDA;

void DeleteImageBlock(float*** block, int outImageNum, int height, int width)
{
	free(block[0][0]);
	free(block[0]);
	free(block);
}


float*** MallocImageBlock(int outImageNum, int height, int width)
{
	int i,j;
	float*** result = (float***)malloc(sizeof(float**)*outImageNum);
	float** result_i=(float**)malloc(sizeof(float*)*outImageNum*height);
	float*  result_i_j = (float*)malloc(sizeof(float)*outImageNum*height*width);

	if(result == NULL || result_i == NULL || result_i_j == NULL)
	{
		if(result != NULL) {free(result);}
		if(result_i != NULL) {free(result_i);}
		if(result_i_j != NULL) {free(result_i_j);}
		return NULL;
	}
	memset(result, 0, sizeof(float**)*outImageNum);
	memset(result_i, 0, sizeof(float*)*outImageNum*height);
	memset(result_i_j, 0, sizeof(float)*outImageNum*height*width);

	if( outImageNum<=0 || height<=0 || width<=0)
	{
		return NULL;
	}

	for(i=0;i<outImageNum;i++)
	{
		result[i] = result_i + i*height;
		for(j=0;j<height;j++)
		{
			result[i][j] = result_i_j + j*width + i*height*width;
		}
	}

	//if(result = (float***)malloc(sizeof(float**)*outImageNum))
	//{
	//	for(i=0;i<outImageNum;i++)
	//	{
	//		if(result[i] =(float**)malloc(sizeof(float*)*height))
	//		{
	//			for(j=0;j<height;j++)
	//			{
	//				if(result[i][j] = (float*)malloc(sizeof(float)*width))
	//				{
	//					initBlock((char*)result[i][j], sizeof(float)*width);//全部初始化为0
	//				}
	//				else
	//				{
	//					DeleteImageBlock(result,outImageNum,height,width);
	//					return NULL;
	//				}
	//			}
	//		}
	//		else
	//		{
	//			DeleteImageBlock(result,outImageNum,height,width);
	//			return NULL;
	//		}
	//
	//	}
	//}
	//else
	//{
	//	return NULL;
	//}

	return result;
}



void DeleteParaBlock(float**** block, int outImageNum, int inputImageNum, int height, int width)
{
	free(block[0][0][0]);
	free(block[0][0]);
	free(block[0]);
	free(block);
}


float**** MallocParaBlock(int outImageNum, int inputImageNum, int height, int width, float* w)
{
	int i,j,k;
	float**** result = (float****)malloc(outImageNum * sizeof(float***));
	float*** result_i = (float***)malloc(outImageNum * inputImageNum * sizeof(float**));
	float** result_i_h = (float**)malloc(outImageNum * inputImageNum * height * sizeof(float*));

	if(result == NULL || result_i == NULL || result_i_h == NULL)
	{
		if(result != NULL){free(result);}
		if(result_i != NULL){free(result_i);}
		if(result_i_h != NULL){free(result_i_h);}
		return NULL;
	}

	if( outImageNum<=0 || inputImageNum<=0 || height<=0 || width<=0)
	{
		return NULL;
	}

	memset(result, 0, sizeof(float**)*outImageNum);
	memset(result_i, 0, sizeof(float*)*outImageNum*inputImageNum);
	memset(result_i_h, 0, sizeof(float)*outImageNum*inputImageNum*height);

	for(i=0; i<outImageNum; i++)
	{
		result[i] = result_i + i*inputImageNum;
		for(j=0; j<inputImageNum; j++)
		{
			result[i][j] = result_i_h + i*inputImageNum*height + j*height;
			for(k=0; k<height; k++)
			{
				result[i][j][k] = w + i*inputImageNum*height*width + j*height*width + k*width;
			}
		}
	}

	//if(result = (float****)malloc(sizeof(float***)*outImageNum))
	//{
	//	for(i=0;i<outImageNum;i++)
	//	{
	//		if(result[i] =(float***)malloc(sizeof(float**)*inputImageNum))
	//		{
	//			for(j=0;j<inputImageNum;j++)
	//			{
	//				if(result[i][j] = (float**)malloc(sizeof(float*)*height))
	//				{
	//					for(k=0;k<height;k++)
	//					{
	//						result[i][j][k] = w+i*inputImageNum*width*height+j*width*height+k*width;
	//					}
	//				}
	//				else
	//				{
	//					DeleteParaBlock(result,outImageNum,inputImageNum,height,width);
	//					return NULL;
	//				}
	//			}
	//		}
	//		else
	//		{
	//			DeleteParaBlock(result,outImageNum,inputImageNum,height,width);
	//			return NULL;
	//		}

	//	}
	//}
	//else
	//{
	//	return NULL;
	//}

	return result;
}


void DeleteSingleImage(float**matrix)
{
	if(matrix != NULL)
	{
		if(matrix[0] != NULL)
		{
			free(matrix[0]);//由于在MallocSingleImage函数中matrix第二维的空间是一次性分配的，所以只释放一次。
		}
		free(matrix);
	}
}

float** MallocSingleImage(int height, int width)
{
	float** result = NULL;
	int i;
	float* block = NULL;

	block = (float*)malloc(height*width*sizeof(float));
	result = (float**)malloc(height*sizeof(float*));
	if(block == NULL || result == NULL)
	{
		if(block != NULL) free(block);
		if(result != NULL) free(block);
		return NULL;
	}
	initBlock((char*)block,height*width*sizeof(float));
	for(i=0; i<height; i++)
	{
		result[i] = block+i*width;
	}

	return result;
}


void Sigmod(float* net, float* output ,int length)
{
	int i=0;
	for(i=0; i<length; i++)
	{
		output[i] = (float)(1.0 / (1+exp(-net[i])));
	}
}

void Softmax(float* net, float* output ,int length)
{
	int i=0;
	float outputSum = 0.0;

	for(i=0; i<length; i++)
	{
		output[i] = exp(net[i]);
		outputSum += output[i];
	}
	if(abs(outputSum) < 0.000001)
	{
		//TODO
	}
	else
	{
		for(i=0; i<length; i++)
		{
			output[i] /= outputSum;
		}
	}

}

void LeakyRelu(float* net, float* output ,int length)
{
	int i=0;
	for(i=0; i<length; i++)
	{
		output[i] = max(net[i],LEAKY_RELU_RATIO*net[i]);
	}
}

void Relu(float* net, float* output ,int length)
{
	int i=0;
	for(i=0; i<length; i++)
	{
		output[i] = max(net[i],0);
	}
}

void GetNet(float** w, float* b, float* input, float* net, int inputSize, int outputSize)
{
	int i,j;

	for(i=0; i<outputSize; i++)
	{
		float temp = 0.0;
		for(j=0; j<inputSize; j++)
		{
			temp += w[i][j]*input[j];
		}
		net[i] = temp + b[i];
	}
}

void Convn(float** inputImage, float** w, float* output, int imageHeight, int imageWidth, int scale)
{
	int i=0,j=0,k=0;
	int p=0,q=0;

	for(i=scale/2; i< imageHeight-scale/2; i++)
	{
		for(j=scale/2; j< imageHeight-scale/2; j++)
		{
			//卷积和180度翻转
			//for(p=0;p<scale;p++)
			//{
			//	for(q=0;q<scale;q++)
			//	{
			//		output[k] += inputImage[i-scale/2+p][j-scale/2+q]*w[p][q];
			//	}
			//}

			for(p=0;p<scale;p++)
			{
				for(q=0;q<scale;q++)
				{
					output[k] += inputImage[i-scale/2+p][j-scale/2+q]*w[scale-p-1][scale-q-1];
				}
			}
			k = k+1;
		}
	}
}

void initBlock(char* block, int size)
{
	int i = 0;
	for(i=0;i<size;i++)
	{
		block[i] = 0x00;
	}
}

void Convn2(float** matrix1, float** matrix2, float* output, int height1, int width1, int height2, int width2, char flag)
{
	int i=0,j=0;
	int p=0,q=0;
	int k=0;
	int startHeight = 0;
	int startWidth = 0;
	int endHeight = 0;
	int endWidth = 0;

	if(flag == FULL)
	{
		startHeight = 1 - height2;
		startWidth = 1 - width2;
		endHeight = height1;
		endWidth = width1;

	}
	else if(flag == VALID)
	{
		startHeight = 0;
		startWidth = 0;
		endHeight = height1 - height2 + 1;
		endWidth = width1 - width2 + 1;
	}

	for(i=startHeight; i<endHeight; i++)
	{
		for(j=startWidth; j<endWidth; j++)
		{
			int pStart = max(0,-i);
			int pEnd = min(height2,height1-i);
			int qStart = max(0,-j);
			int qEnd = min(width2,width1-j);

			for(p=pStart; p<pEnd; p++)
			{
				for(q=qStart; q<qEnd; q++)
				{
					output[k] += matrix1[i+p][j+q] * matrix2[height2-p-1][width2-q-1];//matrix2要做180度旋转
				}
			}
			k++;
		}
	}
}

//矩阵转置
int MatrixTranspose(float** matrix, int height, int width, float** resultMatrix)
{
	int i,j;

	//矩阵空间检测，避免内存错误，程序崩溃
	if(matrix == NULL || resultMatrix == NULL) return MATRIX_TRANSPOSE_ERR;

	for(i=0; i<height; i++)
	{
		if(matrix[i] == NULL) return MATRIX_TRANSPOSE_ERR;
	}
	for(j=0; j<width; j++)
	{
		if(resultMatrix[j] == NULL) return MATRIX_TRANSPOSE_ERR;
	}

	//矩阵转置
	for(i=0; i<height; i++)
	{
		for(j=0; j<width; j++)
		{
			resultMatrix[j][i] = matrix[i][j];
		}
	}

	return 0;
}

int MatrixRotation180(float** matrix, int height, int width, float** resultMatrix)
{
	int i,j;

	//矩阵空间检测，避免内存错误，程序崩溃
	if(matrix == NULL || resultMatrix == NULL) return MATRIX_ROTATION_180_ERR;

	for(i=0; i<height; i++)
	{
		if(matrix[i] == NULL || resultMatrix[i] == NULL) return MATRIX_ROTATION_180_ERR;
	}

	for(i=0; i<height; i++)
	{
		for(j=0; j<width; j++)
		{
			resultMatrix[i][j] = matrix[height-i-1][width-j-1];
		}
	}

	return 0;
}