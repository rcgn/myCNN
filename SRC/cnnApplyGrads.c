
#include "cnnApplyGrads.h"


void AdjustConvnLayerPara(CNNCONVNLAYER* layer, int trainNum, float alpha)
{
	int i,j,p,q;
	int outNum = layer->outputImage->ImageNum;
	int inNum = layer->inputImage->ImageNum;
	int kernelSize = layer->kernelSize;
	float**** k = layer->k;
	float* b = layer->b;
	float**** dk= layer->dk;
	float* db = layer->db;

	for(i=0; i<outNum; i++)
	{
		for(j=0; j<inNum; j++)
		{
			for(p=0; p<kernelSize; p++)
			{
				for(q=0; q<kernelSize; q++)
				{
					k[i][j][p][q] -= alpha * dk[i][j][p][q] / trainNum;
					dk[i][j][p][q] = 0.0f;
				}
			}
		}
		b[i] -= alpha * db[i]/trainNum;
		db[i] = 0.0f;
	}

}

void AdjustAllConnectLayerPara(ALLCONNECTNET* layer, int trainNum, float alpha)
{
	int i,j;
	int outNum = layer->outputSize;
	int inNum = layer->inputSize;
	float** k = layer->k;
	float** dk= layer->dk;
	float* b = layer->b;

	float* db = layer->db;

	for(i=0; i<outNum; i++)
	{
		for(j=0; j<inNum; j++)
		{
			k[i][j] -= alpha * dk[i][j] / trainNum;
			dk[i][j] = 0.0f;
		}
		b[i] -= alpha * db[i]/trainNum;
		db[i] = 0.0f;
	}
	//{
	//	FILE* ini;
	//	ini = fopen("a.csv","a+");
	//	for(i=0;i<9;i++)
	//	{
	//		fprintf(ini,"%f,",k[0][i]);
	//	}
	//	fprintf(ini,"\n");

	//	fclose(ini);
	//}
}


void AdjustSoftmaxLayerPara(CNNSOFTMAXLAYER* layer, int trainNum, float alpha)
{
	int i,j;
	int outNum = layer->outputSize;
	int inNum = layer->inputSize;
	float** k = layer->k;
	float** dk= layer->dk;
	float* b = layer->b;

	float* db = layer->db;

	for(i=0; i<outNum; i++)
	{
		for(j=0; j<inNum; j++)
		{
			k[i][j] -= alpha * dk[i][j] / trainNum;
			dk[i][j] = 0.0f;
		}
		b[i] -= alpha * db[i]/trainNum;
		db[i] = 0.0f;
	}
	//{
	//	FILE* ini;
	//	ini = fopen("a.csv","a+");
	//	for(i=0;i<9;i++)
	//	{
	//		fprintf(ini,"%f,",k[0][i]);
	//	}
	//	fprintf(ini,"\n");

	//	fclose(ini);
	//}
}

int AdjustCnnPara(CNN*cnn, int trainNum, float alpha)
{
	int i;

	if(trainNum == 0 ){ return EMPTY_SAMPLE;}
	if(cnn == NULL){  return EMPTY_CNN;}

	for(i=0; i<cnn->depth; i++)
	{
		if(cnn->type[i] == CONVNLAYER)
		{
			AdjustConvnLayerPara((CNNCONVNLAYER*)cnn->layer[i], trainNum, alpha);
		}
		else if (cnn->type[i] == SAMPLINGLAYER)
		{
			continue;
		}
		else if (cnn->type[i] == ALLCONNECTLAYER)
		{
			AdjustAllConnectLayerPara((ALLCONNECTNET*)cnn->layer[i], trainNum, alpha);
		}
		else if(cnn->type[i] == SOFTMAXLAYER)
		{
			AdjustSoftmaxLayerPara((CNNSOFTMAXLAYER*)cnn->layer[i], trainNum, alpha);
		}
		else
		{
			return ERR_LAYER_TYPE;
		}
	}
	return 0;
}