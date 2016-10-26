#include "cnnTrain.h"

//#define TIME_TEST
#ifdef TIME_TEST
#include <time.h>
#include <Windows.h>
#pragma comment( lib,"winmm.lib" )
#endif

int CnnTrain(CNN* cnn, float** X, int** Y, int imageNum, TRANPARA* trainPara)
{
	int i,j;
	int trainNum = 0;
	int batchNum = 0;
	int rtn = 0;
	float* losses = (float*)malloc((imageNum*trainPara->numepochs/trainPara->batchSize+1)*sizeof(float));

	gSoftmaxLambda = SOFTMAX_LAMBDA / trainPara->batchSize;

	if(losses == NULL){return MOLLOC_ERR_IN_CNN_TRAIN;}
	else{memset(losses,0,(imageNum*trainPara->numepochs/trainPara->batchSize+1)*sizeof(float));}


	for(i=0; i<trainPara->numepochs; i++)
	{
		for(j=0; j<imageNum; j++)
		{
			rtn = SingleImageTrain(cnn, X[j], Y[j],&losses[batchNum]);
			trainNum ++;
			if(trainNum == trainPara->batchSize)
			{
				losses[batchNum] /= trainNum;
				if(batchNum > 0)
				{
					losses[batchNum] = losses[batchNum-1]*0.99f+losses[batchNum]*0.01f;
				}
				printf("%d-%d  Loss: %f\n",i,batchNum,losses[batchNum]);
				AdjustCnnPara(cnn,trainNum,trainPara->alpha);
				batchNum ++;
				trainNum = 0;	
			}
		}
	}
	if(trainNum != 0)//如果结尾还剩余小于batch的数据没有调整上去，补充调整
	{
		losses[batchNum] /= trainNum;
		if(batchNum > 0)
		{
			losses[batchNum] = losses[batchNum-1]*0.99f+losses[batchNum]*0.01f;
		}
		printf("%d-%d  Loss: %f\n",i,batchNum,losses[batchNum]);
		AdjustCnnPara(cnn,trainNum,trainPara->alpha);
		batchNum ++;
		trainNum = 0;
	}

	{
		FILE* ini;
		ini = fopen("losses.csv","w");
		for(i=0; i<imageNum*trainPara->numepochs/trainPara->batchSize; i++)
		{
			fprintf(ini,"%f\n",losses[i]);
		}
		fclose(ini);
	}
	free(losses);
	return rtn;
}

int SingleImageTrain(CNN* cnn, float* X, int* Y, float* loss)
{
	int i;
	float* out= NULL;
	float* dout = NULL;
	int rtn=0;
#ifdef TIME_TEST	
	DWORD time0,time1,time2,time3,time4;
#endif
	int j=0;
	 
	if(cnn->lossType == BP_NN)
	{
		dout = ((ALLCONNECTNET*)(cnn->layer[cnn->depth-1]))->doutput;
	}
	else if(cnn->lossType == SOFTMAX)
	{
		dout = ((CNNSOFTMAXLAYER*)(cnn->layer[cnn->depth-1]))->doutput;
	}
	else
	{
		return EMPTY_OUTPUT;
	}
#ifdef TIME_TEST	
	time0 =timeGetTime();
	for(j=0;j<1000;j++)
	{
#endif
		out = CnnFf(cnn, X);
		if(out == NULL)
		{
			return EMPTY_OUTPUT;
		}

		*loss += LossFunction(cnn, Y, out);
		for(i=0; i<cnn->outSize; i++)
		{
			dout[i] = out[i]-Y[i];
		}
#ifdef TIME_TEST
	}

	time1 = timeGetTime();
	for(j=0;j<1000;j++)
	{
#endif
		rtn = UpdateInput(cnn);
#ifdef TIME_TEST
	}
	time2 = timeGetTime();

	for(j=0;j<1000;j++)
	{
#endif
		if(rtn == 0)
		{
			rtn = UpdatedK(cnn);
		}
#ifdef TIME_TEST
	}
	time3 = timeGetTime();
	{
		FILE* ini;
		ini = fopen("time.csv","a+");
		fprintf(ini,"%d,%d,%d,%d\n",time0,time1,time2,time3);
		fclose(ini);
	}
#endif
	return rtn;
}