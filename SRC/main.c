#include "cnnSetup.h"
#include "cnnff.h"
#include "cnnTrain.h"

//#include "cv.h"
//#include "cxcore.h"
//#include "highgui.h"

#include <stdlib.h>
#include <time.h>
#define LABEL_SIZE		10
#define IMAGE_WIDTH		28
#define IMAGE_HEIGHT	28
#define IMAGE_NUM		1



void SaveCnn(CNN* cnn)
{
	int i,j,k,p,q;
	FILE* ini;
	ini = fopen("cnn_data.txt", "w");

	for (i=0; i<cnn->depth;i++)
	{
		switch(cnn->type[i])
		{
		case CONVNLAYER:
			{
				fprintf(ini,"float w%d[] =\n{\n",i+1);
				for(j=0; j<((CNNCONVNLAYER*)(cnn->layer[i]))->outputImage->ImageNum; j++)
				{
					for(k=0; k<((CNNCONVNLAYER*)(cnn->layer[i]))->inputImage->ImageNum; k++)
					{
						for(p=0; p<((CNNCONVNLAYER*)(cnn->layer[i]))->kernelSize; p++)
						{
							for(q=0; q<((CNNCONVNLAYER*)(cnn->layer[i]))->kernelSize; q++)
							{
								fprintf(ini,"%ff,", ((CNNCONVNLAYER*)(cnn->layer[i]))->k[j][k][p][q]);
							}
						}
						fprintf(ini,"\n");
					}
					fprintf(ini,"\n");
				}
				fprintf(ini,"};\n\n");
				break;
			}
		case SAMPLINGLAYER:
			{
				fprintf(ini,"float w%d[] ={0.0};\n\n",i+1);
				break;
			}
		case ALLCONNECTLAYER:
			{
				fprintf(ini,"float w%d[] =\n{\n",i+1);
				for(j=0; j<((ALLCONNECTNET*)(cnn->layer[i]))->outputSize; j++)
				{
					for(k=0; k<((ALLCONNECTNET*)(cnn->layer[i]))->inputSize; k++)
					{
						fprintf(ini,"%ff,",((ALLCONNECTNET*)(cnn->layer[i]))->k[j][k]);
					}
					fprintf(ini,"\n");
				}
				fprintf(ini,"};\n\n");
				break;
			}
		case SOFTMAXLAYER:
			{
				fprintf(ini,"float w%d[] =\n{\n",i+1);
				for(j=0; j<((CNNSOFTMAXLAYER*)(cnn->layer[i]))->outputSize; j++)
				{
					for(k=0; k<((CNNSOFTMAXLAYER*)(cnn->layer[i]))->inputSize; k++)
					{
						fprintf(ini,"%ff,",((CNNSOFTMAXLAYER*)(cnn->layer[i]))->k[j][k]);
					}
					fprintf(ini,"\n");
				}
				fprintf(ini,"};\n\n");
				break;
			}
		default:
			{
				break;
			}
		}
	}



	for (i=0; i<cnn->depth;i++)
	{
		switch(cnn->type[i])
		{
		case CONVNLAYER:
			{
				fprintf(ini,"float b%d[] =\n{\n",i+1);
				for(j=0; j<((CNNCONVNLAYER*)(cnn->layer[i]))->outputImage->ImageNum; j++)
				{
					fprintf(ini,"%ff,", ((CNNCONVNLAYER*)(cnn->layer[i]))->b[j]);
				}
				fprintf(ini,"\n};\n");
				break;
			}
		case SAMPLINGLAYER:
			{
				fprintf(ini,"float b%d[] ={0.0};\n",i+1);
				break;
			}
		case ALLCONNECTLAYER:
			{
				fprintf(ini,"float b%d[] =\n{\n",i+1);
				for(j=0; j<((ALLCONNECTNET*)(cnn->layer[i]))->outputSize; j++)
				{
					fprintf(ini,"%ff,", ((ALLCONNECTNET*)(cnn->layer[i]))->b[j]);
				}
				fprintf(ini,"\n};\n");
				break;
			}
		case SOFTMAXLAYER:
			{
				fprintf(ini,"float b%d[] =\n{\n",i+1);
				for(j=0; j<((CNNSOFTMAXLAYER*)(cnn->layer[i]))->outputSize; j++)
				{
					fprintf(ini,"%ff,", ((CNNSOFTMAXLAYER*)(cnn->layer[i]))->b[j]);
				}
				fprintf(ini,"\n};\n");
				break;
			}
		default:
			{
				break;
			}
		}
	}

	fclose(ini);
}



int main()
{
	int i,j,imageIndex;
	CNN cnn;
	int result= 0;
	int depth = 6;
	unsigned char types[] = {1,2,1,2,3,4};
	int scales[] = {5,2,5,2,0,0};
	int layerImageNum[] = {6,6,12,12,16,LABEL_SIZE};
	char actFuncFlag[] = {LEAKY_RELU,LEAKY_RELU,LEAKY_RELU,LEAKY_RELU,SIGMOD,SIGMOD};
	char samplingFlag = MEAN_TYPE;
	char ImageFlag = NOFILLER;
	float* out = NULL;
	unsigned char image[IMAGE_WIDTH*IMAGE_HEIGHT*IMAGE_NUM];
	float x[IMAGE_WIDTH*IMAGE_HEIGHT*IMAGE_NUM];
	FILE* imageFile;
	FILE* resultFile;
	char filename[256];
	int label[LABEL_SIZE];
	int labelIndex = 0;
	int labelMax = 0;
	int outIndex = 0;
	float outMax = 0.0;

	int totalNum = 0;
	int errNum = 0;

	int totalDigitNum[LABEL_SIZE] = {0};
	int errDigitNum[LABEL_SIZE] = {0};

	int minGray,maxGray;

	TRANPARA trainPara;
	int trainImageNum = 60000;
	float** X = (float**)malloc(trainImageNum*sizeof(float*));
	int** Y = (int**)malloc(trainImageNum*sizeof(int*));
	
	float* xBlock = (float*)malloc(trainImageNum*IMAGE_HEIGHT*IMAGE_WIDTH*sizeof(float));
	int*   yBlock = (int*)malloc(trainImageNum*LABEL_SIZE*sizeof(int));

	for(i=0;i<trainImageNum;i++)
	{
		X[i] = xBlock+i*IMAGE_HEIGHT*IMAGE_WIDTH*IMAGE_NUM;
		Y[i] = yBlock+i*LABEL_SIZE;
	}


	cnn.depth = 0;
	SetCnn(&cnn,depth,types,scales,layerImageNum,IMAGE_NUM,IMAGE_WIDTH,IMAGE_HEIGHT,actFuncFlag,samplingFlag,ImageFlag);


	printf("读取图像中\n");
	sprintf(filename,"train.txt");
	imageFile = fopen(filename,"a+");
	for(j=0; j<LABEL_SIZE; j++)
	{
		for(imageIndex =0; imageIndex< trainImageNum/LABEL_SIZE; imageIndex++)
		{
			for(i=0; i<LABEL_SIZE; i++)
			{
				fscanf(imageFile, "%d",&Y[imageIndex*LABEL_SIZE+j][i]);
			}
			//正常读图
			for(i=0; i<IMAGE_HEIGHT*IMAGE_WIDTH; i++)
			{
				fscanf(imageFile, "%d",&image[i]);
				X[imageIndex*LABEL_SIZE+j][i] = 1.0f*image[i]/255;
			}

			if(imageIndex % 100 == 0)
			{
				printf(".");
			}
		}
	}


	srand((unsigned int) time(NULL));
	for(i=0;i<trainImageNum*100;i++)
	{
		//if(rand()%2 == 1)
		{
			int rand1 = rand();
			int rand2 = rand();
			int rand3 = rand();//rand()函数生成的随机数最大只到32767，所以需要做一个乘法，扩大随机数范围
			int rand4 = abs((rand1*rand2-rand3)%trainImageNum);
			int rand5 = abs((rand2*rand3-rand1)%trainImageNum);
			int *temp1;
			float* temp2;

			temp1 = Y[rand4];
			temp2 = X[rand4];

			Y[rand4] = Y[rand5];
			X[rand4] = X[rand5];

			Y[rand5] = temp1;
			X[rand5] = temp2;
		}	

	}

	printf("\n开始训练：\n");

	trainPara.alpha = 0.1;
	trainPara.batchSize = 50;
	trainPara.numepochs = 50;

	CnnTrain(&cnn,X,Y,trainImageNum,&trainPara);

	SaveCnn(&cnn);



	fclose(imageFile);


	sprintf(filename,"test.csv");
	resultFile = fopen(filename,"a+");
	imageFile = fopen("test.txt","r");
	for(imageIndex =0; imageIndex< 10000; imageIndex++)
	{
		for(i=0; i<LABEL_SIZE; i++)
		{
			fscanf(imageFile, "%d",&label[i]);
		}
		//正常读图
		for(i=0; i<IMAGE_WIDTH*IMAGE_HEIGHT; i++)
		{
			fscanf(imageFile, "%d",&image[i]);
			x[i] = 1.0f*image[i]/255;
		}
		
		out = CnnFf(&cnn, x);

		labelIndex = 0;
		labelMax = 0;
		outIndex = 0;
		outMax = 0.0;
		for(i=0; i<LABEL_SIZE; i++)
		{
			if(out[i] > outMax)
			{
				outIndex = i;
				outMax = out[i];
			}
			if(label[i] > labelMax)
			{
				labelMax = label[i];
				labelIndex = i;
			}
		}


		fprintf(resultFile,"%d,%d,%d,",imageIndex,labelIndex,outIndex);
		for(i=0;i<LABEL_SIZE;i++)
		{
			fprintf(resultFile,"%f,",out[i]);
		}
		fprintf(resultFile,"\n");
		if(labelIndex != outIndex)
		{
			errNum ++;
			errDigitNum[labelIndex] ++;
		}
		totalNum ++;
		totalDigitNum[labelIndex] ++;
		
		printf("%d, %d\n",totalNum,errNum);
	}
	fclose(imageFile);
	fclose(resultFile);

	resultFile = fopen("test_result.txt","w");
	for(i=0;i<LABEL_SIZE;i++)
	{
		fprintf(resultFile,"%d:   总张数：%d,   错误张数：%d,   错误率：%f\n", i, totalDigitNum[i], errDigitNum[i], 1.0*errDigitNum[i]/totalDigitNum[i]);
	}
	fprintf(resultFile,"总张数：%d,   错误张数：%d,   错误率：%f\n", totalNum, errNum, 1.0*errNum/totalNum);
	fclose(resultFile);



//CNN建立函数测试
	//CNNCONVNLAYER* a;
	//CNNSAMPLINGLAYER* b;
	//ALLCONNECTNET* c;

	//inputImage.ImageNum = 1;
	//inputImage.height = 10;
	//inputImage.width  = 10;
	//inputImage.Image = MallocImageBlock(1,10,10);

	//for(i=0;i<inputImage.height;i++)
	//{
	//	for(j=0;j<inputImage.width;j++)
	//	{
	//		inputImage.Image[0][i][j] = i*inputImage.width+j; 
	//	}
	//}
	//cnn.depth = layerNum;
	//result = SetCnn(&cnn, layerNum, types, scales, layerImageNum, &inputImage, actFuncFlag, samplingFlag, ImageFlag);

	//{

	//	a = (CNNCONVNLAYER*)cnn.layer[0];
	//	b = (CNNSAMPLINGLAYER*)cnn.layer[1];
	//	c = (ALLCONNECTNET*)cnn.layer[2];

	//}

	//DleteCnn(&cnn);

//sigmod及getNet函数测试
	//float input[10] = {-4.0, -3.0, -2.0, -1.0 ,-0.0, 1.0, 2.0, 3.0, 4.0, 5.0};
	//float b[10] = {0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0};
	//float **w = (float**)malloc(sizeof(float*)*3);
	//float output[10];
	//for(i=0;i<3;i++)
	//{
	//	w[i] = (float*)malloc(sizeof(float)*10);
	//}

	//for(i=0;i<3;i++)
	//{
	//	for(j=0;j<10;j++)
	//	{
	//		w[i][j] = i*0.2;
	//	}
	//}

	//GetNet(w,b,input,output,10,3);
	//Sigmod(input, output, 10);



	//float **input =(float**)malloc(sizeof(float*)*10);
	//float **w = (float**)malloc(sizeof(float*)*3);
	//float output[100] = {0};

	//for(i=0;i<3;i++)
	//{
	//	w[i] = (float*)malloc(sizeof(float)*3);
	//}

	//for(i=0;i<10;i++)
	//{
	//	input[i] = (float*)malloc(sizeof(float)*10);
	//}

	//for(i=0;i<3;i++)
	//{
	//	for(j=0;j<3;j++)
	//	{
	//		w[i][j] = i*3+j;
	//	}
	//}

	//for(i=0;i<10;i++)
	//{
	//	for(j=0;j<10;j++)
	//	{
	//		input[i][j] = i*10+j;
	//	}
	//}


	//Convn(input, w, output, 10, 10, 3);()

	getchar();
	return 0;

}
//*/
/*

int main()
{
	float** A = MallocSingleImage(2,2);
	float** B = MallocSingleImage(2,2);
	float* out = (float*)malloc(4*4*sizeof(float));
	int i,j;

	for(i=0;i<2;i++)
	{
		for(j=0;j<2;j++)
		{
			A[i][j]=1.0;
			B[i][j]=1.0;
		}
	}
	for(i=0;i<16;i++)
	{
		out[i] = 0.0;
	}

	Convn2(A,B,out,2,2,2,2,FULL);
	for(i=0;i<4*4;i++)
	{
		if(i%4==0)
		{
			printf("\n");
		}
		printf("%f  ",out[i]);
	}
	getchar();
	getchar();
	getchar();
	getchar();
}
*/