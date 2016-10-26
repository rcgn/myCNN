#ifndef _CNN_TRAIN_H_
#define _CNN_TRAIN_H_
#include "cnnCom.h"
#include "cnnBp.h"
#include "cnnff.h"
#include "cnnApplyGrads.h"


int SingleImageTrain(CNN* cnn, float* X, int* Y, float* loss);
int CnnTrain(CNN* cnn, float** X, int** Y, int imageNum, TRANPARA* trainPara);
#endif