#ifndef _CNN_BP_H_
#define _CNN_BP_H_
#include "cnnCom.h"

float LossFunction(CNN* cnn, int label[], float* out);
float BpLossFunction(int label[], float* out, int outsize);
float SoftMaxLossFunction();
int UpdateAllConnectNetdInput(ALLCONNECTNET* layer);
int UpdateCovnLayerdInput(CNNCONVNLAYER* layer);
int UpdateSamplingLayerdInput(CNNSAMPLINGLAYER* layer);
int UpdateConvnLayerdK(CNNCONVNLAYER* layer);
int UpdateAllConnectNetdK(ALLCONNECTNET* layer);
int UpdateInput(CNN* cnn);
int UpdatedK(CNN* cnn);
#endif