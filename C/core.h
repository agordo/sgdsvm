#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <xmmintrin.h>

#include <assert.h>
#include <math.h>
#include <sys/time.h>

#include "aux.h"







void train_epoch(int Ntrain, int d, float * Xtrain, int *Ltrain,  int* perm,
        sgd_params_t *params, int *t,  float *W, float *B);

void train_epoch_fixed_eta(int Ntrain, int d, float * Xtrain, int *Ltrain,  int* perm,
        float eta,  sgd_params_t *params, float *W, float *B);


float evaluateEta(int Ntrain, int d,  int nmax, float *Xtrain, int* Ltrain, int *perm, sgd_params_t *params, float eta);
float determineEta0(int Ntrain, int d,  int nmax,float *Xtrain,  int* Ltrain,  sgd_params_t *params);


void sgd_train_class_cv(int cls,
        int Ntrain,int d,          
        float *Xtrain,
        int *_Ltrain,
	    int Nval,
        float * Xval,
        int *_Lval,
        sgd_cv_params_t *params_cv,
    	float *W, float *B,
        float *PlattsA, float *PlattsB,
        sgd_output_info_t *output);

void sgd_train_class(int cls,
        int Ntrain,int d,          
        float *Xtrain,
        int *Ltrain,
    	int Nval,
        float * Xval,
        int *Lval,
        sgd_params_t *params,
    	float *W, float *B,
        sgd_output_info_t *output);

