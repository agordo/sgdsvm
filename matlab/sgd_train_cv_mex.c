#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <xmmintrin.h>

#include <assert.h>
#include <math.h>
#include <sys/time.h>

#include <mex.h>

#include "core.h"


mxArray *parseOutput(sgd_output_info_t *output)
{
    /* Very ugly... */
    const char *infonames[8] = {"eta0", "lbd", "beta", "bias_multiplier", "t","updates","epoch","acc"};
    /* Allocate array for the structure */
    mwSize dims[1];
    dims[0]=1;
    
    mxArray *infoArray = mxCreateStructMatrix(1, 1, 8, infonames);
    /* Allocate arrays for the data */
    mxArray *eta0A = mxCreateNumericArray(1, dims, mxSINGLE_CLASS, mxREAL);
    mxArray *lambdaA = mxCreateNumericArray(1, dims, mxSINGLE_CLASS, mxREAL);
    mxArray *betaA = mxCreateNumericArray(1, dims, mxINT32_CLASS, mxREAL);
    mxArray *bias_multiplierA = mxCreateNumericArray(1,dims, mxSINGLE_CLASS, mxREAL);
    mxArray *tA = mxCreateNumericArray(1, dims, mxINT32_CLASS, mxREAL);
    mxArray *updatesA = mxCreateNumericArray(1, dims, mxINT32_CLASS, mxREAL);
    mxArray *epochA = mxCreateNumericArray(1, dims, mxINT32_CLASS, mxREAL);
    mxArray *accA = mxCreateNumericArray(1, dims, mxSINGLE_CLASS, mxREAL);
    /* Associate with arrays and write... */
    float *eta0 = (float*)mxGetData(eta0A);
    eta0[0] = output->eta0;
    float *lbd = (float*)mxGetData(lambdaA);
    lbd[0] = output->lbd;
    int *beta = (int*)mxGetData(betaA);
    beta[0] = output->beta;
    float *bias_multiplier = (float*)mxGetData(bias_multiplierA);
    bias_multiplier[0] = output->bias_multiplier;
    int *t = (int*)mxGetData(tA);
    t[0] = output->t;
    int *updates = (int*)mxGetData(updatesA);
    updates[0] = output->updates;
    int *epoch = (int*)mxGetData(epochA);
    epoch[0] = output->epoch;
    float *acc = (float*)mxGetData(accA);
    acc[0] = output->acc;
    /* set fields ...*/
    mxSetField(infoArray, 0, infonames[0], eta0A);
    mxSetField(infoArray, 0, infonames[1], lambdaA);
    mxSetField(infoArray, 0, infonames[2], betaA);
    mxSetField(infoArray, 0, infonames[3], bias_multiplierA);
    mxSetField(infoArray, 0, infonames[4], tA);
    mxSetField(infoArray, 0, infonames[5], updatesA);
    mxSetField(infoArray, 0, infonames[6], epochA);
    mxSetField(infoArray, 0, infonames[7], accA);
    return infoArray;
}


sgd_cv_params_t *parseOpts(const mxArray *paramsArray){
    
    sgd_cv_params_t *params = mxMalloc(sizeof(sgd_cv_params_t));
    
    const char *fnames[7] = {"eta0","lambdas", "betas", "bias_multipliers","epochs","eval_freq", "t"};
    mxArray *lbdsA;
    mxArray *betasA;
    mxArray *bias_multipliersA;
    lbdsA = mxGetField(paramsArray, 0, fnames[1]);
    betasA = mxGetField(paramsArray, 0, fnames[2]);
    bias_multipliersA = mxGetField(paramsArray, 0, fnames[3]);
    
    if (!mxIsClass(lbdsA, "single"))
    {
        printf("Error in lambdas. Expected single precision\n");
        return NULL;
    }
    if (!mxIsClass(betasA, "int32"))
    {
        printf("Error in betas. Expected int32\n");
        return NULL;
    }
    if (!mxIsClass(bias_multipliersA, "single"))
    {
        printf("Error in bias_multipliers. Expected single precision\n");
        return NULL;
    }
    
    
    params->eta0 =(float)mxGetScalar(mxGetField(paramsArray, 0, fnames[0]));
    params->lbds =  (float*)mxGetData(lbdsA);
    params->nlambdas = (int) mxGetN(lbdsA);
    params->betas =  (int*)mxGetData(betasA);
    params->nbetas = (int) mxGetN(betasA);
    params->bias_multipliers =  (float*)mxGetData(bias_multipliersA);
    params->nbias_multipliers = (int) mxGetN(bias_multipliersA);
    params->epochs =(int)mxGetScalar(mxGetField(paramsArray, 0, fnames[4]));
    params->eval_freq =(int)mxGetScalar(mxGetField(paramsArray, 0, fnames[5]));
    params->t =(int)mxGetScalar(mxGetField(paramsArray, 0, fnames[6]));
    
    return params;
    
}

void mexFunction (int nlhs, mxArray *plhs[],
        int nrhs, const mxArray*prhs[]) {
    
    /* Input parameters */
    /* [0]: Xtrain
     * [1]: Ltrain
     * [2]: Xvalid
     * [3]: Lvalid
     * [4]: opts...
     *
     * [x]: eta0
     * [x]: lambdas
     * [x]: betas
     * [x]: bias_multipliers
     * [x]: epochs
     * [x]: eval_freq
     * [x]: t0
     */
    /* Output parameters */
    /* [0]: W,B,PlattsA,PlattsB,info
     *
     */

    int i;
    int d;
    int Ntrain, Nval;
    float *Xtrain;
    float *Xval;
    int *Ltrain;
    int *Lval;
    sgd_cv_params_t *params;
    const mxArray  *paramsArray;
    
    
    float *W;
    float *B;
    float *PlattsA;
    float *PlattsB;
    
    
    /* Read Data */
    Xtrain =  (float*)mxGetData(prhs[0]);
    d = (int) mxGetM(prhs[0]);
    Ntrain = (int) mxGetN(prhs[0]);
    Ltrain = (int*)mxGetData(prhs[1]);
    Xval =  (float*)mxGetData(prhs[2]);
    Nval = (int) mxGetN(prhs[2]);
    Lval = (int*)mxGetData(prhs[3]);
    paramsArray = prhs[4];
    
    params = parseOpts(paramsArray);
    if (params==NULL)
        return;
    
    
    /* Allocate output data */
    const char *fnames[4] = {"W","B", "PlattsA", "PlattsB"};
    /*const char *fnames[5] = {"W","B", "PlattsA", "PlattsB","info"};*/
    
    
    plhs[0] = mxCreateStructMatrix(1, 1,4, fnames);
    mwSize dims[1];
    dims[0]=d;
    mxArray *Warray = mxCreateNumericArray(1, dims, mxSINGLE_CLASS, mxREAL);
    dims[0] = 1;
    mxArray *Barray = mxCreateNumericArray(1, dims, mxSINGLE_CLASS, mxREAL);
    mxArray *PlattsAarray = mxCreateNumericArray(1, dims, mxSINGLE_CLASS, mxREAL);
    mxArray *PlattsBarray = mxCreateNumericArray(1, dims, mxSINGLE_CLASS, mxREAL);
    W = (float*)mxGetData(Warray);
    B = (float*)mxGetData(Barray);
    PlattsA = (float*)mxGetData(PlattsAarray);
    PlattsB = (float*)mxGetData(PlattsBarray);

    /* Train! */
    /* Since we asked the positive class to be "1", we train class 1... */
    sgd_output_info_t output;
    sgd_train_class_cv(1,Ntrain,d,
            Xtrain,Ltrain,
            Nval,
            Xval,Lval,
            params,
            W,B,PlattsA,PlattsB,
            &output);
    
       
    //mxArray *infoarray = parseOutput(&output);
    
    
    mxSetField(plhs[0], 0, fnames[0], Warray);
    mxSetField(plhs[0], 0, fnames[1], Barray);
    mxSetField(plhs[0], 0, fnames[2], PlattsAarray);
    mxSetField(plhs[0], 0, fnames[3], PlattsBarray);
    //mxSetField(plhs[0], 0, fnames[4], infoarray);
    mxFree(params); 
    return;
    
}

