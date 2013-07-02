#include <stdio.h>
#include <stdlib.h>
#include <string.h>


#include <assert.h>
#include <math.h>
#include <sys/time.h>

#include "core.h"
#include "aux.h"


float evaluateEta(int Ntrain, int d,  int nmax, float *Xtrain, int* Ltrain, int *perm, sgd_params_t *params, float eta)
{

    float cost=0;
    float loss = 0;
    float wnorm = 0;
    int i;
    float s;
    float *xi;
    float yi;

    float *W = (float*)malloc(d * sizeof(float));
    for (i=0; i < d; i++)
        W[i]=0;
    float B=0;

    train_epoch_fixed_eta(nmax,d, Xtrain,Ltrain, perm,
            eta,   params, W, &B);

    for ( i=0; i<nmax; i++)
    {
        /* Test...*/
        xi = Xtrain + perm[i]*d;
        yi = Ltrain[perm[i]];
        s= (vec_dotprod(W, xi,d) + B*params->bias_multiplier);

        /* Get Loss*/
        loss+= max(0, 1 - yi *s);
    }

    for (i=0; i < d; i++) wnorm+=(W[i]*W[i]);
    wnorm = sqrt(wnorm);

    loss = loss / nmax;
    cost = loss + 0.5 * params->lbd * wnorm;
    //printf("Trying eta=%.6f  yields cost %.2f\n",eta, cost);
    return cost;
}

float determineEta0(int Ntrain, int d,  int nmax,float *Xtrain,  int* Ltrain,  sgd_params_t *params)
{
    float eta0;
    float factor = 2.0;
    float eta1 = 1;
    float eta2 = eta1 * factor;

    int *perm = (int*)malloc(Ntrain*sizeof(int));
    rpermute(perm, Ntrain);

    float cost1 = evaluateEta( d, Ntrain,  nmax, Xtrain, Ltrain,  perm, params, eta1);
    float cost2 = evaluateEta( d, Ntrain,  nmax, Xtrain, Ltrain,  perm, params, eta2);
    if (cost2 > cost1)
    {
        float tmp = eta1; eta1 = eta2; eta2 = tmp; 
        tmp = cost1; cost1 = cost2; cost2 = tmp; 
        factor = 1 / factor;
    }
    do
    {
        eta1 = eta2; 
        eta2 = eta2 * factor;
        cost1 = cost2;  
        cost2 = evaluateEta( d, Ntrain,  nmax, Xtrain, Ltrain,  perm, params, eta2);
    }while (cost1 > cost2);
    eta0 = eta1 < eta2?eta1:eta2;
    free(perm);

    return eta0;

}
void train_epoch_fixed_eta(int Ntrain, int d, float * Xtrain, int *Ltrain,  int* perm,
        float eta,  sgd_params_t *params, float *W, float *B)
{
    int i;
    float wDivisor;
    float *xi;
    int yi;

    int npos, nneg;
    float L_ovr;
    int updates;
    float accLoss;
    float s;


    /* Set stuff */
    wDivisor = 1;
    npos = 0; nneg = 0;
    updates=0;
    accLoss=0;
    for (i=0; i < Ntrain; i++)
    {

        /* Get the samples */
        xi = Xtrain + perm[i]*d;
        yi = Ltrain[perm[i]];


        /* Use sample only if pos or if nneg < npos * beta. Otherwise skip
         * sample altogether. IE, skip if sample is negative and there are
         * too many negatives already*/
        if (yi==-1 &&  npos * params->beta <= nneg)
            continue;

        /* Get score*/
        /*s = dp_slow(d, W,xi) / wDivisor;*/
        s= (vec_dotprod(W, xi,d) + B[0]*params->bias_multiplier) / wDivisor;
        /* Update the daming factor part*/
        wDivisor = wDivisor / (1 - eta * params->lbd);
        /* If things get out of hand, actually update w and set the divisor
         * back to one */
        if (wDivisor > 1e5)
        {
            scaleVector_slow(d,W,1/wDivisor);
            B[0]/=wDivisor;
            wDivisor = 1;
        }

        /* Get Loss*/
        L_ovr = max(0, 1 - yi *s);
        if (L_ovr > 0)
        {
            /*add_slow(d,W,xi,yi*eta*wDivisor);*/
            vec_addto(W,yi*eta*wDivisor, xi, d);
            B[0]+= params->bias_multiplier*yi*eta*wDivisor;
            updates++;
            accLoss+= L_ovr;
        }
        if (yi==1)
            npos++;
        else
            nneg ++;

    }

    scaleVector_slow(d,W,1.0/wDivisor);
    B[0]/=wDivisor;
    wDivisor = 1;
}

void train_epoch(int Ntrain, int d, float * Xtrain, int *Ltrain,  int* perm,
        sgd_params_t *params, int *t,  float *W, float *B)
{
    int i;
    float wDivisor;
    float *xi;
    int yi;
    float eta;
    int npos, nneg;
    float L_ovr;
    int updates;
    float accLoss;
    float s;


    /* Set stuff */
    wDivisor = 1;
    npos = 0; nneg = 0;
    updates=0;
    accLoss=0;
    for (i=0; i < Ntrain; i++)
    {

        /* Get the samples */
        xi = Xtrain + perm[i]*d;
        yi = Ltrain[perm[i]];

        /* Update eta */
        eta = params->eta0 / (1 + params->lbd * params->eta0 * *t);
        /* Use sample only if pos or if nneg < npos * beta. Otherwise skip
         * sample altogether. IE, skip if sample is negative and there are
         * too many negatives already*/
        if (yi==-1 &&  npos * params->beta <= nneg)
            continue;

        /* Get score*/
        /*s = dp_slow(d, W,xi) / wDivisor;*/
        s= (vec_dotprod(W, xi,d) + *B*params->bias_multiplier) / wDivisor;
        /* Update the daming factor part*/
        wDivisor = wDivisor / (1 - eta * params->lbd);
        /* If things get out of hand, actually update w and set the divisor
         * back to one */
        if (wDivisor > 1e5)
        {
            scaleVector_slow(d,W,1/wDivisor);
            B[0]/=wDivisor;
            wDivisor = 1;
        }

        /* Get Loss*/
        L_ovr = max(0, 1 - yi *s);
        if (L_ovr > 0)
        {
            /*add_slow(d,W,xi,yi*eta*wDivisor);*/
            vec_addto(W,yi*eta*wDivisor, xi, d);
            B[0]+= params->bias_multiplier*yi*eta*wDivisor;
            updates++;
            accLoss+= L_ovr;
        }
        if (yi==1)
            npos++;
        else
            nneg++;

        (*t) = *t + 1;
    }

    scaleVector_slow(d,W,1.0/wDivisor);
    B[0]/=wDivisor;
    wDivisor = 1;
}
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
        sgd_output_info_t *output)
{


    float bestMap = 0;
    int h, i,j,k,l;


    int *Ltrain = (int*)malloc(Ntrain*sizeof(int));
    int *Lval = (int*)malloc(Nval*sizeof(int));
    for (i=0; i < Ntrain; i++) Ltrain[i] = _Ltrain[i]==cls?1:-1;
    for (i=0; i < Nval; i++) Lval[i] = _Lval[i]==cls?1:-1;

    int ncombs = params_cv->nlambdas * params_cv->nbetas * params_cv->nbias_multipliers;
    sgd_params_t *params = (sgd_params_t*)malloc(ncombs*sizeof(sgd_params_t));
    sgd_output_info_t *outputs = (sgd_output_info_t*)malloc(ncombs*sizeof(sgd_output_info_t));

    l=0;
    for (h=0; h < params_cv->netas; h++){
        for (i=0; i < params_cv->nlambdas; i++){
            for (j=0; j < params_cv->nbetas; j++){
                for (k=0; k < params_cv->nbias_multipliers;k++){
                    params[l].eta0 = params_cv->eta0s[h];
                    params[l].lbd = params_cv->lbds[i];
                    params[l].beta = params_cv->betas[j];
                    params[l].bias_multiplier = params_cv->bias_multipliers[k];
                    params[l].epochs = params_cv->epochs;
                    params[l].eval_freq = params_cv->eval_freq;
                    params[l].t = params_cv->t;
                    l++;
                }
            }
        }
    }



    float *Wtmp = (float*)calloc(d*ncombs,sizeof(float));
    //#pragma omp parallel for
    for (int i=0; i < ncombs; i++)
    {
        float Btmp=0;

        sgd_train_class(cls,
                Ntrain,d,
                Xtrain,
                Ltrain,
                Nval,
                Xval,
                Lval,
                &params[i], 
                &Wtmp[d*i], &Btmp,
                &outputs[i]);
        //#pragma omp critical
        {
            if (outputs[i].acc > bestMap)
            {
                bestMap = outputs[i].acc;
                memcpy(W, &Wtmp[d*i], d*sizeof(float));
                *B = Btmp*params[i].bias_multiplier;

                if (output!=NULL)
                {
                    output->eta0 = outputs[i].eta0;
                    output->lbd = outputs[i].lbd;
                    output->beta = outputs[i].beta;
                    output->bias_multiplier = outputs[i].bias_multiplier;
                    output->t = outputs[i].t;
                    output->updates = outputs[i].updates;
                    output->epoch = outputs[i].epoch;
                    output->acc = outputs[i].acc;


                }
            }
        }
    }
    free(Wtmp); 
    float *scoresVal = (float*)malloc(Nval*sizeof(float));
    compute_scores(W,*B, Nval,  d,  Xval, scoresVal);
    Platts(scoresVal, Lval, Nval, PlattsA,PlattsB);
    free(scoresVal);
    free(Ltrain);
    free(Lval);
    free(params);
    free(outputs);
    return ;
}

void sgd_train_class(int cls,
        int Ntrain,int d,          
        float *Xtrain,
        int *Ltrain,
        int Nval,
        float * Xval,
        int *Lval,
        sgd_params_t *params,
        float *W, float *B,
        sgd_output_info_t *output)
{

    if (output!=NULL)
    {        
        output->eta0=params->eta0;
        output->lbd=params->lbd;
        output->beta=params->beta;
        output->bias_multiplier=params->bias_multiplier;
        output->t=params->t;
        output->updates = 0;
        output->epoch = 0;
        output->acc = 0;
    }

    int epoch;
    int *perm = (int*)malloc(Ntrain*sizeof(int));


    int t= params->t;


    float *scoresVal = (float*)malloc(Nval*sizeof(float));
    float bestMap = 0;
    float *bestW = (float*)malloc(d * sizeof(float));
    float bestB=0;

    if (params->eta0==0)
    {
        int nmax = Ntrain > 1000?1000:Ntrain;
        params->eta0 = determineEta0(Ntrain,d, nmax, Xtrain, Ltrain, params);
    }


    int noimprov=0;
    for (epoch=0; epoch < params->epochs; epoch++)
    {

        /* Create permutation */
        rpermute(perm, Ntrain);
        /* Train epoch */
        train_epoch(Ntrain,d, Xtrain, Ltrain, perm, params, &t,W,B);



        if (epoch % params->eval_freq==0)
        {

            /*printf("End of epoch %d/%d. %d updates. Accumulated loss: %.2f\n", epoch, epochs,updates, accLoss);*/
            compute_scores(W,*B*params->bias_multiplier, Nval,  d, Xval, scoresVal);
            float map = compute_map(Nval, scoresVal, Lval);

            /*LOGIT("Is newmap %.5f worse than old map %.5f?\n", map, bestMap);*/
            if (map > bestMap)
            {
                noimprov=0;
                /*
                 * printf("Improved validation score at end of epoch %d/%d: %.2f\n", epoch, epochs, map*100);
                 * mexEvalString("drawnow");
                 */
                memcpy(bestW, W, d*sizeof(float));
                bestB = B[0];
                bestMap = map;
                if (output!=NULL)
                {
                    output->eta0=params->eta0;
                    output->acc = bestMap*100;
                    output->epoch = epoch+1;
                    output->updates = t;

                }
            }
            else
            {
                noimprov++;
            }
            if (noimprov==3)
            {
                /* Three checks without improving. Bail out*/
                break;
            }
        }
    }

    /* One last time... */
    /*printf("End of epoch %d/%d. %d updates. Accumulated loss: %.2f\n", epoch, epochs,updates, accLoss);*/
    compute_scores(W,*B*params->bias_multiplier, Nval,  d, Xval, scoresVal);
    float map = compute_map(Nval, scoresVal, Lval);
    if (map > bestMap)
    {
        /*
         * printf("Improved validation score at end of epoch %d/%d: %.2f\n", epoch, epochs, map*100);
         * mexEvalString("drawnow");
         */
        memcpy(bestW, W, d*sizeof(float));
        bestB = B[0];
        bestMap = map;
        if (output!=NULL)
        {
            output->eta0=params->eta0;
            output->acc = bestMap*100;
            output->epoch = epoch+1;
            output->updates = t;
        }
    }

    /* copy bestW back*/
    memcpy( W, bestW, d*sizeof(float));
    B[0] = bestB;
    free(perm) ;
    free(scoresVal);
    free(bestW);

    return;
}



