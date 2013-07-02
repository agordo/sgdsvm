import numpy as np
from numpy import ctypeslib
import ctypes
from ctypes import *
lsgd = ctypes.cdll.LoadLibrary("/home/lear/gordo/Code/sgd_albert2/libsgd2.so")

lsgd.Platts.argtypes = [np.ctypeslib.ndpointer(dtype=c_float, ndim=1) #*scores
                        ,np.ctypeslib.ndpointer(dtype= c_int, ndim=1) #Lval  
                ,c_int # l    
                , POINTER(c_float) #PlattsA
                , POINTER(c_float)] #PlattsB

def LearnPlatts(scores, Lval):
    PlattsA= c_float()
    PlattsB= c_float()
    lsgd.Platts(scores, Lval.T[0], len(Lval), byref(PlattsA), byref(PlattsB))
    return PlattsA.value, PlattsB.value


class sgd_params_t(Structure):
    _fields_ = [
            ("eta0", c_float),
            ("lbd", c_float),
            ("beta", c_int32),
            ("bias_multiplier", c_float),
            ("epochs", c_int32),
            ("eval_freq", c_int32),
            ("t", c_int32),
            ("weightPos", c_float),
            ("weightNeg", c_float)]


class sgd_cv_params_t(Structure):
    _fields_ = [
            ("eta0s", POINTER(c_float)),
            ("netas", c_int32),
            ("lbds", POINTER(c_float)),
            ("nlambdas", c_int32),
            ("betas", POINTER(c_int32)),
            ("nbetas", c_int32),
            ("bias_multipliers", POINTER(c_float)),
            ("nbias_multipliers", c_int32),
            ("epochs", c_int32),
            ("eval_freq", c_int32),
            ("t", c_int32),
            ("weightPos", c_float),
            ("weightNeg", c_float)]




class sgd_output_info_t(Structure):
    _fields_ = [
            ("eta0", c_float),
            ("lbd", c_float),
            ("beta", c_int32),
            ("bias_multiplier", c_float),
            ("t", c_int32),
            ("updates", c_int32),
            ("epoch", c_int32),
            ("acc", c_float),
            ("weightPos", c_float),
            ("weightNeg", c_float)]







class pq_info_t(Structure):
    _fields_ = [
            ("nsq", c_int32),
            ("ksq", c_int32),
            ("dsq", c_int32),
            ("centroids", POINTER(c_float))]




lsgd.sgd_train_class_cv_pq.argtypes = [c_int # c
                , POINTER(pq_info_t)
                           ,c_int #Ntrain
                           ,c_int #d
                           ,np.ctypeslib.ndpointer(dtype=c_uint8, ndim=1) #Xtrain_pq
                           ,np.ctypeslib.ndpointer(dtype= c_int, ndim=1) #Ltrain
                           ,c_int #Nval
                           ,np.ctypeslib.ndpointer(dtype=c_uint8, ndim=1) #Xval_pq
                           ,np.ctypeslib.ndpointer(dtype= c_int, ndim=1) #Lval  
                           ,POINTER(sgd_cv_params_t)
                           ,np.ctypeslib.ndpointer(dtype=c_float, ndim=1) #*W
                           ,POINTER(c_float) #*B
                           ,POINTER(c_float) #*PlattsA
                           ,POINTER(c_float) #*PlattsB
                           ,POINTER(sgd_output_info_t)]


lsgd.sgd_train_class_pq.argtypes = [c_int # c
                , POINTER(pq_info_t)
                           ,c_int #Ntrain
                           ,c_int #d                                                      
                           ,np.ctypeslib.ndpointer(dtype=c_uint8, ndim=1) #Xtrain_pq
                           ,np.ctypeslib.ndpointer(dtype= c_int, ndim=1) #Ltrain
                ,c_int #Nval
                           ,np.ctypeslib.ndpointer(dtype=c_uint8, ndim=1) #Xval_pq
                           ,np.ctypeslib.ndpointer(dtype= c_int, ndim=1) #Lval  
                           ,POINTER(sgd_params_t)
               ,np.ctypeslib.ndpointer(dtype=c_float, ndim=1) #*W
                           ,POINTER(c_float) #*B
                           ,POINTER(sgd_output_info_t)]





def sgdalbert_train_cv_pq(pq,Xtrain_pqcodes,Ltrain,Xval_pqcodes,Lval, params):
    nc = np.max(Ltrain)+1
    infos = [None]*nc
    d = pq.nsq*pq.dsq
    ntrain = len(Ltrain)
    nval = len(Lval)
    W = np.zeros((nc,d),dtype = np.float32)
    Bias = [None]*nc
    PlattsA = [None]*nc
    PlattsB = [None]*nc
    for c in range(nc):
        print "Doing class %d/%d"%(c,nc)
        info = sgd_output_info_t()
        bias_tmp=c_float()
        plattsA_tmp=c_float()
        plattsB_tmp=c_float()    
        lsgd.sgd_train_class_cv_pq(c,byref(pq),  ntrain,d,  Xtrain_pqcodes,Ltrain.T[0],nval, Xval_pqcodes,Lval.T[0],byref(params),W[c], byref(bias_tmp), byref(plattsA_tmp), byref(plattsB_tmp), byref(info))
        infos[c] = info
        Bias[c] = bias_tmp.value
        PlattsA[c] = plattsA_tmp.value
        PlattsB[c] = plattsB_tmp.value    
    return W,np.array(Bias, dtype=np.float32),np.array(PlattsA, dtype=np.float32),np.array(PlattsB, dtype=np.float32),infos



def sgdalbert_train_cv_pq_oneclass(cls,pq,  Xtrain_pqcodes,Ltrain,Xval_pqcodes,Lval,  params):
    d = pq.nsq*pq.dsq
    ntrain = len(Ltrain)
    nval = len(Lval)
    W = np.zeros(d,dtype = np.float32)
    bias_tmp=c_float()
    plattsA_tmp=c_float()
    plattsB_tmp=c_float()    
    info = sgd_output_info_t()
    lsgd.sgd_train_class_cv_pq(cls,byref(pq),  ntrain,d,  Xtrain_pqcodes,Ltrain.T[0],nval, Xval_pqcodes,Lval.T[0],byref(params),W, byref(bias_tmp), byref(plattsA_tmp), byref(plattsB_tmp), byref(info))
    return W,bias_tmp.value,plattsA_tmp.value,plattsB_tmp.value,info

def sgdalbert_update_pq_oneclass(cls, pq, W,Bias, Xtrain_pqcodes,Ltrain_, params):
    Ltrain = Ltrain_.copy()
    d = pq.nsq*pq.dsq
    ntrain = len(Ltrain)
    info = sgd_output_info_t()
    ppos = np.where(Ltrain==cls)
    pneg = np.where(Ltrain!=cls)    
    Ltrain[ppos] = 1
    Ltrain[pneg] = -1
    bias_tmp=c_float(Bias/params.bias_multiplier)
    lsgd.sgd_train_class_pq(cls,byref(pq),ntrain, d, Xtrain_pqcodes,Ltrain.T[0],0, np.empty([1], dtype = c_uint8), np.empty([1], dtype = c_int32),byref(params),W, byref(bias_tmp), byref(info))
    return W,bias_tmp.value*params.bias_multiplier,info.t



'''
lsgd.sgd_train_class_pq.argtypes = [c_int # c
                , POINTER(pq_info_t)
                           ,np.ctypeslib.ndpointer(dtype=c_float, ndim=1) #*W
                           ,np.ctypeslib.ndpointer(dtype=c_float, ndim=1) #*B
                           ,c_int #d
                           ,c_int #Ntrain
                           ,c_int #Nval
                           ,np.ctypeslib.ndpointer(dtype=c_uint8, ndim=1) #Xtrain_pq
                           ,np.ctypeslib.ndpointer(dtype= c_int, ndim=1) #Ltrain
                           ,np.ctypeslib.ndpointer(dtype=c_uint8, ndim=1) #Xval_pq
                           ,np.ctypeslib.ndpointer(dtype= c_int, ndim=1) #Lval  
                           ,POINTER(sgd_params_t)
                           ,POINTER(sgd_output_info_t)]

lsgd.train_epoch_pq.argtypes = [POINTER(pq_info_t)
                           ,c_int #Ntrain
                           ,c_int #d
                           ,np.ctypeslib.ndpointer(dtype=c_uint8, ndim=1) #Xtrain_pq
                           ,np.ctypeslib.ndpointer(dtype= c_int, ndim=1) #Ltrain
                           ,np.ctypeslib.ndpointer(dtype= c_int, ndim=1) #perm                          
                           ,POINTER(sgd_params_t) # params
                           ,POINTER(c_int32) # t
                           ,np.ctypeslib.ndpointer(dtype=c_float, ndim=1) #*W
                           ,POINTER(c_float)] #*B




def sgdalbert_update_pq_oneclass(cls, pq, W,Bias, Xtrain_pqcodes,Ltrain_, params):
    Ltrain = Ltrain_.copy()
    d = pq.nsq*pq.dsq
    ntrain = len(Ltrain)
    info = sgd_output_info_t()
    ppos = np.where(Ltrain==cls)
    pneg = np.where(Ltrain!=cls)    
    Ltrain[ppos] = 1
    Ltrain[pneg] = 1
    import pdb;pdb.set_trace()
    lsgd.sgd_train_class_pq(cls,byref(pq),W[0], Bias[0:1], d, ntrain,0, Xtrain_pqcodes,Ltrain.T[0],np.empty([1], dtype = c_uint8), np.empty([1], dtype = c_int32),byref(params), byref(info))
    return W,Bias,info

























lsgd.sgd_train_class_cv.argtypes = [c_int # c
                           ,np.ctypeslib.ndpointer(dtype=c_float, ndim=1) #*W
                           ,np.ctypeslib.ndpointer(dtype=c_float, ndim=1) #*B
                           ,np.ctypeslib.ndpointer(dtype=c_float, ndim=1) #*PlattsA
                           ,np.ctypeslib.ndpointer(dtype=c_float, ndim=1) #*PlattsB
                           ,c_int #d
                           ,c_int #Ntrain
                           ,c_int #Nval
                           ,np.ctypeslib.ndpointer(dtype=c_float, ndim=2) #Xtrain
                           ,np.ctypeslib.ndpointer(dtype= c_int, ndim=1) #Ltrain
                           ,np.ctypeslib.ndpointer(dtype=c_float, ndim=2) #Xval
                           ,np.ctypeslib.ndpointer(dtype= c_int, ndim=1) #Lval  
                           ,POINTER(sgd_cv_params_t)
                           ,POINTER(sgd_output_info_t)]

def sgdalbert_train_cv(Xtrain,Ltrain,Xval,Lval,params):
    nc = np.max(Ltrain)+1
    infos = [None]*nc
    ntrain, d = Xtrain.shape
    nval = Xval.shape[0]
    W = np.zeros((nc,d),dtype = np.float32)
    Bias = np.zeros([nc],dtype = np.float32)
    PlattsA = np.zeros([nc],dtype = np.float32)
    PlattsB = np.zeros([nc],dtype = np.float32)
    for c in range(nc):
        print "Doing class %d/%d"%(c,nc)
        info = sgd_output_info_t()
        lsgd.sgd_train_class_cv(c,W[c], Bias[c:(c+1)], PlattsA[c:(c+1)], PlattsB[c:(c+1)], d, ntrain,nval, Xtrain,Ltrain.T[0],Xval,Lval.T[0],byref(params), byref(info))
        infos[c] = info
    return W,Bias,PlattsA,PlattsB,infos
'''


'''
def compute_map(s,l):
    ranks=[]
    nrl = 0
    acc = 0
    for i in range(len(l)):
        if l[i]==-1: continue
        c=0
        c_eq=0
        for j in range(len(l)):
            if i ==j: continue
            if s[j] > s[i]: c+=1
            elif s[j]==s[i]: c_eq+=1
        c += c_eq/2
        ranks.append(c)
        nrl+=1
    ranks.sort()
    for i in range(len(ranks)):
        acc += (i+1.0)/(ranks[i]+1.0)
    acc/=nrl        
    return acc
    '''
