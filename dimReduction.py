import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Input, BatchNormalization, Activation
from keras.backend import transpose
from sklearn.metrics import pairwise as pw
import tensorflow as tf
from tfdeterminism import patch
import scipy.stats, random, os

# 7-layer Autoencoder
def auto(esetNoisy,esetNormal,hidden_size,hidden_size2,activ1,optim,code_size,random_seed):
    ## to improve reproducibility
    random.seed(random_seed)
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)
    autoencoder = Sequential()
    # hidden_1
    autoencoder.add(Dense(hidden_size, input_dim=esetNoisy.shape[1], use_bias=False))
    autoencoder.add(BatchNormalization())
    autoencoder.add(Activation(activ1))
    # hidden_12
    autoencoder.add(Dense(hidden_size2, use_bias=False))
    autoencoder.add(BatchNormalization())
    autoencoder.add(Activation(activ1))
    # code
    autoencoder.add(Dense(code_size, use_bias=False))
    autoencoder.add(BatchNormalization())
    autoencoder.add(Activation(activ1))
    # hidden_2
    autoencoder.add(Dense(hidden_size2, use_bias=False))
    autoencoder.add(BatchNormalization())
    autoencoder.add(Activation(activ1))
    # hidden_22
    autoencoder.add(Dense(hidden_size, use_bias=False))
    autoencoder.add(BatchNormalization())
    autoencoder.add(Activation(activ1))
    # output
    autoencoder.add(Dense(esetNoisy.shape[1], use_bias=False))
    autoencoder.add(BatchNormalization())
    autoencoder.add(Activation(activ1))

    autoencoder.compile(optimizer=optim, loss='mean_squared_error')
    autoencoder.fit(esetNoisy, esetNormal,
                    steps_per_epoch=1,
                    epochs=20,
                    shuffle=False,
                    verbose=0)
    return autoencoder

def evaluateAuto(esetSC2Var_noisy_predicted, esetSC2Var):
    rhoList1=[] # Spearman Correlation
    rhoList2=[] # Pearson
    rhoList3=[] # Kendall
    for ind in range(esetSC2Var_noisy_predicted.shape[0]):
        rho1,pval1=scipy.stats.spearmanr(esetSC2Var_noisy_predicted[ind],esetSC2Var[ind])
        rho2,pval2=scipy.stats.pearsonr(esetSC2Var_noisy_predicted[ind],esetSC2Var[ind])
        rho3,pval3=scipy.stats.kendalltau(esetSC2Var_noisy_predicted[ind],esetSC2Var[ind])
        rhoList1.append(rho1)
        rhoList2.append(rho2)
        rhoList3.append(rho3)
    distList = pw.euclidean_distances(esetSC2Var_noisy_predicted,esetSC2Var) # Euclidean Distance
    return rhoList1, rhoList2, rhoList3, distList

def predEval(code_sizeList,esetSC2Var_noisy,esetSC2Var,hidden_size,hidden_size2,activ1,optim,random_seed):
    auto_list = []
    abs_spearman = []
    abs_pearson = []
    abs_kendall = []
    abs_eucDist = []
    scalarTestLoss = []
    for code_size in code_sizeList:
        # Prediction
        autoencoder=auto(esetSC2Var_noisy,esetSC2Var,hidden_size,hidden_size2,activ1,optim,code_size,random_seed)
        auto_list.append(autoencoder)
        esetSC2Var_noisy_predicted = autoencoder.predict(esetSC2Var, steps=1)
        # Evaluation: Spearman, Pearson, Euclidean Distance
        rho1, rho2, rho3, dist = evaluateAuto(esetSC2Var_noisy_predicted, esetSC2Var)
        abs_rho1=[abs(i) for i in rho1]
        abs_spearman.append(np.sum(abs_rho1)/len(abs_rho1))
        abs_rho2=[abs(i) for i in rho2]
        abs_pearson.append(np.sum(abs_rho2)/len(abs_rho2))
        abs_rho3=[abs(i) for i in rho3]
        abs_kendall.append(np.sum(abs_rho3)/len(abs_rho3))
        abs_dist=[abs(i) for i in dist]
        abs_eucDist.append(np.sum(abs_dist)/len(abs_dist))
        # Evaluation: Scalar Test loss
        scalarTestLoss.append(autoencoder.evaluate(esetSC2Var, esetSC2Var, steps=1)) 
    return auto_list, abs_spearman, abs_pearson, abs_kendall, abs_eucDist, scalarTestLoss

def run(esetSC2, geneList, random_seed, hidden_size, hidden_size2, code_sizeList, activ1, optim, noise_factor):
    ## to improve reproducibility
    patch()   
    os.environ['PYTHONHASHSEED']=str(random_seed)
    config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    session = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(session)

    ### genes from feature selection + autoencoder
    esetSC2Var=esetSC2.loc[geneList]
    ## General Processing; original
    esetSC2Var=esetSC2Var.values
    esetSC2Var=esetSC2Var/np.max(esetSC2Var)
    esetSC2Var = transpose(esetSC2Var)


    ## Prediction & Evaluation
    random.seed(random_seed)
    np.random.seed(random_seed)
    auto_list = None
    best_model_ind = None
    ## Denoising Autoencoder Preparation
    esetSC2Var_noisy = esetSC2Var + noise_factor * np.random.normal(size=esetSC2Var.shape) 
    esetSC2Var_noisy = np.clip(esetSC2Var_noisy, 0.0, 1.0) ## already transposed
    evalOut = predEval(code_sizeList,esetSC2Var_noisy,esetSC2Var,hidden_size,hidden_size2,activ1,optim,random_seed)
    [auto_list, abs_spearman, abs_pearson, abs_kendall, abs_eucDist, scalarTestLoss] = evalOut
    bests = [abs_spearman.index(max(abs_spearman)), \
                abs_pearson.index(max(abs_pearson)), \
                abs_kendall.index(max(abs_kendall)), \
                abs_eucDist.index(min(abs_eucDist)), \
                scalarTestLoss.index(min(scalarTestLoss))]
    best_model_ind = max(set(bests), key=bests.count)
    print(bests, best_model_ind)
    return best_model_ind, auto_list[best_model_ind]
