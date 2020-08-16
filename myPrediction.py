import pandas as pd
import numpy as np
from keras.models import Model
from keras.backend import transpose
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
import os

## Split separate files for Control vs sPTD and Control vs PPROM
def split(play_x, play_y0, current_comparison, cv_set_size, random_state):
    df = pd.DataFrame(play_x)
    df['play_y'] = play_y0
    not_current_comparison = list(set(play_y0)-set([current_comparison, 'Control']))[0]
    df = df[df['play_y'] != not_current_comparison]
    df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    df_cv = df[:cv_set_size].reset_index(drop=True)
    df_train = df[cv_set_size:].reset_index(drop=True)
    train_y0=df_train['play_y'].values.tolist()
    train_y=[]
    for t in train_y0:
        if t=='Control': train_y.append(0)
        else:  train_y.append(1)
    train_x=df_train.drop(['play_y'], axis=1)
    cv_y0=df_cv['play_y'].values.tolist()
    cv_y=[]
    for t in cv_y0:
        if t=='Control': cv_y.append(0)
        else:  cv_y.append(1)
    cv_x=df_cv.drop(['play_y'],axis=1)
    return train_x, train_y, cv_x, cv_y

def LR(c, weights):
    return LogisticRegression(penalty='l2', C=c, class_weight=weights, solver='newton-cg', random_state=0)

def SVM(k, c, weights):
    return svm.SVC(kernel=k, C=c, class_weight=weights, probability=True, random_state=0)

def RF(weights):
    return RandomForestClassifier(max_depth=5, criterion="entropy", class_weight=weights, random_state=0)

## Predicts output with multiple models
def models(train_x, train_y, cv_x, cv_y, weights):
    output_auc,output_aupr,output_dict=dict(),dict(),dict()
    # Linear Regression & SVM
    for c in [1, 3, 7, 10, 30, 100]: # multiple regularization parameters
        # Linear Regression
        current_model = LR(c, weights)
        current_model.fit(train_x, train_y)
        predicted = current_model.predict_proba(cv_x)
        output_auc['LR_'+str(c)] = roc_auc_score(cv_y, predicted[:, 1])
        output_aupr['LR_'+str(c)] = average_precision_score(cv_y, predicted[:, 1], pos_label=1)
        output_dict['LR_'+str(c)] = (output_auc['LR_'+str(c)]+output_aupr['LR_'+str(c)])/2
        # SVM
        for k in ['linear', 'poly', 'rbf']:
            current_model = SVM(k, c, weights)
            current_model.fit(train_x, train_y)
            predicted = current_model.predict_proba(cv_x)
            output_auc['SVM_'+k+'_'+str(c)] = roc_auc_score(cv_y, predicted[:, 1])
            output_aupr['SVM_'+k+'_'+str(c)] = average_precision_score(cv_y, predicted[:, 1], pos_label=1)
            output_dict['SVM_'+k+'_'+str(c)] = (output_auc['SVM_'+k+'_'+str(c)]+output_aupr['SVM_'+k+'_'+str(c)])/2
    # Random Forest
    current_model = RF(weights)
    current_model.fit(train_x, train_y)
    predicted = current_model.predict_proba(cv_x)
    output_auc['RF'] = roc_auc_score(cv_y, predicted[:, 1])
    output_aupr['RF'] = average_precision_score(cv_y, predicted[:, 1], pos_label=1)
    output_dict['RF'] = (output_auc['RF']+output_aupr['RF'])/2
    return output_auc,output_aupr,output_dict


# Preparation for prediction on the validation set
def prep_validation(autoencoder, esetSC2Var, anoSC2, val_ind, val_names):
    encoder = Model(inputs=autoencoder.input, outputs=autoencoder.layers[3].output) # in the 7 layer case
    encoded = encoder.predict(esetSC2Var, steps=1)
    val_x = np.append(encoded[val_ind,], anoSC2.loc[val_names, ['GA']].to_numpy(), 1)
    return val_x

def run(esetSC2, anoSC2, current_comparison, diffGenes, autoencoder):
    anoSC2 = anoSC2[['IndividualID', 'GA', 'Group', 'Train']]

    ## Preprocess data for dimensionality reduction
    esetSC2Var = esetSC2.loc[diffGenes]
    esetSC2Var = esetSC2Var.values
    esetSC2Var = esetSC2Var / np.max(esetSC2Var)
    esetSC2Var = transpose(esetSC2Var)

    ## DREAM Train and test data given were named as "play" and "validation" sets
    play_names = list(anoSC2[anoSC2['Train']==1].index.values)
    val_names = sorted(list(set(anoSC2.index.values)-set(play_names)))
    ctrl_vs_exp_names = list(anoSC2[anoSC2['Group']=='Control'].append(anoSC2[anoSC2['Group']==current_comparison]).index.values)

    play_ind = [esetSC2.columns.get_loc(i) for i in play_names]
    val_ind = [esetSC2.columns.get_loc(i) for i in val_names]
    ctrl_vs_exp_ind = [esetSC2.columns.get_loc(i) for i in ctrl_vs_exp_names]

    ## Preprocess play labels
    play_y0 = [i[0] for i in anoSC2.loc[play_names, ['Group']].values.tolist()]
    
    ## Cross Validation Preparation
    cv_num = 5 # Number of cross validations
    cv_set_size = round(len(ctrl_vs_exp_names)/cv_num) # Size of cross validation sets
    cnt_ctrl = len([i for i in play_y0 if i=='Control'])
    cnt_exp = len([i for i in play_y0 if i==current_comparison])
    weights_ctrl_vs_exp = None #{0:1, 1:round(cnt_ctrl/cnt_exp)} # key: 0==Control, 1==Experimental

    ## Call trained autoencoder models to get new dimensions
    # Model after Dimensionality Reduction
    encoder = Model(inputs=autoencoder.input, outputs=autoencoder.layers[3].output) # in the 7 layer case
    encoded = encoder.predict(esetSC2Var, steps=1)
    # separated data after dimensionality reduction; 
    # added 'GA' feature as well; NOT 'GADel', because NA in validation data
    play_x = np.append(encoded[play_ind,], anoSC2.loc[play_names, ['GA']].to_numpy(), 1) 
    # find the model with best performance
    temp_auc = dict()
    temp_aupr = dict()
    temp_ctrl_vs_exp = dict()
    train_x, train_y, cv_x, cv_y = None, None, None, None
    for ind in range(cv_num): # ind also used as a random seed
        # Control(0) vs Exp (1)
        train_x, train_y, cv_x, cv_y = split(play_x, play_y0, current_comparison, cv_set_size, ind)
        output_auc,output_aupr,output_dict = models(train_x, train_y, cv_x, cv_y, weights_ctrl_vs_exp)
        for key in output_dict:
            if key not in temp_ctrl_vs_exp:
                temp_ctrl_vs_exp[key] = 0
                temp_auc[key] = 0
                temp_aupr[key] = 0
            temp_ctrl_vs_exp[key] += output_dict[key]
            temp_auc[key] += output_auc[key]
            temp_aupr[key] += output_aupr[key]
    fin_ctrl_vs_exp = {key:temp_ctrl_vs_exp[key]/cv_num for key in temp_ctrl_vs_exp}
    fin_auc = {key:temp_auc[key]/cv_num for key in temp_auc}
    fin_aupr = {key:temp_aupr[key]/cv_num for key in temp_aupr}
    fin_rec = sorted(fin_ctrl_vs_exp.items(), key=lambda item: item[1], reverse=True)[0][0]
    # print('# Control(0) vs exp(1)', sorted(fin_ctrl_vs_exp.items(), key=lambda item: item[1], reverse=True))
    # print(fin_auc[fin_rec], fin_aupr[fin_rec])

    ## Final Prediction on Validation Set
    # Individual ID
    individual_id = [i[0] for i in anoSC2.loc[val_names,['IndividualID']].values.tolist()]
    val_x = prep_validation(autoencoder, esetSC2Var, anoSC2, val_ind, val_names)
    ctrl_vs_exp_x = np.append(train_x, cv_x, 0)
    ctrl_vs_exp_y = train_y + cv_y

    fin_param = fin_rec.split('_')
    fin_model = None
    if fin_param[0]=='LR': fin_model = LR(int(fin_param[1]), weights_ctrl_vs_exp)
    elif fin_param[0]=='SVM': fin_model = SVM(fin_param[1], int(fin_param[2]), weights_ctrl_vs_exp)
    elif fin_param[0]=='RF': fin_model = RF(weights_ctrl_vs_exp)
    fin_model.fit(ctrl_vs_exp_x, ctrl_vs_exp_y)
    predicted = fin_model.predict_proba(val_x)
    ctrl_vs_exp_predicted = list(predicted[:,1])

    # Per individual
    ctrl_vs_exp_predicted_dict = dict()
    for ind in range(len(individual_id)):
        if individual_id[ind] not in ctrl_vs_exp_predicted_dict:
            ctrl_vs_exp_predicted_dict[individual_id[ind]]=[]
        ctrl_vs_exp_predicted_dict[individual_id[ind]].append(ctrl_vs_exp_predicted[ind])
    ctrl_vs_exp_predicted_per_indiv = {key:np.mean(ctrl_vs_exp_predicted_dict[key]) for key in ctrl_vs_exp_predicted_dict}
    return fin_rec, fin_ctrl_vs_exp[fin_rec], fin_auc[fin_rec], fin_aupr[fin_rec], ctrl_vs_exp_predicted_per_indiv
