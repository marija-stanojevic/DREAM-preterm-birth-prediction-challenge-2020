## Team ZO_script.py

import pandas as pd
import numpy as np
import pickle as pc
import os, sys, random, warnings
import featureSelection
import dimReduction
import myPrediction

warnings.filterwarnings("ignore")

## Three options: (1) run a pre-trained (2) run one new model (3) multiRun
mode=0 
if len(sys.argv)>1:
    if sys.argv[1]=='new':
        mode=1
        print("######## Running a new model... ########")
    elif sys.argv[1]=='multiRun':
        mode=2
        print("##### MultiRun, to be used with: 'zsh multiRun.sh'")
    else:
        print("######## Not a known input, will run the prerun reproducible model. If done in error, try 'new' instead. ########")
else:
    if len([l for l in os.listdir() if l.endswith('.pc')])==2:
        print("######## Running a pre-trained model from a saved file. If you want to run a new model:'python3 Team\ ZO_script.py new'  ########")
    else:
        mode=1
        print("######## There is at least one prerun .pc file missing. Running a new model... ########")

## DREAM PTB SC2 Fixed Parameters
anoSC2Name = 'anoSC2_v20_nokey.csv'
esetSC2Name = 'esetSC2.csv'
outName = 'Team ZO_prediction.csv'
maxGeneCount = 100
random_seed = 1205

## dimReduction Parameters
hidden_size = 200
hidden_size2 = 80
code_sizeList = [20,30,40,50,60,70,80] ## == code
activ1='relu'
optim='adam'
noise_factor = 0.4
os.environ['PYTHONHASHSEED']=str(random_seed)

## Default Variables
esetSC2 = pd.read_csv(esetSC2Name, index_col=0)
anoSC2 = pd.read_csv(anoSC2Name, index_col=0)
anoSC2 = anoSC2.loc[esetSC2.columns.values]

Control_id = anoSC2.index.values[anoSC2['Group']=="Control"]
sPTD_id = anoSC2.index.values[anoSC2['Group']=="sPTD"]
PPROM_id = anoSC2.index.values[anoSC2['Group']=="PPROM"]

sPTDDiffGenes, PPROMDiffGenes = None, None
sPTD_model_ind, sPTD_autoencoder = None, None
PPROM_model_ind, PPROM_autoencoder = None, None

## Feature Selection
Control_exp = esetSC2[Control_id]
random.seed(random_seed)
np.random.seed(random_seed)
sPTD_exp = esetSC2[sPTD_id]
sPTD_exp_sample = sPTD_exp.sample(n=Control_exp.shape[1], replace=True, random_state=1, axis=1)
random.seed(random_seed)
np.random.seed(random_seed)
PPROM_exp = esetSC2[PPROM_id]
PPROM_exp_sample = PPROM_exp.sample(n=Control_exp.shape[1], replace=True, random_state=1, axis=1)
sPTDDiffGenes, PPROMDiffGenes = featureSelection.run(esetSC2, anoSC2, Control_exp, sPTD_exp_sample, PPROM_exp_sample, maxGeneCount)
print(len(sPTDDiffGenes), len(PPROMDiffGenes))

## (1) New run #################################################################
if mode==1: 
         
    ## Dimensionality Reduction via Autoencoder
    sPTD_model_ind, sPTD_autoencoder = dimReduction.run(esetSC2, sPTDDiffGenes, random_seed, hidden_size, hidden_size2, code_sizeList, activ1, optim, noise_factor)
    PPROM_model_ind, PPROM_autoencoder = dimReduction.run(esetSC2, PPROMDiffGenes, random_seed, hidden_size, hidden_size2, code_sizeList, activ1, optim, noise_factor)
    if 'sPTD_pre-trained_model.pc' not in os.listdir():
        with open('Team ZO_sPTD_pre-trained_model.pc','wb') as fileh:
            pc.dump(sPTD_autoencoder, fileh)
    if 'PPROM_pre-trained_model.pc' not in os.listdir():
        with open('Team ZO_PPROM_pre-trained_model.pc','wb') as fileh:
            pc.dump(PPROM_autoencoder, fileh)
    ## Prediction
    sPTD_rec, sPTD_mean, sPTD_auc, sPTD_aupr, sPTD_finDict = myPrediction.run(esetSC2, anoSC2, 'sPTD', sPTDDiffGenes, sPTD_autoencoder)
    PPROM_rec, PPROM_mean, PPROM_auc, PPROM_aupr, PPROM_finDict = myPrediction.run(esetSC2, anoSC2, 'PPROM', PPROMDiffGenes, PPROM_autoencoder)
    print('sPTD', 'AUC:', sPTD_auc, 'AUPR:', sPTD_aupr, 'Mean:', sPTD_mean)
    print('PPROM', 'AUC:', PPROM_auc, 'AUPR:', PPROM_aupr, 'Mean:', PPROM_mean)
    with open(outName,'w') as writeFile:
        writeFile.write(','.join(['IndividualID', 'sPTD', 'PPROM'])+'\n')
        for key in sorted(list(sPTD_finDict)):
            writeFile.write(','.join([str(key), str(sPTD_finDict[key]), str(PPROM_finDict[key])])+'\n')

## (2) Multi run #################################################################
elif mode==2: 
    ## Feature Selection
    random.seed(random_seed)
    np.random.seed(random_seed)
    sPTD_exp_sample = sPTD_exp.sample(n=Control_exp.shape[1], replace=True, random_state=1, axis=1)
    random.seed(random_seed)
    np.random.seed(random_seed)
    PPROM_exp = esetSC2[PPROM_id]
    PPROM_exp_sample = PPROM_exp.sample(n=Control_exp.shape[1], replace=True, random_state=1, axis=1)
    sPTDDiffGenes, PPROMDiffGenes = featureSelection.run(esetSC2, anoSC2, Control_exp, sPTD_exp_sample, PPROM_exp_sample, maxGeneCount)

    ## Dimensionality Reduction via Autoencoder
    sPTD_model_ind, sPTD_autoencoder = dimReduction.run(esetSC2, sPTDDiffGenes, random_seed, hidden_size, hidden_size2, code_sizeList, activ1, optim, noise_factor)
    PPROM_model_ind, PPROM_autoencoder = dimReduction.run(esetSC2, PPROMDiffGenes, random_seed, hidden_size, hidden_size2, code_sizeList, activ1, optim, noise_factor)

    ## Prediction
    sPTD_rec, sPTD_mean, sPTD_auc, sPTD_aupr, sPTD_finDict = myPrediction.run(esetSC2, anoSC2, 'sPTD', sPTDDiffGenes, sPTD_autoencoder)
    PPROM_rec, PPROM_mean, PPROM_auc, PPROM_aupr, PPROM_finDict = myPrediction.run(esetSC2, anoSC2, 'PPROM', PPROMDiffGenes, PPROM_autoencoder)
    print(sPTD_rec, sPTD_auc, sPTD_aupr, sPTD_mean, PPROM_rec, PPROM_auc, PPROM_aupr, PPROM_mean)
    with open('multirun.txt','a') as simFile:
        simFile.write(','.join([str(w) for w in [sPTD_model_ind, sPTD_rec, sPTD_auc, sPTD_aupr, sPTD_mean, PPROM_model_ind, PPROM_rec, PPROM_auc, PPROM_aupr, PPROM_mean]])+'\n')
    if 'save.txt' not in os.listdir():
        with open('save.txt','w') as writeFile:
            writeFile.write(','.join([str(w) for w in [sPTD_model_ind, sPTD_rec, sPTD_auc, sPTD_aupr, sPTD_mean, PPROM_model_ind, PPROM_rec, PPROM_auc, PPROM_aupr, PPROM_mean]])+'\n')
        with open('sPTD_pre-trained_model.pc','wb') as fileh:
            pc.dump(sPTD_autoencoder, fileh)
        with open('PPROM_pre-trained_model.pc','wb') as fileh:
            pc.dump(PPROM_autoencoder, fileh)
    else:
        prev_sPTD_mean, prev_PPROM_mean= None, None
        with open('save.txt','r') as myFile:
            for l in myFile:
                [prev_sPTD_model_ind, prev_sPTD_rec, prev_sPTD_auc, prev_sPTD_aupr, prev_sPTD_mean, \
                prev_PPROM_model_ind, prev_PPROM_rec, prev_PPROM_auc, prev_PPROM_aupr, prev_PPROM_mean]=l.rstrip().split(',')
                break
        # sPTD 
        if float(prev_sPTD_mean)<sPTD_mean: # updated
             with open('sPTD_pre-trained_model.pc','wb') as fileh:
                pc.dump(sPTD_autoencoder, fileh)
        else: # not updated
            [sPTD_model_ind, sPTD_rec, sPTD_auc, sPTD_aupr, sPTD_mean, \
            PPROM_model_ind, PPROM_rec, PPROM_auc, PPROM_aupr, PPROM_mean] = [prev_sPTD_model_ind, prev_sPTD_rec, prev_sPTD_auc, prev_sPTD_aupr, prev_sPTD_mean, \
                                                                            PPROM_model_ind, PPROM_rec, PPROM_auc, PPROM_aupr, PPROM_mean]
        # PPROM
        if float(prev_PPROM_mean)<PPROM_mean: # updated
            with open('PPROM_pre-trained_model.pc','wb') as fileh:
                pc.dump(PPROM_autoencoder, fileh)
        else: # not updated
            [sPTD_model_ind, sPTD_rec, sPTD_auc, sPTD_aupr, sPTD_mean, \
            PPROM_model_ind, PPROM_rec, PPROM_auc, PPROM_aupr, PPROM_mean] = [sPTD_model_ind, sPTD_rec, sPTD_auc, sPTD_aupr, sPTD_mean, \
                                                                            prev_PPROM_model_ind, prev_PPROM_rec, prev_PPROM_auc, prev_PPROM_aupr, prev_PPROM_mean]
        with open('save.txt','w') as writeFile:
            writeFile.write(','.join([str(w) for w in [sPTD_model_ind, sPTD_rec, sPTD_auc, sPTD_aupr, sPTD_mean, PPROM_model_ind, PPROM_rec, PPROM_auc, PPROM_aupr, PPROM_mean]])+'\n')

## (3) Pre-trained model run #################################################################
else: 
    with open('sPTD_pre-trained_model.pc','rb') as fileh:
        sPTD_autoencoder = pc.load(fileh)
    with open('PPROM_pre-trained_model.pc','rb') as fileh:
        PPROM_autoencoder = pc.load(fileh)
    ## Prediction
    sPTD_rec, sPTD_mean, sPTD_auc, sPTD_aupr, sPTD_finDict = myPrediction.run(esetSC2, anoSC2, 'sPTD', sPTDDiffGenes, sPTD_autoencoder)
    PPROM_rec, PPROM_mean, PPROM_auc, PPROM_aupr, PPROM_finDict = myPrediction.run(esetSC2, anoSC2, 'PPROM', PPROMDiffGenes, PPROM_autoencoder)
    print('sPTD', 'AUC:', sPTD_auc, 'AUPR:', sPTD_aupr, 'Mean:', sPTD_mean)
    print('PPROM', 'AUC:', PPROM_auc, 'AUPR:', PPROM_aupr, 'Mean:', PPROM_mean)
    with open(outName,'w') as writeFile:
        writeFile.write(','.join(['IndividualID', 'sPTD', 'PPROM'])+'\n')
        for key in sorted(list(sPTD_finDict)):
            writeFile.write(','.join([str(key), str(sPTD_finDict[key]), str(PPROM_finDict[key])])+'\n')
