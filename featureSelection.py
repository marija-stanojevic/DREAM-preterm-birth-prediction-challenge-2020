## feature selection

from scipy.stats import ttest_ind as ttest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math, statistics

def getDiffGenes(esetSC2, log2FC, neglog10pval, f_thres, p_thres, outlier_sd_fold):
    genes = esetSC2.index.values
    log2FC_mean = np.mean(log2FC)
    log2FC_stdev = statistics.stdev(log2FC)
    neglog10pval_mean = np.mean(neglog10pval)
    neglog10pval_stdev = statistics.stdev(neglog10pval)
    diff_genes1 = set()
    diff_genes2 = set()
    colors = ['#000000' for r in range(len(log2FC))]
    for i in range(len(log2FC)):
        ## volcano
        if abs(log2FC[i])>f_thres and neglog10pval[i]>p_thres:
            diff_genes1.add(genes[i])
            colors[i] = '#FF2D00'
        ## outlier based on x-axis (log2FC)
        if abs(log2FC[i]-log2FC_mean)>(outlier_sd_fold*log2FC_stdev):
            diff_genes2.add(genes[i])
            colors[i] = '#FF2D00'
        ## outlier based on y-axis (neglog10pval)
        if abs(neglog10pval[i]-neglog10pval_mean)>(outlier_sd_fold*neglog10pval_stdev):
            diff_genes2.add(genes[i])
            colors[i] = '#FF2D00'
    diff_genes2 = diff_genes2 - diff_genes1
    return diff_genes1, diff_genes2, colors

def runDiffGenes(esetSC2, maxGeneCount, current_exp, log2FC, neglog10pval):
    fc_std = statistics.stdev(log2FC)
    pval_std = statistics.stdev(neglog10pval)
    sd_fold_list = [r/100 for r in range(int(2*100),int(5*100))]
    for s in sd_fold_list:
        f_thres = fc_std*s
        p_thres = pval_std*s
        outlier_sd_fold = s * 2.5
        set1, set2, colors = getDiffGenes(esetSC2, log2FC, neglog10pval, f_thres, p_thres, outlier_sd_fold)
        merged_set = set1|set2
        if len(merged_set) < maxGeneCount:
            ## Visualization
            # print(s, f_thres, p_thres, len(merged_set),len(set1),len(set2))
            # figure = plt.figure()
            # plt.scatter(log2FC, neglog10pval, facecolors='none', edgecolors=colors)
            # plt.xlabel('log2FC')
            # plt.ylabel('-log10pval')
            # figure.savefig(current_exp+'.png')
            # figure.clf()
            return merged_set

def run(esetSC2, anoSC2, Control_exp, sPTD_exp_sample, PPROM_exp_sample, maxGeneCount):
    sPTD_log2FC = [math.log2(np.mean(Control_exp.iloc[c]))-math.log2(np.mean(sPTD_exp_sample.iloc[c])) for c in range(Control_exp.shape[0])]
    PPROM_log2FC = [math.log2(np.mean(Control_exp.iloc[c]))-math.log2(np.mean(PPROM_exp_sample.iloc[c])) for c in range(Control_exp.shape[0])]

    sPTD_neglog10pval = [-math.log10(x) for x in ttest(Control_exp, sPTD_exp_sample, axis=1).pvalue]
    PPROM_neglog10pval = [-math.log10(x) for x in ttest(Control_exp, PPROM_exp_sample, axis=1).pvalue]

    sPTDDiffGenes = runDiffGenes(esetSC2, maxGeneCount, 'Control_vs_sPTD', sPTD_log2FC, sPTD_neglog10pval)
    PPROMDiffGenes = runDiffGenes(esetSC2, maxGeneCount, 'Control_vs_PPROM', PPROM_log2FC, PPROM_neglog10pval)
    return sPTDDiffGenes, PPROMDiffGenes
