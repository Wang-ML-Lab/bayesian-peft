import numpy as np
import re

def collect_metrics(input_strings):
    acc = []
    ece = []
    nll = []
    brier = []
    ood_acc = []
    ood_auc = []

    if 'ood_auc' in input_strings[0]:
        pattern = r"val_acc: ([\d.]+), val_ece: ([\d.]+), val_nll: ([\d.]+), val_brier: ([\d.]+), ood_acc: ([\d.]+), ood_auc: ([\d.]+)"
        for input_string in input_strings:
            matches = re.search(pattern, input_string)
            if matches:
                acc.append(float(matches.group(1)) * 100)
                ece.append(float(matches.group(2)) * 100)
                nll.append(float(matches.group(3)))
                brier.append(float(matches.group(4)))
                ood_acc.append(float(matches.group(5)))
                ood_auc.append(float(matches.group(6)))

        return acc, ece, nll, brier, ood_acc, ood_auc

    elif 'val_brier' in input_strings[0]:
        pattern = r"val_acc: ([\d.]+), val_ece: ([\d.]+), val_nll: ([\d.]+), val_brier: ([\d.]+)"
        for input_string in input_strings:
            matches = re.search(pattern, input_string)
            if matches:
                acc.append(float(matches.group(1)) * 100)
                ece.append(float(matches.group(2)) * 100)
                nll.append(float(matches.group(3)))
                brier.append(float(matches.group(4)))

        return acc, ece, nll, brier, None, None

    else:
        pattern = r"val_acc: ([\d.]+), val_ece: ([\d.]+), val_nll: ([\d.]+)"
        for input_string in input_strings:
            matches = re.search(pattern, input_string)
            if matches:
                acc.append(float(matches.group(1)) * 100)
                ece.append(float(matches.group(2)) * 100)
                nll.append(float(matches.group(3)))

        return acc, ece, nll, None, None, None


file_path = '/common/home/yw1131/repos/BBBlora/checkpoints/blob/meta-llama/Llama-2-7b-hf/dataset/blob-dataset-sample10-eps0.05-kllr0.0025-beta0.05-gamma8-seed'
datasets = ['winogrande_s']
# datasets = ['winogrande_s', 'ARC-Challenge', 'ARC-Easy', 'winogrande_m', 'obqa', 'boolq']
# datasets = ['rte', 'mrpc', 'wic', 'cola', 'boolq']

all_acc = []
all_ece = []
all_nll = []

for dataset in datasets:
    input_strings = []
    file_path_new = file_path.replace('dataset', dataset)
    
    for i in range(1, 4):
        with open(file_path_new + str(i) + '/log.txt', 'r') as file:
            log_data = file.read()
        lines = log_data.strip().split('\n')

        last_val_acc_line = ""
        for line in lines:
            if "val_acc" in line:
                last_val_acc_line = line

        if last_val_acc_line:
            result = "val_acc" + last_val_acc_line.split("val_acc")[-1].strip()
        else:
            result = ""

        input_strings.append(result)

    acc, ece, nll, _, _, _ = collect_metrics(input_strings)
    all_acc.append(acc)
    all_ece.append(ece)
    all_nll.append(nll)

print('acc:', end=' ')
for acc in all_acc:
    print('%.2f\\scriptsize{$\\pm$%.2f} &' % (np.mean(acc), np.std(acc)), end=' ')
print()

print('ece:', end=' ')
for ece in all_ece:
    print('%.2f\\scriptsize{$\\pm$%.2f} &' % (np.mean(ece), np.std(ece)), end=' ')
print()

print('nll:', end=' ')
for nll in all_nll:
    print('%.4f\\scriptsize{$\\pm$%.4f} &' % (np.mean(nll), np.std(nll)), end=' ')
print()
# import numpy as np
# import re
# import torch
# def print_all(input_strings):
#     acc = []
#     ece = []
#     nll = []
#     brier = []
#     ood_acc = []
#     ood_auc = []

#     if 'ood_auc' in input_strings[0]:
#         pattern = r"val_acc: ([\d.]+), val_ece: ([\d.]+), val_nll: ([\d.]+), val_brier: ([\d.]+), ood_acc: ([\d.]+), ood_auc: ([\d.]+)"
#         for input_string in input_strings:
#             matches = re.search(pattern, input_string)

#             if matches:
#                 val_acc = float(matches.group(1))  
#                 val_ece = float(matches.group(2))  
#                 val_nll = float(matches.group(3))
#                 val_brier = float(matches.group(4)) 
#                 val_ood_acc = float(matches.group(5))
#                 val_ood_auc = float(matches.group(6))
#                 acc.append(val_acc)
#                 ece.append(val_ece)
#                 nll.append(val_nll)
#                 brier.append(val_brier)
#                 ood_acc.append(val_ood_acc)
#                 ood_auc.append(val_ood_auc)

#         print(len(acc))

#         acc = np.array(acc) * 100
#         ece = np.array(ece) * 100


#         print('acc:%.2f\scriptsize{$\pm$%.2f}'%(np.mean(acc), np.std(acc)))
#         print('ece:%.2f\scriptsize{$\pm$%.2f}'%(np.mean(ece), np.std(ece)))
#         print('nll:%.2f\scriptsize{$\pm$%.2f}'%(np.mean(nll), np.std(nll)))
#         print('brier:%.2f\scriptsize{$\pm$%.2f}'%(np.mean(brier), np.std(brier)))
#         print('ood_acc:%.2f\scriptsize{$\pm$%.2f}'%(np.mean(ood_acc), np.std(ood_acc)))
#         print('ood_auc:%.2f\scriptsize{$\pm$%.2f}'%(np.mean(ood_auc), np.std(ood_auc)))
#         print('.........')
#         print('acc:%.2f(%.2f)'%(np.mean(acc), np.std(acc)))
#         print('ece:%.2f(%.2f)'%(np.mean(ece), np.std(ece)))
#         print('nll:%.4f(%.4f)'%(np.mean(nll), np.std(nll)))
#         print('brier:%.2f(%.2f)'%(np.mean(brier), np.std(brier)))
#         print('ood_acc:%.2f(%.2f)'%(np.mean(ood_acc), np.std(ood_acc)))
#         print('ood_auc:%.2f(%.2f)'%(np.mean(ood_auc), np.std(ood_auc)))
#     elif 'val_brier' in input_strings[0]:
#         pattern = r"val_acc: ([\d.]+), val_ece: ([\d.]+), val_nll: ([\d.]+), val_brier: ([\d.]+)"
#         for input_string in input_strings:
#             matches = re.search(pattern, input_string)
#             if matches:
#                 val_acc = float(matches.group(1))  
#                 val_ece = float(matches.group(2)) 
#                 val_nll = float(matches.group(3)) 
#                 val_brier = float(matches.group(4))  
#                 acc.append(val_acc)
#                 ece.append(val_ece)
#                 nll.append(val_nll)
#                 brier.append(val_brier)
#         print(len(acc))

#         acc = np.array(acc) * 100
#         ece = np.array(ece) * 100


#         print('acc:%.2f\scriptsize{$\pm$%.2f} &'%(np.mean(acc), np.std(acc)))
#         print('ece:%.2f\scriptsize{$\pm$%.2f} &'%(np.mean(ece), np.std(ece)))
#         print('nll:%.2f\scriptsize{$\pm$%.2f} &'%(np.mean(nll), np.std(nll)))
#         print('brier:%.2f\scriptsize{$\pm$%.2f} &'%(np.mean(brier), np.std(brier)))

#         print('.........')
#         print('acc:%.2f(%.2f)'%(np.mean(acc), np.std(acc)))
#         print('ece:%.2f(%.2f)'%(np.mean(ece), np.std(ece)))
#         print('nll:%.4f(%.4f)'%(np.mean(nll), np.std(nll)))
#         print('brier:%.2f(%.2f)'%(np.mean(brier), np.std(brier)))
#     else:
#         pattern = r"val_acc: ([\d.]+), val_ece: ([\d.]+), val_nll: ([\d.]+)"
#         for input_string in input_strings:
#             matches = re.search(pattern, input_string)

#             if matches:
#                 val_acc = float(matches.group(1))  
#                 val_ece = float(matches.group(2)) 
#                 val_nll = float(matches.group(3)) 
#                 acc.append(val_acc)
#                 ece.append(val_ece)
#                 nll.append(val_nll)
#         print(len(acc))

#         acc = np.array(acc) * 100
#         ece = np.array(ece) * 100


#         print('acc:%.2f\scriptsize{$\pm$%.2f} &'%(np.mean(acc), np.std(acc)))
#         print('ece:%.2f\scriptsize{$\pm$%.2f} &'%(np.mean(ece), np.std(ece)))
#         print('nll:%.4f\scriptsize{$\pm$%.4f} &'%(np.mean(nll), np.std(nll)))
#         print('.........')
#         print('acc:%.2f(%.2f)'%(np.mean(acc), np.std(acc)))
#         print('ece:%.2f(%.2f)'%(np.mean(ece), np.std(ece)))
#         print('nll:%.4f(%.4f)'%(np.mean(nll), np.std(nll)))



# file_path = '/home/yibinwang/research/BBBlora/checkpoints/mcdropout/meta-llama/Llama-2-7b-hf/dataset/mcdropout-dataset-sample10-seed'
# for dataset in ['winogrande_s', 'ARC-Challenge', 'ARC-Easy', 'winogrande_m', 'obqa', 'boolq']:
# # for dataset in ['rte', 'mrpc', 'wic', 'cola', 'boolq']:
# # for dataset in ['winogrande_s']:   
#     input_strings=[]
#     file_path_new = file_path.replace('dataset', dataset)
#     print('dataset:', dataset)
#     for i in range(1, 4):
#         with open(file_path_new + str(i) + '/log.txt', 'r') as file:
#             log_data = file.read()
#         # Split the log data into lines
#         lines = log_data.strip().split('\n')

#         # Initialize variables to keep track of the last "val_acc" line
#         last_val_acc_line = ""

#         # Iterate over the lines to find the last occurrence of "val_acc"
#         for line in lines:
#             if "val_acc" in line:
#                 last_val_acc_line = line

#         # Extract the content after "val_acc" in the last "val_acc" line
#         if last_val_acc_line:
#             result = "val_acc" + last_val_acc_line.split("val_acc")[-1].strip()
#         else:
#             result = ""

#         input_strings.append(result)
#     print_all(input_strings)
#     print('=================================')



