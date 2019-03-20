#Each of the Models are run independently and the final part of this code aggregates the AVERAGE ACCURACIES AND STD ERRORS to generate a Bokeh Plot with Error Bars
#Sorry for the cluttered code, I could have condensed it a lot more but was under time constrain. 

import pandas as pd
import numpy as np
import preprocess_assg3_fns as pp
import nbc
import lr_svm_fns as lr_svm
import statistics
import math


print("Logistic Regression CV Started")
#LR
pp.initial_cleanup_3()
df = pd.read_csv("trainingSet.csv")
trainingSet = df.sample(frac=1, random_state=18)
trainingSet.index = pd.RangeIndex(len(trainingSet.index))
trainingSet.index += 1 

LR_SVM_Set = pp.preprocess_LR_SVM_0(trainingSet)
S = []
for i in np.array_split(LR_SVM_Set, 10):
    S.append(i)
Sc = []
for j in range(len(S)):
    ind = list(range(len(S)))
    ind.remove(j)
    train = S[0].copy()
    train = train[0:0]
    for i in ind:
        train = pd.concat([train,S[i]])
    Sc.append(train)

F = [0.025, 0.05, 0.075, 0.1, 0.15, 0.2]
lr_training_accuracies_avg = []
lr_testing_accuracies_avg = []
lr_training_stderror = []
lr_testing_stderror = []
for t_frac in F:
    idx = list(range(10))
    lr_training_accuracies = []
    lr_testing_accuracies = []
    for i in idx:
        test_set = S[i]
        train_set = Sc[i]
        train_set = train_set.sample(frac=t_frac, random_state=18)
        #run LR
        training_acc, testing_acc = lr_svm.lr_0(trainingSet=train_set,testSet=test_set)
        lr_training_accuracies.append(training_acc)
        lr_testing_accuracies.append(testing_acc)
    lr_training_accuracies_avg.append(statistics.mean(lr_training_accuracies))
    lr_testing_accuracies_avg.append(statistics.mean(lr_testing_accuracies))
    lr_training_stderror.append((statistics.stdev(lr_training_accuracies))/math.sqrt(10))
    lr_testing_stderror.append((statistics.stdev(lr_testing_accuracies))/math.sqrt(10))
        
print("SVM CV  Started")
#SVM
pp.initial_cleanup_3()
df = pd.read_csv("trainingSet.csv")
trainingSet = df.sample(frac=1, random_state=18)
trainingSet.index = pd.RangeIndex(len(trainingSet.index))
trainingSet.index += 1 

LR_SVM_Set = pp.preprocess_LR_SVM_0(trainingSet)
S = []
for i in np.array_split(LR_SVM_Set, 10):
    S.append(i)
Sc = []
for j in range(len(S)):
    ind = list(range(len(S)))
    ind.remove(j)
    train = S[0].copy()
    train = train[0:0]
    for i in ind:
        train = pd.concat([train,S[i]])
    Sc.append(train)

F = [0.025, 0.05, 0.075, 0.1, 0.15, 0.2]
svm_training_accuracies_avg = []
svm_testing_accuracies_avg = []
svm_training_stderror = []
svm_testing_stderror = []
for t_frac in F:
    print("t_frac: ",t_frac)
    idx = list(range(10))
    print(idx)
    svm_training_accuracies = []
    svm_testing_accuracies = []
    for i in idx:
        print("idx: ",i)
        test_set = S[i]
        train_set = Sc[i]
        train_set = train_set.sample(frac=t_frac, random_state=18)
        #run SVM
        training_acc, testing_acc = lr_svm.svm(trainingSet=train_set,testSet=test_set)
        svm_training_accuracies.append(training_acc)
        svm_testing_accuracies.append(testing_acc)
    svm_training_accuracies_avg.append(statistics.mean(svm_training_accuracies))
    svm_testing_accuracies_avg.append(statistics.mean(svm_testing_accuracies))
    svm_training_stderror.append((statistics.stdev(svm_training_accuracies))/math.sqrt(10))
    svm_testing_stderror.append((statistics.stdev(svm_testing_accuracies))/math.sqrt(10))
    
print("NBC CV Started")
#NBC
pp.initial_cleanup_3()
df = pd.read_csv("trainingSet.csv")
trainingSet = df.sample(frac=1, random_state=18)
trainingSet.index = pd.RangeIndex(len(trainingSet.index))
trainingSet.index += 1 
NBC_Set = pp.preprocess_NBC(trainingSet)
S = []
for i in np.array_split(NBC_Set, 10):
    S.append(i)
Sc = []
for j in range(len(S)):
    ind = list(range(len(S)))
    ind.remove(j)
    train = S[0].copy()
    train = train[0:0]
    for i in ind:
        train = pd.concat([train,S[i]])
    Sc.append(train)

F = [0.025, 0.05, 0.075, 0.1, 0.15, 0.2]
nbc_training_accuracies_avg = []
nbc_testing_accuracies_avg = []
nbc_training_stderror = []
nbc_testing_stderror = []
for t_frac in F:
    print("t_frac: ",t_frac)
    idx = list(range(10))
    nbc_training_accuracies = []
    nbc_testing_accuracies = []
    for i in idx:
        print("idx: ",i)
        test_set = S[i]
        train_set = Sc[i]
        training_acc, testing_acc = hw2.nbc(t_frac=t_frac,trainingSet=train_set,testSet=test_set)
        nbc_training_accuracies.append(training_acc)
        nbc_testing_accuracies.append(testing_acc)
    nbc_training_accuracies_avg.append(statistics.mean(nbc_training_accuracies))
    nbc_testing_accuracies_avg.append(statistics.mean(nbc_testing_accuracies))
    nbc_training_stderror.append((statistics.stdev(svm_training_accuracies))/math.sqrt(10))
    nbc_testing_stderror.append((statistics.stdev(svm_testing_accuracies))/math.sqrt(10))
    

#Plotting
from bokeh.io import output_file
from bokeh.models import ColumnDataSource, Whisker
from bokeh.plotting import figure, show 


#Standard Error Bars creation
NBC_lower = []
NBC_upper = []
for i in range(len(nbc_testing_accuracies_avg)):
    NBC_lower.append(nbc_testing_accuracies_avg[i]-nbc_testing_stderror[i])
    NBC_upper.append(nbc_testing_accuracies_avg[i]+nbc_training_stderror[i])  
    
LR_lower = []
LR_upper = []
for i in range(len(lr_testing_accuracies_avg)):
    LR_lower.append(lr_testing_accuracies_avg[i]-lr_testing_stderror[i])
    LR_upper.append(lr_testing_accuracies_avg[i]+lr_testing_stderror[i])  
    
SVM_lower = []
SVM_upper = []
for i in range(len(svm_training_accuracies_avg)):
    SVM_lower.append(svm_testing_accuracies_avg[i]-svm_testing_stderror[i])
    SVM_upper.append(svm_testing_accuracies_avg[i]+svm_testing_stderror[i])  
    
#Rescaling T-Frac into Total Training Data Sample Size
Total_Sample_Size = []
for i in F:
    Total_Sample_Size.append(i*5200)

def learning_curve():
    source = ColumnDataSource(data=dict(TSS=Total_Sample_Size, NBC_training_acc = nbc_training_accuracies_avg,
                                                 LR_training_acc = lr_training_accuracies_avg, 
                                                  SVM_training_acc= svm_training_accuracies_avg,
                                                 NBC_testing_acc = nbc_testing_accuracies_avg,
                                                 LR_testing_acc = lr_testing_accuracies_avg,
                                                 SVM_testing_acc = svm_testing_accuracies_avg,
                                       NBC_lower = NBC_lower, NBC_upper = NBC_upper, LR_lower = LR_lower, LR_upper = LR_upper
                                       ,SVM_lower = SVM_lower, SVM_upper = SVM_upper))

    p = figure(title="Model Evaluation")

    p.circle(x='TSS', y='NBC_testing_acc', source=source, size=9, legend="NBC", alpha=0.6)
    p.line(x='TSS', y='NBC_testing_acc', source=source, legend="NBC")

    p.circle(x='TSS', y='LR_testing_acc', fill_color=None, line_color="red", source=source, size=9, legend="LR")
    p.line(x='TSS', y='LR_testing_acc', line_color="red", line_width=2, source=source, legend="LR")

    p.square(x='TSS', y='SVM_testing_acc', fill_color="green", line_color="green", source=source, size=9, legend="SVM", alpha=0.6)
    p.line(x='TSS', y='SVM_testing_acc', line_color="green", source=source, legend="SVM")

    p.xaxis.axis_label = 'Total Training Data Size'
    p.yaxis.axis_label = 'Accuracy'

    w1 = Whisker(source=source, base='TSS', upper='NBC_upper', lower='NBC_lower', line_color='blue')
    w1.upper_head.line_color = 'blue' 
    w1.lower_head.line_color = 'blue'
    p.add_layout(w1)
    w2 = Whisker(source=source, base='TSS', upper='LR_upper', lower='LR_lower', line_color='red')
    w2.upper_head.line_color = 'red' 
    w2.lower_head.line_color = 'red'
    p.add_layout(w2)
    w3 = Whisker(source=source, base='TSS', upper='SVM_upper', lower='SVM_lower', line_color='green')
    w3.upper_head.line_color = 'green' 
    w3.lower_head.line_color = 'green'
    p.add_layout(w3)
    p.legend.location = "top_center"
    show(p)

learning_curve()
