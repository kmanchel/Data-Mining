import pandas as pd
import numpy as np
from bokeh.io import output_file, show
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure 

#NBC slightly modified from before. it will return accuracies now instead of printing them out.
def nbc(t_frac=1):
    df = pd.read_csv("trainingSet.csv")
    df = df.sample(frac=t_frac, random_state=47)
    success_count = df[(df.iloc[:,52] == 1)]['decision'].count()
    fail_count = df[(df.iloc[:,52] == 0)]['decision'].count()
    total_count = df['decision'].count()
    prob_success = success_count/total_count
    prob_fail = fail_count/total_count
    #Creating a dictionary of conditionary probabilities for all success scenarios
    cond_probabilities_1 = {}
    columns = list(range(0,52))
    for i in columns:
            factors = list(df.iloc[:,i].unique())
            vals = {}
            for j in factors:
                label_count_success = df[(df.iloc[:,52] == 1) & (df.iloc[:, i] == j)].iloc[:,i].count()
                success_count = df[(df.iloc[:,52] == 1)]['decision'].count()
                vals[j] = label_count_success/success_count
            cond_probabilities_1[i] = vals
        
    #Creating a dictionary of conditionary probabilities for all failure scenarios
    cond_probabilities_0 = {}
    columns = list(range(0,52))
    for i in columns:
            factors = list(df.iloc[:,i].unique())
            vals = {}
            for j in factors:
                label_count_fail = df[(df.iloc[:,52] == 0) & (df.iloc[:, i] == j)].iloc[:,i].count()
                fail_count = df[(df.iloc[:,52] == 0)]['decision'].count()
                vals[j] = label_count_fail/fail_count
            cond_probabilities_0[i] = vals

    def product_cond_probs(dec, df_row):
        row = df.iloc[df_row]
        columns = list(range(0,52))
        product = 1
        if (dec==1):
            for i in columns:
                factor = row[i]
                try:
                    product = product * float(cond_probabilities_1.get(i).get(factor))
                except:
                    product = product 
        elif (dec==0):
            for i in columns:
                factor = row[i]
                try:
                    product = product * float(cond_probabilities_0.get(i).get(factor))
                except:
                    product = product 
        return(product)
    #cond_probs = pd.DataFrame.from_dict(cond_probabilities)
    #cond_probs.to_csv('cond_probs.csv', index=False)
    
    #MAKING MY PREDICTIONS
    
    #Using Training Set:
    prediction = []
    for n in range(0,len(df)):
        numerator_success = product_cond_probs(dec = 1, df_row = n)*prob_success
        denominator = (product_cond_probs(dec = 1, df_row = n)*prob_success)+(product_cond_probs(dec = 0, df_row = n)*prob_fail)
        p_success = numerator_success/denominator
        numerator_fail = product_cond_probs(dec = 0, df_row = n)*prob_fail
        p_fail = numerator_fail/denominator
        if p_success > p_fail:
            prediction.append(1)
        else:
            prediction.append(0)
    df['prediction'] = prediction
    correct = df[(df['decision'] == df['prediction'])]['prediction'].count()
    training_acc = round(correct/len(df),2)
    df.to_csv('checking.csv', index=False)
    
    
    #Using Test Set:
    df = pd.read_csv("testSet.csv")
    prediction = []
    for n in range(0,len(df)):
        numerator_success = product_cond_probs(dec = 1, df_row = n)*prob_success
        denominator = (product_cond_probs(dec = 1, df_row = n)*prob_success)+(product_cond_probs(dec = 0, df_row = n)*prob_fail)
        p_success = numerator_success/denominator
        numerator_fail = product_cond_probs(dec = 0, df_row = n)*prob_success
        p_fail = numerator_fail/denominator
        if p_success > p_fail:
            prediction.append(1)
        else:
            prediction.append(0)
    df['prediction'] = prediction
    correct = df[(df['decision'] == df['prediction'])]['prediction'].count()
    testing_acc = round(correct/len(df),2)
    return(training_acc,testing_acc)

def discretize_and_split(bins):
    df2 = pd.read_csv("dating.csv")
    discrete_valued_columns = ['gender', 'race', 'race_o', 'samerace', 'field', 'decision']
    discrete_valued_col_ind = []
    for i in discrete_valued_columns:
        discrete_valued_col_ind.append(df2.columns.get_loc(i))
    columns_ind = list(np.arange(0,52))
    continuous_valued_columns_ind = [x for x in columns_ind if x not in discrete_valued_col_ind]
    for i in continuous_valued_columns_ind:
        binned = pd.cut(df2.iloc[:, i],bins=bins,labels=list(range(0,bins)))
        df2.iloc[:, i] = binned
        num_in_bins = []
        for j in range(0,bins):
            num_in_bins.append(len(df2[df2.iloc[:, i] == j]))
        #print(list(df2.columns.values)[i],": ", num_in_bins)
    testSet = df2.sample(frac=0.2, random_state=47)
    trainingSet = df2.drop(testSet.index)
    testSet.to_csv('testSet.csv', index=False)
    trainingSet.to_csv('trainingSet.csv', index=False)

#Run the discretize_and_split and nbc functions as per the bins suggested
b = [2,5,10,50,100,200]
training_accuracies = []
testing_accuracies = []
for i in b:
    discretize_and_split(i)
    training_acc, testing_acc = nbc()
    training_accuracies.append(training_acc)
    testing_accuracies.append(testing_acc)

#Plotting the Accuracies: 

#plot['training_acc'] = training_accuracies
#plot['testing_acc'] = testing_accuracies
#plot.head()
#=df.cumsum()
#plot.plot(title = "y-axis:Accuracy vs x-axis:b")



output_file("5_3.html")

p = figure(plot_width=400, plot_height=400, title="Blue: Training, Red: Test", x_axis_label='Bins', y_axis_label='Accuracy')

# add a line renderer
p.line(x=b, y=training_accuracies, line_width=2, line_color = 'blue')
p.line(x=b, y=testing_accuracies, line_width=2, line_color = 'red')

show(p)