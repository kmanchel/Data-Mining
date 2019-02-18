import pandas as pd

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
    print("Training Accuracy: ", round(correct/len(df),2))
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
    print("Test Accuracy: ", round(correct/len(df),2))
    
    return(True)

nbc()