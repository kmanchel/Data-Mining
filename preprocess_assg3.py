import pandas as pd
import numpy as np
df = pd.read_csv("dating-full.csv")
df = df[:6500]
df.to_csv('dating-full.csv', index=False)

def initial_cleanup_3():
    df = pd.read_csv("dating-full.csv")
    df = df[:6500]
    length = len(df)
    processed = []
    c = 0
    for i in range(0,(length)):
        race = df['race'][i]
        l1 = len(race)
        race = race.strip("'")
        l2 = len(race)
        if l1>l2:
            c += 1
        processed.append(race)
    df['race'] = processed
    processed = []
    for i in range(0,(length)):
        race_o = df['race_o'][i]
        l1 = len(race_o)
        race_o = race_o.strip("'")
        l2 = len(race_o)
        if l1>l2:
            c += 1
        processed.append(race_o)
    df['race_o'] = processed
    processed = []
    for i in range(0,(length)):
        field = df['field'][i]
        l1 = len(field)
        field = field.strip("'")
        l2 = len(field)
        if l1>l2:
            c += 1
        processed.append(field)
    df['field'] = processed
    processed = []
    processed = []
    c = 0
    total = 0
    for i in df['field']:
        if i.islower() == False:
            c += 1
        i = i.lower()
        total += 1
        i
        processed.append(i)
    df['field'] = processed

    df2 = df.copy()
    def std_importance():
        df2['pref_o_total'] = df2['pref_o_attractive']+df2['pref_o_sincere']+df2['pref_o_intelligence']+df2['pref_o_funny']
        +df2['pref_o_ambitious']+df2['pref_o_shared_interests']
        df2['pref_o_attractive'] = df2['pref_o_attractive']/df2['pref_o_total']
        df2['pref_o_sincere'] = df2['pref_o_sincere']/df2['pref_o_total']
        df2['pref_o_intelligence'] = df2['pref_o_intelligence']/df2['pref_o_total']
        df2['pref_o_funny'] = df2['pref_o_funny']/df2['pref_o_total']
        df2['pref_o_ambitious'] = df2['pref_o_ambitious']/df2['pref_o_total']
        df2['pref_o_shared_interests'] = df2['pref_o_shared_interests']/df2['pref_o_total']

        df2['important_total'] = df2['attractive_important']+df2['sincere_important']+df2['intelligence_important']+df2['funny_important']
        +df2['ambition_important']+df2['shared_interests_important']
        df2['attractive_important'] = df2['attractive_important']/df2['important_total']
        df2['sincere_important'] = df2['sincere_important']/df2['important_total']
        df2['intelligence_important'] = df2['intelligence_important']/df2['important_total']
        df2['funny_important'] = df2['funny_important']/df2['important_total']
        df2['ambition_important'] = df2['ambition_important']/df2['important_total']
        df2['shared_interests_important'] = df2['shared_interests_important']/df2['important_total']
    std_importance()
    df.iloc[:,9:21] = df2.iloc[:,9:21]
    
    testSet = df.sample(frac=0.2, random_state=47)
    trainingSet = df.drop(testSet.index)
    testSet.to_csv('testSet.csv', index=False)
    trainingSet.to_csv('trainingSet.csv', index=False)
    return()


#Run this shit before running the one hot encode function
def initial_cleanup(df):
    df = df[:6500]
    length = len(df)
    processed = []
    c = 0
    for i in range(0,(length)):
        race = df['race'][i]
        l1 = len(race)
        race = race.strip("'")
        l2 = len(race)
        if l1>l2:
            c += 1
        processed.append(race)
    df['race'] = processed
    processed = []
    for i in range(0,(length)):
        race_o = df['race_o'][i]
        l1 = len(race_o)
        race_o = race_o.strip("'")
        l2 = len(race_o)
        if l1>l2:
            c += 1
        processed.append(race_o)
    df['race_o'] = processed
    processed = []
    for i in range(0,(length)):
        field = df['field'][i]
        l1 = len(field)
        field = field.strip("'")
        l2 = len(field)
        if l1>l2:
            c += 1
        processed.append(field)
    df['field'] = processed
    processed = []
    processed = []
    c = 0
    total = 0
    for i in df['field']:
        if i.islower() == False:
            c += 1
        i = i.lower()
        total += 1
        i
        processed.append(i)
    df['field'] = processed

    df2 = df.copy()
    def std_importance():
        df2['pref_o_total'] = df2['pref_o_attractive']+df2['pref_o_sincere']+df2['pref_o_intelligence']+df2['pref_o_funny']
        +df2['pref_o_ambitious']+df2['pref_o_shared_interests']
        df2['pref_o_attractive'] = df2['pref_o_attractive']/df2['pref_o_total']
        df2['pref_o_sincere'] = df2['pref_o_sincere']/df2['pref_o_total']
        df2['pref_o_intelligence'] = df2['pref_o_intelligence']/df2['pref_o_total']
        df2['pref_o_funny'] = df2['pref_o_funny']/df2['pref_o_total']
        df2['pref_o_ambitious'] = df2['pref_o_ambitious']/df2['pref_o_total']
        df2['pref_o_shared_interests'] = df2['pref_o_shared_interests']/df2['pref_o_total']

        df2['important_total'] = df2['attractive_important']+df2['sincere_important']+df2['intelligence_important']+df2['funny_important']
        +df2['ambition_important']+df2['shared_interests_important']
        df2['attractive_important'] = df2['attractive_important']/df2['important_total']
        df2['sincere_important'] = df2['sincere_important']/df2['important_total']
        df2['intelligence_important'] = df2['intelligence_important']/df2['important_total']
        df2['funny_important'] = df2['funny_important']/df2['important_total']
        df2['ambition_important'] = df2['ambition_important']/df2['important_total']
        df2['shared_interests_important'] = df2['shared_interests_important']/df2['important_total']
    std_importance()
    df.iloc[:,9:21] = df2.iloc[:,9:21]
    return(df)

def preprocess_NBC(df):
    df2 = df.copy()
    discrete_valued_columns = ['gender', 'race', 'race_o', 'samerace', 'field', 'decision']
    discrete_valued_col_ind = []
    for i in discrete_valued_columns:
        discrete_valued_col_ind.append(df.columns.get_loc(i))
    columns_ind = list(np.arange(0,52))
    continuous_valued_columns_ind = [x for x in columns_ind if x not in discrete_valued_col_ind]
    for i in continuous_valued_columns_ind:
        binned = pd.cut(df.iloc[:, i],bins=5,labels=[0,1,2,3,4])
        df2.iloc[:, i] = binned
        num_in_bins = [len(df2[df2.iloc[:, i] == 0]), len(df2[df2.iloc[:, i] == 1]), 
                       len(df2[df2.iloc[:, i] == 2]), len(df2[df2.iloc[:, i] == 3]), len(df2[df2.iloc[:, i] == 4])]
    return(df2)

#Below function has a bug. An updated version created underneath
def preprocess_LR_SVM(df):
    gender_one_hot = pd.get_dummies(df['gender'], drop_first = True, prefix="gender_")
    #print(gender_one_hot.shape)
    race_one_hot = pd.get_dummies(df['race'], drop_first = True, prefix="race_")
    #print(race_one_hot.shape)
    race_o_one_hot = pd.get_dummies(df['race_o'], drop_first = True, prefix="race_o_")
    #print(race_o_one_hot.shape)
    field_one_hot = pd.get_dummies(df['field'], drop_first = True, prefix="field_")
    #print(field_one_hot.shape)
    df.drop(columns=['race','race_o','gender','field'], inplace = True)
    df = df.join(gender_one_hot).join(race_one_hot).join(race_o_one_hot).join(field_one_hot)
    return(df)

#Updated preprocessing function for LR and SVM
def preprocess_LR_SVM_0(df):
    gender_one_hot = pd.get_dummies(df['gender'], drop_first = True, prefix="gender_")
    #print(gender_one_hot.shape)
    race_one_hot = pd.get_dummies(df['race'], drop_first = True, prefix="race_")
    #print(race_one_hot.shape)
    race_o_one_hot = pd.get_dummies(df['race_o'], drop_first = True, prefix="race_o_")
    #print(race_o_one_hot.shape)
    field_one_hot = pd.get_dummies(df['field'], drop_first = True, prefix="field_")
    #print(field_one_hot.shape)
    df.drop(columns=['race','race_o','gender','field'], inplace = True)
    df = df.join(gender_one_hot).join(race_one_hot).join(race_o_one_hot).join(field_one_hot)
    df.insert(0, 'intercept', 1)
    return(df)

def train_test_split(df):
    testSet = df.sample(frac=0.2, random_state=47)
    trainingSet = df.drop(testSet.index)
    testSet.to_csv('testSet.csv', index=False)
    trainingSet.to_csv('trainingSet.csv', index=False)
    
def sort(field):
    field = field.astype('category')
    field_unique = list(field.unique())
    field_sorted = sorted(field_unique)
    return(field_sorted)

def one_hot_encoding(df):
    #One Hot Encoding for Gender
    gender_one_hot_encoding = {'female':[1],'male':[0]}


    #One Hot Encoding for Race
    race_sorted = sort(df['race'])
    N = len(race_sorted)
    enc = list(np.zeros(N-1,dtype=int))
    race_one_hot = []
    for i in range(0,N-1):
        enc[i]=1
        race_one_hot.append(enc)
        enc = list(np.zeros(N-1,dtype=int))
    race_one_hot.append(list(np.zeros(N-1,dtype=int)))
    race_one_hot_encoding = dict(zip(race_sorted, race_one_hot))


    #One Hot Encoding for race_o
    race_o_sorted = sort(df['race_o'])
    N = len(race_o_sorted)
    enc = list(np.zeros(N-1,dtype=int))
    race_o_one_hot = []
    for i in range(0,N-1):
        enc[i]=1
        race_o_one_hot.append(enc)
        enc = list(np.zeros(N-1,dtype=int))
    race_o_one_hot.append(list(np.zeros(N-1,dtype=int)))
    race_o_one_hot_encoding = dict(zip(race_o_sorted, race_o_one_hot))

    #One Hot Encoding for Field
    field_sorted = sort(df['field'])
    N = len(field_sorted)
    enc = list(np.zeros(N-1,dtype=int))
    field_one_hot = []
    for i in range(0,N-1):
        enc[i]=1
        field_one_hot.append(enc)
        enc = list(np.zeros(N-1,dtype=int))
    field_one_hot.append(list(np.zeros(N-1,dtype=int)))
    field_one_hot_encoding = dict(zip(field_sorted, field_one_hot))
    print("Mapped vector for female in column gender: ",gender_one_hot_encoding['female'])
    print("Mapped vector for Black/African American in column race: ",race_one_hot_encoding['Black/African American'])
    print("Mapped vector for Other in column race_o: ",race_o_one_hot_encoding['Other'])
    print("Mapped vector for economics in column field: ",field_one_hot_encoding['economics'])

one_hot_encoding(initial_cleanup(df))
train_test_split(preprocess_LR_SVM_0(df))
