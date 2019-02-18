import pandas as pd
import numpy as np
df = pd.read_csv("dating.csv")
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
    print(list(df2.columns.values)[i],": ", num_in_bins)
df2.to_csv('dating_binned.csv', index=False)