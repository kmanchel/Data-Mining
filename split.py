import pandas as pd
df = pd.read_csv("dating_binned.csv")
testSet = df.sample(frac=0.2, random_state=47)
trainingSet = df.drop(testSet.index)
testSet.to_csv('testSet.csv', index=False)
trainingSet.to_csv('trainingSet.csv', index=False)