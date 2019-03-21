#There are 3 functions below. One is the LR, another is SVM and the final is the lr_svm which.
#Arguments for each of the functions are taken from the raw_input entered on commandline
#Inputs required: Training data filename, test data filename, specify learning algorithm to be used ("1" for Logistic Regression, "2" for SVM)
#Sample Command line input: $python lr svm.py trainingSet.csv testSet.csv 1

#The code will run the lr_svm function 

import sys
import numpy as np
import pandas as pd

#Logistic Regression Function
def lr_0(trainingSet, testSet):
    #Preprocessing Training Set
    df_train = trainingSet
    y_train = df_train['decision']
    x_train = df_train.copy()
    x_train.drop(columns=['decision'], inplace = True)
    
    #Preprocessing Testing Set
    df_test = testSet
    y_test = df_test['decision']
    x_test = df_test.copy()
    x_test.drop(columns=['decision'], inplace = True)
    
    def logistic(z):
        return 1 / (1 + np.exp(-z))
    def scores(weights, x):
        return(-np.dot(x,weights.T))
    def predict(y_i):
        return(logistic(y_i))
    
    def gradient_descent(x,y,lmbda = 0.01,lr=0.01,max_iter=500):
        weights = np.zeros(len(x.columns))
        for i in range(0,max_iter):
            y_i = predict(scores(weights=weights,x=x))
            err = list(y) - y_i
            del_J = np.dot(err,x)+(lmbda*weights)
            weights -= lr*del_J
        return(weights)
    
    w_trained = gradient_descent(x=x_train,y=y_train)
    
    df_train['prediction'] = list((np.round(predict(scores(weights=w_trained,x=x_train)))))
    #print("Successfully Trained")
    #print("x_test Length: ",len(x_test.columns.values), "| weights length: ",len(w_trained))
    #if t_frac!=0.025:
        #print("x_train Columns List: ", x_train.columns.values)
        #print("x_test Columns List: ", x_test.columns.values)
    df_test['prediction'] = list((np.round(predict(scores(weights=w_trained,x=x_test)))))
    #print("Successfully Tested")
    
    correct_train = df_train[(df_train['decision'] == df_train['prediction'])]['prediction'].count()
    training_acc = correct_train/len(df_train)
    #print("Training Accuracy: ", training_acc)

    correct_test = df_test[(df_test['decision'] == df_test['prediction'])]['prediction'].count()
    testing_acc = correct_test/len(df_test)
    #print("Testing Accuracy: ", testing_acc)
    df_test.drop(columns=['prediction'], inplace = True)
    return(training_acc,testing_acc)


def svm(trainingSet, testSet):
    #Preprocessing Training Set
    df_train = trainingSet
    df_train.loc[df_train['decision'] < 1, 'decision'] = -1
    y_train = df_train['decision']
    x_train = df_train.copy()
    x_train.drop(columns=['decision'], inplace = True)

    #Preprocessing Testing Set
    df_test = testSet
    df_test.loc[df_test['decision'] < 1, 'decision'] = -1
    y_test = df_test['decision']
    x_test = df_test.copy()
    x_test.drop(columns=['decision'], inplace = True)
    
    def yi_hat(weights,x):
        return(np.dot(x,weights.T))
    
    Iter = 500
    def subgradient_descent(x,y,lmbda=0.01,lr=0.5,max_iter=500):
        weights = np.zeros(x.shape[1])
        N = y.shape[0]
        row_index = list(range(N))
        for i in range(0,max_iter):
            y_product = np.multiply(y,yi_hat(weights=weights,x=x))
            #Fix indexing issue below and you will be good
            del_ji_non_int = [np.dot(y[k],x[k:k+1].values) if y_product[k] < 1 else 0 for k in row_index]
            #del_j_non_int = (sum((lmbda*weights)-del_ji_non_int))*(-1/(N-1))
            del_j_non_int = ((lmbda*weights)+sum(del_ji_non_int)[0])*(-1/(N-1))
            weights -= lr*del_j_non_int
        return(weights)
    
    import time
    t0 = time.time()
    w = subgradient_descent(x=x_train,y=y_train.values, max_iter=Iter)
    classifications_train = np.sign(np.dot(x_train,w.T))
    classifications_test = np.sign(np.dot(x_test,w.T))
    t1 = time.time()
    print("SVM Run Time: ",t1-t0,"\nMax Iterations: ",Iter)
    
    df_train['prediction'] = list(classifications_train)
    correct_train = df_train[(df_train['decision'] == df_train['prediction'])]['prediction'].count()
    training_acc = correct_train/len(df_train)
    

    df_test['prediction'] = list(classifications_test)
    correct_test = df_test[(df_test['decision'] == df_test['prediction'])]['prediction'].count()
    testing_acc = correct_test/len(df_test)
    
    df_test.drop(columns=['prediction'], inplace = True)
    return(training_acc,testing_acc)

def lr_svm(trainingSetFilename, testSetFilename, modelIdx):
    trainingSet = pd.read_csv(trainingSetFilename)
    testSet = pd.read_csv(testSetFilename)
    if modelIdx==1:
        print("Model Running: Logistic Regression")
        training_acc, testing_acc = lr_0(trainingSet=trainingSet,testSet=testSet)
    elif modelIdx==2:
        print("Model Running: SVM")
        training_acc, testing_acc = svm(trainingSet=trainingSet,testSet=testSet)
    print("Training Accuracy: ", round(training_acc,2))
    print("Testing Accuracy: ", round(testing_acc,2))

trainingSetFilename=str(sys.argv[1])
testSetFilename=str(sys.argv[2])    
modelIdx=int(sys.argv[3])
print("Training file: ", trainingSetFilename)
print("Test file: ", testSetFilename)
print("Model Index: ", modelIdx)
lr_svm(trainingSetFilename=trainingSetFilename, testSetFilename=testSetFilename, modelIdx=modelIdx)





