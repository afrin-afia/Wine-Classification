import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sklearn as sk
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns

def read_and_org():
    df = pd.read_csv("winequalityN.csv")

    df.fillna(value=df.mean(), inplace= True)

    #extract features & labels from dataframe
    y_graph= df['quality'] #target. This y is a series object, not array
    y_graph= y_graph.values   #covert to array from series object

    #uncommenting next 2 lines will draw the histogram
    #plt.hist(y_graph, bins=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    #plt.show()

    #replace categorical values that are represented by strings in the dataframe [red/white]
    df.replace({'type': 'white'}, 0, inplace=True)
    df.replace({'type': 'red'}, 1, inplace=True)

    #Replace y values by good (0), average (1) and below average(2)-- 3 classes
    df['quality'].values[df['quality'] <= 5] = 2
    df['quality'].values[df['quality'] == 6] = 1
    df['quality'].values[df['quality'] > 6] = 0
    #6: average, 0-5: below and >=7: good... this division distributes the dataset eqally among three classes(histogram). So taking this one!

    #next 4 lines draw correlation matrix
    #corr = df.corr()
    #plt.subplots(figsize=(15,10))
    #sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True)
    #plt.show()

    #extract features & labels from dataframe
    y= df['quality'] #target. This y is a series object, not array
    y= y.values   #covert to array from series object
    x= df.drop(['quality'], axis=1)   #input features, this is dataframe. We need to convert this to array.
    x= x.values

    #normalize features
    x= sk.preprocessing.normalize(x, norm='max', axis= 1)

    #uncommenting next 2 lines will draw the histogram
    #plt.hist(y, bins='auto')
    #plt.show()

    return x,y


def GNB():
    print("-----GNB-----")
    #only take the cont. features (omit type)
    gx_train= x_train[:, [1, 11]]
    gx_val= x_val[:, [1, 11]]
    gx_test= x_test[:, [1, 11]]
    params_NB = np.logspace(0,-9, num=50)
    max_val_accu= 0
    best_param= 0
    for v in params_NB:
        #print (v)
        gnb = GaussianNB(var_smoothing=v)
        gaussian = gnb.fit(gx_train, y_train)   
        val_accu= gaussian.score(gx_val,y_val)
        if(val_accu > max_val_accu):
            max_val_accu= val_accu
            best_param= v
    #print (best_param)
    print ("Highest Validation Accuracy for parameter "+ str(best_param)+" is: "+str(max_val_accu))

    #test with best param
        #first train again with best param
    gnb = GaussianNB(var_smoothing= best_param)
    gaussian = gnb.fit(gx_train, y_train)
        #now test
    test_accu= gaussian.score(gx_test,y_test)
    print("Test accuracy:", test_accu)
    return test_accu

def KNN():
    print("-----KNN-----")
    n= [5,10,15,20,25,30,35,40,45,50,55,60]
    max_val_accu= 0
    best_param= 0
    #find best weights
    best_weights= 'distance'
    neigh = KNeighborsClassifier(weights='distance')
    neigh.fit(x_train, y_train)
    val_accu_dist= neigh.score(x_val,y_val)

    neigh = KNeighborsClassifier(weights='uniform')
    neigh.fit(x_train, y_train)
    val_accu_uni= neigh.score(x_val,y_val)

    if(val_accu_uni > val_accu_dist):
        best_weights= 'uniform'

    for i in n:
        neigh = KNeighborsClassifier(n_neighbors=i, weights=best_weights)
        neigh.fit(x_train, y_train)
        val_accu= neigh.score(x_val,y_val)
        if(val_accu > max_val_accu):
            max_val_accu= val_accu
            best_param= i
        #print(neigh.score(x_val, y_val))
    print("Highest Validation Accuracy for #neighbors= "+str(best_param)+ " and weights= "+best_weights+ " is: "+str(max_val_accu))

    #test
        #first train again
    neigh = KNeighborsClassifier(n_neighbors=best_param, weights='distance')
    neigh.fit(x_train, y_train)
        #now test
    test_accu= neigh.score(x_test,y_test)
    print("Test accuracy:", test_accu)
    return test_accu

def find_best_depth():
    max_accu=0
    best_depth= 0
    for i in range (10,500,50):
        
        dtree = DecisionTreeClassifier(max_depth= i)
        dtree.fit(x_train , y_train)
        score= dtree.score(x_val, y_val)
        if(score> max_accu):
            max_accu=score
            best_depth= i
        
    print("Highest Validation Accuracy for parameter (max depth) "+str(best_depth)+" is: "+str(max_accu))
    return max_accu,best_depth
    

def find_best_depth2(c,l,s, max_accu, best_depth):
    mindepth= int(best_depth/2)
    maxdepth= int(best_depth*2)
    for i in range (mindepth,maxdepth,10):
        
        dtree = DecisionTreeClassifier(criterion=c, min_samples_leaf= l, min_samples_split= s, max_depth= i)
        dtree.fit(x_train , y_train)
        score= dtree.score(x_val, y_val)
        if(score> max_accu):
            max_accu=score
            best_depth= i
        
    return max_accu,best_depth


def DTree():

    #find best max_depth, considering all other hyperparameters as default ones
    max_accu, best_depth= find_best_depth()

    #now keep best max_depth as it is, tune other hyperparameters
    dtree = DecisionTreeClassifier(max_depth=best_depth)
    param_dict= {
        "criterion": ['gini', 'entropy'],
        "min_samples_split": range (3,10),
        "min_samples_leaf": range(1,5)
    }
    grid= GridSearchCV(dtree, param_grid= param_dict, cv=10)
    grid.fit(x_train, y_train)
    #grid.best_params_
    c= grid.best_params_["criterion"]
    l= grid.best_params_["min_samples_leaf"]
    s= grid.best_params_["min_samples_split"]
    #now find new best depth with tuned parameters
    max_accu,best_depth= find_best_depth2(c,l,s, max_accu, best_depth)
    print("Validation accuracy after two round of hyperparameter tuning: ", max_accu)
    #test
    #use best param
    dtree = DecisionTreeClassifier(max_depth=best_depth, criterion=c, min_samples_leaf= l, min_samples_split= s)
    dtree.fit(x_train, y_train)
    test_accu= dtree.score(x_test,y_test)
    print("Test accuracy:", test_accu)

    return best_depth, c, l, s, test_accu  #we need these hyperparams. for random forest

def RF(d,c,l,s):

    print("-----Random Forest-----")
    max_accu=0
    for i in range (100,1000,50):
        rnd = RandomForestClassifier(n_estimators=i,random_state = 0, max_depth=d,min_samples_split=s, min_samples_leaf=l, criterion=c)
        # fit data
        rnd.fit(x_train,y_train)
        # predicting score
        accu = rnd.score(x_val,y_val)
        if(accu>max_accu):
            max_accu= accu
            best_n_e= i
        
    #test
        #train again
    print('Random forest validation accuracy: ', max_accu)
    rnd = RandomForestClassifier(n_estimators=best_n_e,random_state = 0, max_depth=d,min_samples_split=s, min_samples_leaf=l, criterion=c)
    rnd.fit(x_train,y_train)
    test_accu = rnd.score(x_test,y_test)
    print('Random Forest Test Accuracy : ',test_accu)
    return test_accu

def RF_():
    print("-----Random Forest With Larger Training Dataset-----")
    rnd = RandomForestClassifier(n_estimators=800,random_state = 0)
    rnd.fit(x_tv,y_tv)
    test_accu = rnd.score(x_test,y_test)
    print('Random Forest Test Accuracy : ',test_accu)

#Read file, reorganize the dataset to generate x and y 
x,y= read_and_org()


#split x and y
x_tv, x_test, y_tv, y_test = train_test_split(x,y,test_size=0.2,train_size=0.8)
x_train, x_val, y_train, y_val = train_test_split(x_tv,y_tv,test_size = 0.25,train_size =0.75)

#MODEL 0: Baseline (Majorit Guess)
#baseline
class0= (y_tv == 0).sum()
class1= (y_tv == 1).sum()
class2= (y_tv == 2).sum()
m= max(class0, class1, class2)
a= m/(class0+class1+class2)
print("Baseline (most frequent classs is predicted) accuracy: ", a)

#MODEL 1: Gaussian Naive-Bayes
gnb_accu= GNB()*100

#MODEL 2: K-NN
knn_accu= KNN()*100

#MODEL 3: Dtree
d,c,l,s, dtree_accu= DTree()
dtree_accu= dtree_accu*100

#MODEL 4.1: RF with hyperparameter tuning
rf_accu= RF(d,c,l,s)*100

#MODEL 4.2: RF without tuning 
#RF_()

#draw comparison chart 
v= [a*100, gnb_accu,knn_accu,dtree_accu, rf_accu]
models=['Baseline', 'Gausian NB', 'K-NN', 'DecisionTree', 'RandomForest']
plt.bar(models, v)
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.show()
