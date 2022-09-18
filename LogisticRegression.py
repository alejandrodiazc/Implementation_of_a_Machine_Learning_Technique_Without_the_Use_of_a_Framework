###########################################################################################
# Libraries

import pandas as pd
import numpy as np
import math
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


# this function applies the sigmoid function as the hipothesis function
def get_h_theta(thetas,x):#thetas is a list formed by all the initial values of theta
    h_theta=[]
    for i in range(len(x)):
        dummy1=[]
        for j in range(1,len(thetas)):
            dummy1.append(thetas[j]*x.iloc[i,j-1])
        dummy2=1/(1+math.exp(-(thetas[0]+sum(dummy1))))
        h_theta.append(dummy2)
    return h_theta

# this function gets the cost function (log-loss)
def get_j_theta(h_theta,y):
    y_ln_=[]
    for i in range(len(y)):
        if y.iloc[i,:].item()==1:
            dummy=math.log(h_theta[i])
            y_ln_.append(dummy)
        elif y.iloc[i,:].item()==0:
        #elif y[i]==0:
            # a very small number is added, in order to avoid 1-h_theta[i] equals zero (natural logarithm of zero is not possible)
            avoid_zero=0.00000000000000000000000000000000001
            dummy=math.log(1-h_theta[i]+avoid_zero)
            y_ln_.append(dummy)
    j_theta=(-1/len(y))*sum(y_ln_)
    return j_theta

# this function calculates the new value of each theta, through the derivatives of the cost function
def get_new_thetas(thetas,alpha,h_theta,y,x):
    new_thetas=[]
    h_theta_y=[]
    for i in range(len(h_theta)):
        dummy=h_theta[i]-y.iloc[i,:].item()
        h_theta_y.append(dummy)
    new_theta_0=thetas[0]-alpha*(1/len(h_theta_y))*sum(h_theta_y)
    new_thetas.append(new_theta_0)
    
    for j in range(1,len(thetas)):
        h_theta_y_x=[]
        for i in range(len(h_theta)):
            dummy=(h_theta[i]-y.iloc[i,:].item())*x.iloc[i,j-1]
            h_theta_y_x.append(dummy)
        new_theta=thetas[j]-alpha*(1/len(h_theta_y_x))*sum(h_theta_y_x)
        new_thetas.append(new_theta)  
    return new_thetas

# this function gets the class (or y value) for the x entries, with the newer thetas.
def define_class(new_thetas,x_pred):
    class_h_theta=[]
    for i in range(len(x_pred)):
        dummy1=[]
        for j in range(1,len(new_thetas)):
            dummy1.append(new_thetas[j]*x_pred.iloc[i,j-1])
        dummy2=round(1/(1+math.exp(-(new_thetas[0]+sum(dummy1)))))
        class_h_theta.append(dummy2)
    return class_h_theta

# this function joins all of the previous functions and retrieves the predictions, once all of the iterations for training are performed
def make_predictions(thetas,x,y,alpha,x_pred,num_iters):
    for i in range(num_iters):
        if i!=0:
            thetas=new_thetas
        h_theta=get_h_theta(thetas,x)
        j_theta=get_j_theta(h_theta,y)
        new_thetas=get_new_thetas(thetas,alpha,h_theta,y,x)
        class_h_theta=define_class(new_thetas,x_pred)
        y_pred=class_h_theta
    return y_pred#,new_thetas

# Display all of the metrics for evaluating the predictions
def get_metrics(y,y_pred):
    cm=confusion_matrix(y_true=y,y_pred=y_pred)
    acc=accuracy_score(y_true=y,y_pred=y_pred)
    prec=precision_score(y_true=y,y_pred=y_pred)
    rec=recall_score(y_true=y,y_pred=y_pred)
    f1=f1_score(y_true=y,y_pred=y_pred)
    return cm,acc,prec,rec,f1

###########################################################################################

data=pd.read_csv('Iris.csv')
data=data[data['Species'] != 'Iris-virginica']
data=data.drop('Id',axis=1)
data['Species'].mask(data['Species']=='Iris-setosa',int(1),inplace=True)
data['Species'].mask(data['Species']=='Iris-versicolor',int(0),inplace=True)


data_train,data_test=train_test_split(data,test_size=0.25)
y_train=pd.DataFrame(data_train['Species'])
y_test=pd.DataFrame(data_test['Species']).iloc[:,0].tolist()
X_train=data_train.drop('Species',axis=1)
X_test=data_test.drop('Species',axis=1)
thetas=[10,0.5,0.25,0.3,0.1]
alpha=0.5

y_pred=make_predictions(thetas,X_train,y_train,alpha,X_test,1)
cm,acc,prec,rec,f1=get_metrics(y_test,y_pred)

print("Predictions from the test set:",y_pred)
print("True positives of test vs. predictions:",cm[0][0])
print("False negatives of test vs. predictions:",cm[0][1])
print("Flase positives of test vs. predictions:",cm[1][0])
print("True negatives of test vs. predictions:",cm[1][1])
print("Accuracy of predictions:",acc)
print("Precision of predictions:",prec)
print("Recall of predictions:",rec)
print("F1-score of predictions:",f1)