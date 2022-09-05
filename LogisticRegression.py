import pandas as pd

data=pd.read_csv('Iris.csv')
data=data[data['Species'] != 'Iris-virginica']
data=data.drop('Id',axis=1)
data['Species'].mask(data['Species']=='Iris-setosa',1,inplace=True)
data['Species'].mask(data['Species']=='Iris-versicolor',0,inplace=True)

from sklearn.model_selection import train_test_split

data_train,data_test=train_test_split(data,test_size=0.25)
Y_train=pd.DataFrame(data_train['Species'])
Y_test=pd.DataFrame(data_test['Species'])
X_train=data_train.drop('Species',axis=1)
X_test=data_test.drop('Species',axis=1)


import numpy as np
import math

def get_h_theta(thetas,x):#thetas es una lista con todos los valores iniciales de theta
    h_theta=[]
    for i in range(len(x)):
        dummy1=[]
        for j in range(1,len(thetas)):
            dummy1.append(thetas[j]*x.iloc[i,j-1])
        dummy2=1/(1+math.exp(-(thetas[0]+sum(dummy1))))
        h_theta.append(dummy2)
    return h_theta

def get_j_theta(h_theta,y):
    y_ln_=[]
    for i in range(len(y)):
        if y.iloc[i,:].item()==1:
        #if y[i]==1:
            dummy=math.log(h_theta[i])
            y_ln_.append(dummy)
        elif y.iloc[i,:].item()==0:
        #elif y[i]==0:
            dummy=math.log(1-h_theta[i])
            y_ln_.append(dummy)
    j_theta=(-1/len(y))*sum(y_ln_)
    return j_theta

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

def define_class(new_thetas,x_pred):
    class_h_theta=[]
    for i in range(len(x_pred)):
        dummy1=[]
        for j in range(1,len(new_thetas)):
            dummy1.append(new_thetas[j]*x_pred.iloc[i,j-1])
        dummy2=round(1/(1+math.exp(-(new_thetas[0]+sum(dummy1)))))
        class_h_theta.append(dummy2)
    return class_h_theta

def make_predictions(thetas,x,y,alpha,x_pred):
    h_theta=get_h_theta(thetas,x)
    j_theta=get_j_theta(h_theta,y)
    new_thetas=get_new_thetas(thetas,alpha,h_theta,y,x)
    class_h_theta=define_class(new_thetas,x_pred)
    return class_h_theta,new_thetas


thetas=[10,0.5,0.25,0.3,0.1]
alpha=0.5
x=X_train
y=Y_train
x_pred=X_test

class_h_theta,new_thetas=make_predictions(thetas,x,y,alpha,x_pred)
print(class_h_theta)
print(Y_test['Species'].tolist())