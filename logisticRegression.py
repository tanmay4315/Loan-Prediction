import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
path="loandata.txt"
data=pd.read_csv(path)
#data preprocessing
data['Gender'].fillna('Male',inplace=True)
data['Married'].fillna('Yes',inplace=True)
data['Dependents'].fillna('0',inplace=True)
data['Self_Employed'].fillna('No',inplace=True)
data['LoanAmount'].fillna(round(data['LoanAmount'].mean(),1),inplace=True)
data['Loan_Amount_Term'].fillna(round(data['Loan_Amount_Term'].mean(),1),inplace=True)
data['Credit_History'].fillna(1.0,inplace=True)
data.pop('Loan_ID')
nrows=len(data)
col=list(data.columns)
ncols=len(col)
t_depen=list(set(data.loc[:]['Dependents']))
t_area=list(set(data.loc[:]['Property_Area']))
fdata=[]
for i in range(nrows):
    colu=[]
    for j in col:
        if j=='Property_Area':
            if data.loc[i,j]=='Rural':
                colu.append(1.0)
                colu.append(0.0)
                colu.append(0.0)
            elif data.loc[i,j]=='Urban':
                colu.append(0.0)
                colu.append(1.0)
                colu.append(0.0)
            else:
                colu.append(0.0)
                colu.append(0.0)
                colu.append(1.0)
        elif j=='Dependents':
            if data.loc[i,j]=='0':
                colu.append(1.0)
                colu.append(0.0)
                colu.append(0.0)
                colu.append(0.0)
            elif data.loc[i,j]=='1':
                colu.append(0.0)
                colu.append(1.0)
                colu.append(0.0)
                colu.append(0.0)
            elif data.loc[i,j]=='2':
                colu.append(0.0)
                colu.append(0.0)
                colu.append(1.0)
                colu.append(0.0)
            elif data.loc[i,j]=='3+':
                colu.append(0.0)
                colu.append(0.0)
                colu.append(0.0)
                colu.append(1.0)
        elif data.loc[i,j]=='No':
            colu.append(0.0)
        elif data.loc[i,j]=='Yes':
            colu.append(1.0)
        elif data.loc[i,j]=='Male':
            colu.append(1.0)
        elif data.loc[i,j]=='Female':
            colu.append(0.0)
        elif data.loc[i,j]=='Y':
            colu.append(1.0)
        elif data.loc[i,j]=='N':
            colu.append(0.0)
        elif data.loc[i,j]=='Graduate':
            colu.append(1.0)
        elif data.loc[i,j]=='Not Graduate':
            colu.append(0.0)
        else:
            colu.append(data.loc[i,j])
    fdata.append(colu)
nrows=len(fdata)
ncols=len(fdata[1])
train_x=[]
train_y=[]
test_x=[]
test_y=[]
for i in range(nrows):
    if i%10==0:
        test_x.append(fdata[i][0:ncols-1])
        test_y.append(fdata[i][-1])
    else:
        train_x.append(fdata[i][0:ncols-1])
        train_y.append(fdata[i][-1])
beta=[0]*(ncols+1)
#Training model
for i in range(1):
    beta[-1]=beta[-1]+0.0000001*sum([(train_y[j]-(1/(1+math.exp(-(sum([train_x[j][l]*beta[l] for l in range(len(train_x[1]))]))))))*1 for j in range(len(train_x))])
    for k in range(len(train_x[1])): #here k used for updating value of beta
        beta[k]=beta[k]+0.00000001*sum([(train_y[j]-(1/(1+math.exp(-(sum([train_x[j][l]*beta[l] for l in range(len(train_x[1]))]))))))*train_x[j][k] +beta[-1] for j in range(len(train_x))])

#testing Model
correct=0
for i in range(len(test_x)):
    val=1/(1+math.exp(-(sum([test_x[i][j]*beta[j] for j in range(len(test_x[1]))]))))
    if round(val)== test_y[i]:
        correct+=1
#accuracy
print(correct*100/len(test_x))

