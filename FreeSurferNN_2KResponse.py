# -*- coding: utf-8 -*-
"""
@author: Cedric Beaulac
File created on 28/08/21
Postdoc Project #1 
Feature Extraction Technique #1
Using FreeSurfer output as NN classifier input
"""


####################################
# Import packages
####################################

from __future__ import print_function #all part of pytorch
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import itertools

import numpy as np 
import pandas as pd
import seaborn as sns #package
import matplotlib.pyplot as plt
import nibabel as nbl
from nipype.interfaces.freesurfer import AddXFormToHeader
import time
import cv2
from sklearn.linear_model import LogisticRegression #package
from sklearn import metrics

from ignite.engine import Engine, Events, create_supervised_trainer, create_supervised_evaluator
from ignite.handlers import EarlyStopping
from ignite.metrics import Precision,Accuracy

###################################
# Importing the clinical data for diagnostic and the predictor data
# NN trained on ADNI1 for genetic purposes (sub-cohort that only contains DAT and NC)
####################################

# These predictors are ALL of the FreeFreesurfer extracted statistics
Predictors = pd.read_csv(r'Data_02.csv',index_col=0)
#Predictors = pd.read_csv(r'Data_02.csv',index_col=0)

# Small Predictors is the subset of 56 features selected by experts
SPredictors = pd.read_csv(r'ExpertFeatures_01 .csv',index_col=0)
#SPredictors = pd.read_csv(r'ExpertFeatures_01 .csv',index_col=0)

# Clinical contains the response (AD Diagnosis)
Clinical = pd.read_csv(r'ClinicalInfo.csv')

ClinicalID = Clinical[Clinical['RID'].isin(np.array(Predictors.index,dtype=int))]

#Including age as a predictor for both set of predictors
ClinicalID = ClinicalID.set_index('RID')

#Age used to be included to control for it, we are now looking at a new approach to do so
#Age = ClinicalID['AGE']
#Predictors = Predictors.join(Age,how='right')
#SPredictors = SPredictors.join(Age,how='right')

ClinicalF = ClinicalID[ClinicalID['VISCODE']=='bl']


ClinicalF['AUX.DIAGNOSIS'].value_counts()
ClinicalF['AUX.STRATIFICATION'].value_counts()

Response= ClinicalF['AUX.STRATIFICATION']
Response= Response.loc[(Response=='sDAT') | (Response=='sNC') | (Response=='eDAT') | (Response=='uNC')]
Response.loc[(Response=='sDAT') | (Response=='eDAT')] = "DAT"
Response.loc[(Response=='sNC') | (Response=='uNC')] = "NC"
Response.value_counts()

#PandaData is the data containing our data set with all the features
PandaData = Predictors.join(Response,how='right')

PandaSData = SPredictors.join(Response,how='right')

####################################
# Arguments
####################################

parser = argparse.ArgumentParser(description='Freesurfer feature extraction (Cedric Beaulac)')
parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 50)')
parser.add_argument('--xdim', type=int, default=PandaData.shape[1]-1, metavar='N',
                    help='dimension of the predictor')     
parser.add_argument('--n', type=int, default=PandaData.shape[0], metavar='N',
                    help='number of subjects')  
parser.add_argument('--ntr', type=int, default=150, metavar='N',
                    help='number of training subjects')
parser.add_argument('--nval', type=int, default=50, metavar='N',
                    help='number of training subjects')   
parser.add_argument('--nc', type=int, default=2, metavar='N',
                    help='number of class')                
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--nMC', type=int, default=50, metavar='N',
                    help='number of epochs to train (default: 10)')
args = parser.parse_args()

#torch.manual_seed(args.seed)

device = torch.device("cpu") #should be CUDA when running on the big powerfull server

###################################
# Define the NN Classifier
####################################
class NN1(nn.Module):
    def __init__(self,f):
        super(NN1, self).__init__()
        self.fc1 = nn.Linear(args.xdim, 750)
        #self.fc2 = nn.Linear(1000,500)
        self.fc3 = nn.Linear(args.xdim,f)
        self.fc4 = nn.Linear(f, args.nc)
        self.act = F.relu
        self.dropout = nn.Dropout(0.5)
    def forward(self, x):
        #x1 = self.act(self.fc1(x))
        #x2 = self.dropout(x1)
        #x3 = F.relu(self.fc2(x2))
        #x4 = self.dropout(x3)
        features = self.act(self.fc3(x))
        return F.log_softmax(self.fc4(features),dim=1),features
      

####################################
# Set up Data set
####################################

# Numpy/tensor data

npData = np.array(PandaData)
c,v = np.unique(npData[:,-1],return_inverse=True)
npData[:,-1] = v
npData = npData.astype(float)

perm = np.random.permutation(npData.shape[0])
Data = torch.tensor(npData[perm, :])

npSData = np.array(PandaSData)
c,v = np.unique(npSData[:,-1],return_inverse=True)
npSData[:,-1] = v
npSData = npSData.astype(float)

SData = npSData[perm, :]

#Prepare torch data and normalize predictors
TrainData = Data[0:args.ntr]
TrainData[:,0:-1] = torch.nn.functional.normalize(TrainData[:,0:-1])
ValData = Data[args.ntr:(args.ntr+args.nval)]
ValData[:,0:-1] = torch.nn.functional.normalize(ValData[:,0:-1])
TestData = Data[(args.ntr+args.nval):]
TestData[:,0:-1] = torch.nn.functional.normalize(TestData[:,0:-1])

#Divide small data set as well
TrainSData = SData[0:(args.ntr+args.nval)]
TestSData = SData[(args.ntr+args.nval):]

####################################
# Define the training procedure
####################################
#Define data loaders
train_loader = torch.utils.data.DataLoader(
    TrainData, batch_size=100, shuffle=True)
val_loader = torch.utils.data.DataLoader(ValData)
test_loader = torch.utils.data.DataLoader(TestData)

#training function
def train(epoch,optimizer):
    model.train()
    train_loss = 0
    for id,data in enumerate(train_loader):
        data = data.to(device)
        inputs,target = data[:,0:-1],data[:,-1].type(torch.LongTensor)
        #print(target[0:10])
        #target = torch.cat((0*torch.ones(125),torch.ones(125))).type(torch.LongTensor)
        output,f = model(inputs.float())
        #print(output[1:10,:])
        loss = F.nll_loss(output, target)
        loss.backward()
        #print('====> Epoch: {} Average loss: {:.4f}'.format(epoch,loss ))
        optimizer.step()

#accuracy function for assesment
def accuracy(args,model,data):
    inputs,target = data[:,0:-1],data[:,-1].type(torch.LongTensor)
    output,f = model(inputs.float())
    pred = output.argmax(dim=1, keepdim=True) 
    correct = pred.eq(target.view_as(pred)).sum().item()
    acc = correct/data.shape[0]
    return acc

#accuracy function for assesment
def AUC(args,model,data):
    inputs,target = data[:,0:-1],data[:,-1].type(torch.LongTensor)
    output,f = model(inputs.float())
    pred = output.argmax(dim=1, keepdim=True) 
    fpr, tpr, thresholds = metrics.roc_curve(target, pred)
    auc = metrics.auc(fpr, tpr)
    return auc

#return pred for CV
def predictions(args,model,data):
    inputs,target = data[:,0:-1],data[:,-1].type(torch.LongTensor)
    output,f = model(inputs.float())
    pred = output.argmax(dim=1, keepdim=True) 
    return pred

#initialize model parameters
def model_init(args,model,std):
    torch.nn.init.normal_(model.fc1.weight,0,std) 
    torch.nn.init.normal_(model.fc3.weight,0,std) 
    return model



####################################
# Model Training with Cross validated parameters
####################################
perm = np.random.permutation(npData.shape[0])
Data = torch.tensor(npData[perm, :])
TrainData = Data[0:(args.ntr+args.nval)]
TrainData[:,0:-1] = torch.nn.functional.normalize(TrainData[:,0:-1])
TestData = Data[(args.ntr+args.nval):]
TestData[:,0:-1] = torch.nn.functional.normalize(TestData[:,0:-1])
model = NN1(f=56)
model = model_init(args,model,5)
optimizer = optim.Adagrad(model.parameters(),lr=0.01)
for epoch in range(1, 750+ 1):
    train(epoch,optimizer)
    acc = accuracy(args,model,TestData)
    print('====> Test Data Accuracy:{:.4f}'.format(acc))

Accuracy = accuracy(args,model,TestData)



#Processing all of the data through the NN to extract the features
Response= ClinicalF['AUX.STRATIFICATION']
#PandaData is the data containing our data set with all the features
PandaData = Predictors.join(Response,how='right')
model.eval()
Features = model(torch.nn.functional.normalize(torch.tensor(PandaData.iloc[:,0:-1].values)).type(torch.FloatTensor))[1]
FeaturesData = pd.DataFrame(Features.detach().numpy(),index=PandaData.index)
#Save the features
FeaturesData.to_csv('FreeSurfer+NN_Features_2k_CV.csv')
#Save the NN
torch.save(model.state_dict(), 'model')


####################################
# Rigourous comparison between NN features AND 56 expert-selected features
####################################

# Define vectors to store the results
NNAccuracy = np.zeros(args.nMC)
NNAUC = np.zeros(args.nMC)

LRAccuracy = np.zeros(args.nMC)
LRAUC = np.zeros(args.nMC)

for i in range (0,args.nMC):
    #Random permutation and setting all the needed data (Comments in previous sections)
    perm = np.random.permutation(npData.shape[0])
    Data = torch.tensor(npData[perm, :])
    TrainData = Data[0:(args.ntr+args.nval)]
    TrainData[:,0:-1] = torch.nn.functional.normalize(TrainData[:,0:-1])
    ValData = Data[args.ntr:(args.ntr+args.nval)]
    ValData[:,0:-1] = torch.nn.functional.normalize(ValData[:,0:-1])
    TestData = Data[(args.ntr+args.nval):]
    TestData[:,0:-1] = torch.nn.functional.normalize(TestData[:,0:-1])
    SData = npSData[perm, :]
    TrainSData = SData[0:(args.ntr+args.nval)]
    TestSData = SData[(args.ntr+args.nval):]
    # Training NN model
    model = NN1(f=56)
    model = model_init(args,model,5)
    optimizer = optim.Adagrad(model.parameters(),lr=0.01)
    for epoch in range(1, 750+ 1):
        train(epoch,optimizer)
        acc = accuracy(args,model,TestData)
        #print('====> Test Data Accuracy:{:.4f}'.format(acc))
    Accuracy = accuracy(args,model,TestData)
    NNAccuracy[i] = Accuracy 
    auc = AUC(args,model,TestData)
    NNAUC[i] = auc
    #Training Logistic Regression
    lrm = LogisticRegression(penalty='l1', solver='saga', max_iter=10000).fit(TrainSData[:,0:-1],TrainSData[:,-1])
    pred = lrm.predict(TestSData[:,0:-1])
    correct = np.sum(pred==TestSData[:,-1])
    acc = correct/TestSData.shape[0]
    LRAccuracy[i] = acc
    fpr, tpr, thresholds = metrics.roc_curve(TestSData[:,-1], pred)
    LRAUC[i] = metrics.auc(fpr, tpr)
    print('====> MC:{:.0f}, NN Accu:{:.5f}, LR Accu:{:.5f}'.format(i+1,Accuracy,acc))
