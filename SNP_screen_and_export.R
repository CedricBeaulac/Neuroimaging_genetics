## This R script will run an analysis to select SNPs related to MRI imaging phenotypes from ADNI2
## The specific data and study considered are described in Beulac et al. (2022)
## We will use a combination of univariate screening and bgsmtr. For the latter,
## difficulty choosing the tunning parameters for the two methods provided in the software
## package lead to the use of hold-out data. At some point we should make this the default for bgsmtr.


## Read in the data
#install.packages("readxl")
library(mice)
setwd("")
SNP.data<-read.csv('GeneData.csv')
SNP.grouping<-read.csv('SNP_Mapping.csv')
MRI.expert.data<-read.csv('ResExpeFeatures.csv')
MRI.NN.data<-read.csv('ResAutoFeatures.csv')

## Remove zeroed NN features
MRI.NN.data.clean<-MRI.NN.data[,colSums(abs(MRI.NN.data))!=0]
SNP.names<-colnames(SNP.data[,2:ncol(SNP.data)])


## Analysis Steps
## 1) Prescreening: loop through with simple linear regression with p-value threshohd 0.01
## 1b) We should present the result of the presecreening in the paper as well.
## 2) To examine SNP associations jointly, we take those SNPs that have a p-value <= 0.01 and include them in the multivariate model.
## 2b) We select the tunning parameters by looking at prediction error on 20% holdout data
## 2c) The run the model with the chosen parameters on the entire dataset and select those SNPs that have 95% HDI's excluding zeros






# For each SNP: each MRI measure is regressed onto the SNP and the p-value is extracted
# If at least one of these p-value is <=0.01 we retain the SNP else is is screened out

X.expert<-SNP.data[,2:ncol(SNP.data)]
Y.expert<-MRI.expert.data[,2:ncol(MRI.expert.data)]
SNP.expert.pvalue<-matrix(0,nrow=ncol(Y.expert),ncol = ncol(X.expert))

## now extract the p-value
for (i in 1:ncol(X.expert))
{
  SNP <- X.expert[,i]
  for (j in 1:ncol(Y.expert))
  {
    MRI<-Y.expert[,j]
    SNP.expert.pvalue[j,i]<-summary(lm(MRI~SNP))$coefficients[,4][2]
  }
}

#obtain the minimum p-value for each SNP
SNP.expert.pmin<-apply(SNP.expert.pvalue,2,min)

#Extract top 25 SNPs
topex <- order(SNP.expert.pmin)[1:100]
SNP.Names.Ex = colnames(X.expert)[topex]

# keep relevant SNPs for bayesian analysis
SNP.expert.retain<-SNP.expert.pmin<=0.01
X.expert.screened<-X.expert[,SNP.expert.retain]
#or
X.expert.screened<-X.expert[,topex]

## deal with missing values for X
X.expert.screened.missing<-which(is.na(X.expert.screened),arr.ind=TRUE)
nrow(X.expert.screened.missing) ## there are two hundred missing values

## remove SNPs with missing data
#X.expert.screened<-X.expert.screened[ , colSums(is.na(X.expert.screened)) == 0]

## we could use imputation procedures - do you want to investigate?
X.imput = mice(X.expert.screened)
X.expert.screened = complete(X.imput)

## Alternatively, we could remove subjects with missing values
#Y.expert <-Y.expert[ rowSums(is.na(X.expert.screened)) == 0, ]
#X.expert.screened<-X.expert.screened[ rowSums(is.na(X.expert.screened)) == 0, ]


dim(X.expert.screened)

## deal with missing values for Y
Y.expert.missing<-which(is.na(Y.expert),arr.ind=TRUE)
nrow(Y.expert.missing) ## there are no missing values
dim(Y.expert)

# Export data set for Compute Canada
write.csv(X.expert.screened,'X.expert.csv')
write.csv(Y.expert,'Y.expert.csv')

## next apply screening for the neural network measures
X.NN<-SNP.data[,2:ncol(SNP.data)]
Y.NN<-MRI.NN.data.clean[,2:ncol(MRI.NN.data.clean)]
SNP.NN.pvalue<-matrix(0,nrow=ncol(Y.NN),ncol = ncol(X.NN))


## now extract the p-value
for (i in 1:ncol(X.NN))
{
  SNP <- X.NN[,i]
  for (j in 1:ncol(Y.NN))
  {
    MRI<-Y.NN[,j]
    SNP.NN.pvalue[j,i]<-summary(lm(MRI~SNP))$coefficients[,4][2]
  }
}

#obtain the minimum p-value for each SNP
SNP.NN.pmin<-apply(SNP.NN.pvalue,2,min)

#Extract top 25 SNPs, match them with their genes and match them with Expert SNPs
topNN <- order(SNP.NN.pmin)[1:100]
SNP.Names.NN = colnames(X.NN)[topNN]
group<-SNP.grouping$GENE[match(SNP.Names.NN,SNP.grouping$RSID)]
SNP.Names.NN
group
which(SNP.Names.Ex %in% SNP.Names.NN)
which(SNP.Names.NN %in% SNP.Names.Ex)

SNP.NN.retain<-SNP.NN.pmin<=0.01
X.NN.screened<-X.NN[,SNP.NN.retain]
#or
X.NN.screened<-X.NN[,topNN]


## deal with missing values for X
X.NN.screened.missing<-which(is.na(X.NN.screened),arr.ind=TRUE)
nrow(X.NN.screened.missing) ## there are 150 missing values
## remove SNPs with missing data
## Cedric: we could use imputation procedures - do you want to investigate?
X.imput = mice(X.NN.screened)
X.NN.screened = complete(X.imput)


Y.NN <-Y.NN[ rowSums(is.na(X.NN.screened)) == 0, ]
X.NN.screened<-X.NN.screened[rowSums(is.na(X.NN.screened)) == 0, ]
dim(X.NN.screened) ## we lose a good number of SNPs - 26 SNPs

## deal with missing values for Y
Y.NN.missing<-which(is.na(Y.NN),arr.ind=TRUE)
nrow(Y.NN.missing) ## there are no missing values


# Export data set for Compute Canada
write.csv(X.NN.screened,'X.NN.csv')
write.csv(Y.NN,'Y.NN.csv')

# Cross-validation was done using and alternative code on cluster of PC
