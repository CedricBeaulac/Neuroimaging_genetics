## This R script will run an analysis to select SNPs related to MRI imaging phenotypes from ADNI2
## The specific data and study considered are described in Beulac et al. (2022)
## We will use a combination of univariate screening and bgsmtr. For the latter,
## difficulty choosing the tunning parameters for the two methods provided in the software
## package lead to the use of hold-out data. At some point we should make this the default for bgsmtr.


## Read in the data
#install.packages("readxl")
library(bgsmtr)
setwd("")
X.expert<-read.csv('X.expert.csv')
X.expert <- X.expert[,2:ncol(X.expert)]
Y.expert<-read.csv('Y.expert.csv')
Y.expert <- Y.expert[,2:ncol(Y.expert)]
X.NN<-read.csv('X.NN.csv')
X.NN <- X.NN[,2:ncol(X.NN)]
Y.NN<-read.csv('Y.NN.csv')
Y.NN <- Y.NN[,2:ncol(Y.NN)]
SNP.grouping<-read.csv('SNP_Mapping.csv')


## Analysis Steps
## 2c) The run the model with the chosen parameters on the entire dataset and select those SNPs that have 95% HDI's excluding zeros

## Expert features
lambda1.grid<-c(10,100,1000,10000,100000)
lambda2.grid<-c(10,100,1000,10000,100000)

l1 = lambda1.grid[1]
l2 = lambda2.grid[3]

SNP.names<-colnames(X.expert)
## get the gene membership
group<-SNP.grouping$GENE[match(SNP.names,SNP.grouping$RSID)]

## Run on the whole data with selected values
fit.expert<-bgsmtr(X = t(X.expert), Y = t(Y.expert),group=group,tuning='WAIC',lam_1_fixed = l1,lam_2_fixed = l2,iter_num = 10000,burn_in = 5000)

## extract the estimates
W.est.expert<-fit.expert$Gibbs_W_summaries$W_post_mean

rowSums(sign(fit.expert$Gibbs_W_summaries$W_2.5_quantile)*sign(fit.expert$Gibbs_W_summaries$W_97.5_quantile)==1)
#or
scores= apply(abs(fit.expert$Gibbs_W_summaries$W_post_mean/fit.expert$Gibbs_W_summaries$W_post_sd),1,max)
topex <- colnames(X.expert)[order(scores,decreasing = TRUE)][1:100]

rowSums(sign(fit.expert$Gibbs_W_summaries$W_2.5_quantile)*sign(fit.expert$Gibbs_W_summaries$W_97.5_quantile)==1)

## NN features
lambda1.grid<-c(10,100,1000,10000,100000)
lambda2.grid<-c(10,100,1000,10000,100000)

l1 = lambda1.grid[1]
l2 = lambda2.grid[3]

SNP.names<-colnames(X.NN)
## get the gene membership
group<-SNP.grouping$GENE[match(SNP.names,SNP.grouping$RSID)]

## Run on the whole data with selected values
fit.NN<-bgsmtr(X = t(X.NN), Y = t(Y.NN),group=group,tuning='WAIC',lam_1_fixed = l1,lam_2_fixed = l2,iter_num = 10000,burn_in = 5000)

## extract the estimates
scores= apply(abs(fit.NN$Gibbs_W_summaries$W_post_mean/fit.NN$Gibbs_W_summaries$W_post_sd),1,max)
topnn <- colnames(X.NN)[order(scores,decreasing = TRUE)][1:100]

which(topnn %in% topex)


rowSums(sign(fit.NN$Gibbs_W_summaries$W_2.5_quantile)*sign(fit.NN$Gibbs_W_summaries$W_97.5_quantile)==1)
