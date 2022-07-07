## Data visualization
## The specific data and study considered are described in Beulac et al. (2022)

library(tsne)
library(ggplot2)
setwd("")

AD.status <- read.csv('AD_status.csv')
MRI.expert.data<-read.csv('ResExpeFeatures.csv')
MRI.NN.data<-read.csv('ResAutoFeatures.csv')

## Remove zeroed NN features
MRI.NN.data<-MRI.NN.data[,colSums(abs(MRI.NN.data))!=0]

## 2d embedding for data visualisation
NN.embedding <- tsne(MRI.NN.data[AD.status[,2]=='NC'|AD.status[,2]=='AD',2:ncol(MRI.NN.data)])
NN.embedding.df <- data.frame(cbind(data.frame(NN.embedding),AD.status[AD.status[,2]=='NC'|AD.status[,2]=='AD',2]))
colnames(NN.embedding.df) <- c('X1','X2','Status')
p <- ggplot(NN.embedding.df, aes(X1, X2, colour = Status)) 
p <- p + geom_point(size=3) + theme(text = element_text(size = 20))   
p <- p + theme(text = element_text(size = 20),
               axis.title.x =element_blank(),axis.title.y =element_blank(),
               panel.border = element_blank(),
               panel.grid.major = element_blank(),
               panel.grid.minor = element_blank(),
               axis.ticks = element_blank(),
               axis.text.x = element_blank(),axis.text.y = element_blank(),
               
               legend.position = 'bottom')   
p

MRI.embedding <- tsne(MRI.expert.data[AD.status[,2]=='NC'|AD.status[,2]=='AD',2:ncol(MRI.expert.data)])
MRI.embedding.df <- data.frame(cbind(data.frame(MRI.embedding),AD.status[AD.status[,2]=='NC'|AD.status[,2]=='AD',2]))
colnames(MRI.embedding.df) <- c('X1','X2','Status')
p <- ggplot(MRI.embedding.df, aes(X1, X2, colour = Status)) 
p <- p + geom_point(size=3) + theme(text = element_text(size = 20))   
p <- p + theme(text = element_text(size = 20),
               axis.title.x =element_blank(),axis.title.y =element_blank(),
               panel.border = element_blank(),
               panel.grid.major = element_blank(),
               panel.grid.minor = element_blank(),
               axis.ticks = element_blank(),
               axis.text.x = element_blank(),axis.text.y = element_blank(),
               legend.position = 'bottom')   
p


