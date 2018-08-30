# Shelli Kesler
# https://stackoverflow.com/questions/16228954/how-to-use-mlp-multilayer-perceptron-in-r
# classvar should be numeric (1/0)


library(monmlp)
library(caret)
library(ROCR)

data = read.csv('pathTodata.csv')

set.seed(1)

inTune <- createDataPartition(y = data[,ncol(data)],p=.8,list=FALSE)
train.samp <- data[inTune,];test.samp <- data[-inTune,]
trainX <- as.matrix(train.samp[,-ncol(train.samp)]); trainY <- as.matrix(train.samp[,ncol(train.samp)]) 
testX <- as.matrix(test.samp[,-ncol(test.samp)]); testY <- as.matrix(test.samp[,ncol(test.samp)]) 

fitMLP = monmlp.fit(trainX, trainY, hidden1=6, n.ensemble=15, monotone=1, bag=TRUE)
preds = monmlp.predict(x = testX, weights = fitMLP)

plot(performance(prediction(preds, testY), "tpr","fpr" ) )
performance(prediction(preds, testY), "auc" )@y.values[[1]]
