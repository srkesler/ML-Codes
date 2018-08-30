# Shelli Kesler 02/03/18
# https://stats.stackexchange.com/questions/61034/naive-bayes-on-continuous-variables

library(caret)
library(klaR)
data = read.csv('/pathToData.csv')

set.seed(1)
inTune <- createDataPartition(y = data[,ncol(data)],p=.8,list=FALSE)
train.samp <- data[inTune,];test.samp <- data[-inTune,]
trainX <- as.matrix(train.samp[,-ncol(train.samp)]); trainY <- as.factor(train.samp[,ncol(train.samp)]) 
testX <- as.matrix(test.samp[,-ncol(test.samp)]); testY <- as.factor(test.samp[,ncol(test.samp)]) 
NBmodel <- train(trainX,trainY,'nb',trControl=trainControl(method='cv',number=10))
predictions <- predict(NBmodel$finalModel,testX)
confusionMatrix(predictions$class,testY)

