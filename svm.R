# Shelli Kesler
# http://svmlight.joachims.org

library(klaR)
library(caret)
data = read.csv('pathToData.csv')
set.seed(1)
inTune <- createDataPartition(y = data[,ncol(data)],p=.8,list=FALSE)
train.samp <- data[inTune,];test.samp <- data[-inTune,]
testY <- as.factor(test.samp[,ncol(test.samp)]) 
modelSVM <- svmlight(outcomeVar ~ ., data = train.samp, pathsvm = 'pathTo/svm_light')
predictions <- predict(modelSVM, test.samp)
confusionMatrix(predictions$class,testY)
