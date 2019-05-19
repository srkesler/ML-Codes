#Shelli Kesler
#Random forest classification with nested feature selection via recursive feature elimination
#nperm = number of times to repeat
#part = percentage of data for training set e.g. .75

rfRFE <- function(data,nperm,part){
library(caret)
library(randomForest)
library(AUC)
library(doMC)

registerDoMC(cores=detectCores())

rfauc <- matrix(data=NA,nrow=nperm,ncol=1)

for (i in 1:nperm) {
inTrain = createDataPartition(y = data[,ncol(data)],p=part,list=FALSE)

training = up_data[inTrain,]
testing = up_data[-inTrain,]

X <- as.matrix(training[,-ncol(training)]); Y <- training[,ncol(training)]
testY = testing[,ncol(testing)]

control <- rfeControl(functions=rfFuncs,method="loocv", allowParallel = TRUE)
rfe.results <- rfe(X,Y,rfeControl = control)

newX = X[,rfe.results$optVariables]
new.test = cbind(testing[,rfe.results$optVariables],testing[,ncol(testing)])

mtry=round(sqrt(ncol(newX)))

rfFitcv <- randomForest(newX,Y,mtry=mtry, ntree=1000,replace=TRUE)
probs<-predict(rfFitcv,newdata=new.test,type="prob")[,2]
rfauc[i,] <- auc(roc(probs,testY))
}
meanAUC <- mean(rfauc)
write.csv(rfauc,file="AUC.csv")
return(meanAUC)
}