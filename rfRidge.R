#Shelli Kesler
#Random forest classification with nested feature selection via regularized regression (binomial)
#nperm = number of times to repeat
#part = percentage of data for training set e.g. .75
#alpha = regularization scheme: ridge = 0, lasso = 1, elasticnet = between 0 and 1 e.g. 0.4, 0.5, etc.

rfRidge <- function(data,nperm,part,alpha){
library(caret)
library(randomForest)

rfauc <- matrix(data=NA,nrow=nperm,ncol=1)

for (i in 1:nperm) {
inTrain = createDataPartition(y = data[,ncol(data)],p=part,list=FALSE)

training = up_data[inTrain,]
testing = up_data[-inTrain,]

X <- as.matrix(training[,-ncol(training)]); Y <- training[,ncol(training)]
testY = testing[,ncol(testing)]

suppressMessages(library(glmnet))
logcv = cv.glmnet(X,as.factor(Y), family = "binomial", alpha=alpha)
tmp_coeffs <- coef(logcv, s=logcv$lambda.min)
rvars = matrix(tmp_coeffs@Dimnames[[1]][tmp_coeffs@i + 1])
rvars = rvars[-1,]

newX = X[,rvars]
new.test = cbind(testing[,rvars],testing[,ncol(testing)])

mtry=round(sqrt(ncol(newX)))

rfFitcv <- randomForest(newX,Y,mtry=mtry, ntree=1000,replace=TRUE)
probs<-predict(rfFitcv,newdata=new.test,type="prob")[,2]
detach("package:glmnet")
library(AUC)
rfauc[i,] <- auc(roc(probs,testY))
}
meanAUC <- mean(rfauc)
write.csv(rfauc,file="AUC.csv")
return(meanAUC)
}