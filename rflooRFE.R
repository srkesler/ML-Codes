#Shelli Kesler 10/12/15
# Leave-one-out cross-validation for random forest classification
# Includes nested feature selection via RFE
# NOTE: the outcome variable must be in the last column of the dataframe and must be a factor (e.g. "no", "yes")
# NOTE: dataframe must contain complete cases only

rflooRFE <- function(data,ntree,nperm){
set.seed(100)
library(caret)
library(randomForest)
library(AUC)
library(doMC)
  
registerDoMC(cores=detectCores())
nsubs=nrow(data)

# set up storage variables
probs<-matrix(data=NA, nrow=nsubs,ncol=1)
class.labels <-as.factor(data[,ncol(data)])
rfe.feats <- matrix(data=NA,nrow=ncol(data)-1,ncol=1)
rfauc <- matrix(data=NA,nrow=nperm,ncol=1)

#outer loop to repeat nperm times since rfe is stocastic
for (t in 1:nperm){
#loocv loop
for (i in 1:nsubs) {
  #hold out 1 sample for testing
  training = data[-i,];testing = data[i,] 
  
  #feature selection on training data
  mtry1 <- round(sqrt(ncol(data)-1))
  X <- as.matrix(training[,-ncol(training)]); Y <- training[,ncol(training)]
  control <- rfeControl(functions=rfFuncs,method="loocv", allowParallel = TRUE)
  rfe.results <- rfe(X,Y,rfeControl = control)
  
  #store selected features from each loop in one variable 
  add.col<-function(df, new.col) {n.row<-dim(df)[1]
  length(new.col)<-n.row
  cbind(df, new.col)}
  rfe.feats <- add.col(rfe.feats,as.matrix(rfe.results$optVariables))
  
  #create new train/test data with selected features
  new.train = cbind(training[,rfe.results$optVariables],training[,ncol(training)])
  new.test = cbind(testing[,rfe.results$optVariables],testing[,ncol(testing)])
  
  #fit a random forest model using selected features
  trainX <- as.matrix(new.train[,-ncol(new.train)]); trainY <- new.train[,ncol(new.train)]
  mtry2 <- round(sqrt(ncol(trainX)))
  rfFitcv <- randomForest(trainX,trainY,mtry=mtry2, ntree=ntree,replace=TRUE)
  probs[i,]<-predict(rfFitcv,newdata=new.test,type="prob")[,2]
  write.csv(probs,file=paste0('probs_',t,'.csv'))
  rfauc[t,] <- auc(roc(probs,class.labels))
  
  }
  }
#calculate the mean AUC across the nperm iterations
meanAUC <- mean(rfauc)
write.csv(rfe.feats[,-1],file="rfeFeats.csv")
write.csv(rfauc,file="AUC.csv")

#calculate the percentage of iterations that each feature is selected by RFE
Tf = as.data.frame(table(rfe.feats[,-1]))
perT = Tf$Freq/ncol(rfe.feats[,-1])
rfeSum = cbind(Tf,perT)
write.csv(rfeSum,file="rfeSum.csv")

return(meanAUC)
}