#Shelli Kesler 10/12/15
# Leave-one-out cross-validation for random forest classification
# up-samples the training data within each loop
# NOTE: the outcome variable must be in the last column of the dataframe
# NOTE: dataframe must contain complete cases only
# Usage: loocv(dataframe)

rfloocv <- function(data){
set.seed(100)
library(caret)
library(randomForest)
library(AUC)
library(pRF)
library(multtest)
  
nsubs=nrow(data)
# set up storage variables
myclass<-matrix(data=NA, nrow=nsubs,ncol=4)
# gini<-matrix(data=NA, nrow=ncol(data)-1, ncol=nsubs)
pvals<-matrix(data=NA, nrow=ncol(data)-1, ncol=nsubs)
probs<-matrix(data=NA, nrow=nsubs,ncol=1)
class.labels <-as.factor(data[,ncol(data)])

#loocv loop
for (i in 1:nsubs) {
  training = data[-i,];testing = data[i,]
  #training = upSample(training[,-ncol(training)], training[,ncol(training)]) #balance classes
  trainX <- as.matrix(training[,-ncol(training)]); trainY <- training[,ncol(training)]
  mtry <- round(sqrt(ncol(trainX)))
  #fit RF model to training data
  rfFitcv <- randomForest(trainX,trainY,mtry=mtry, ntree=500,replace=TRUE)
  #G <- as.data.frame(rfFitcv$importance)
  # gini[,i] <- G$MeanDecreaseGini
  #predict classes using RF model
  predictions <- predict(rfFitcv,newdata=testing,type="response")
  #permute feature pvals
  p.test <- pRF(response=training[,ncol(testing)],predictors=training[,-ncol(testing)],n.perms=100,mtry=mtry,type="classification",alpha=0.05,ntree=500,seed=100)
  pvals[,i] <- p.test$Res.table$p.value
  probs[i,]<-predict(rfFitcv,newdata=testing,type="prob")[,2]
  CM<-confusionMatrix(predictions,testing[,ncol(testing)]) 
  myclass[i,]<-as.vector(CM$table)
}
#performance
TN<-sum(myclass[,1]) #column1 = true negatives
FP<-sum(myclass[,2]) #column2 = false positives
FN<-sum(myclass[,3]) #column3 = false negatives
TP<-sum(myclass[,4]) #column4 = true positives
nsucc<-(TN+TP)
acc<-nsucc/nsubs
sens<-TP/(TP+FN)
spec<-TN/(FP+TN)
ntrials<-TN+FP+FN+TP
binomtest <- binom.test(nsucc,ntrials,.5,alternative = "two.sided",conf.level = 0.95)
rfauc <- auc(roc(probs,class.labels))
plot(roc(probs,class.labels),col="red")
# rownames(gini)=rownames(rfFitcv$importance)
rownames(pvals)=p.test$Res.table$Feature.id
feat.pvals = apply(pvals, 1, function(x) mean(x));feat.pvals=as.matrix(feat.pvals)
# write.csv(gini,file="GiniLOOCV.csv")
# gini <<- gini
write.csv(feat.pvals,file="FeaturepvalsLOOCV.csv")
labels <- c('Accuracy','pval','AUC','Sens','Spec')
S <-c(acc,binomtest$p.value,rfauc,sens,spec)
stats <- data.frame(matrix(ncol = 5, nrow = 0))
colnames(stats)=labels; stats[1,]=S
write.csv(stats,file="LOOCVresults.csv")
results <-list("binom"=binomtest,"AUC"=rfauc, "sensitivity"=sens,"specificity"=spec)
return(results)
}