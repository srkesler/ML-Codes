#Shelli Kesler 12/05/16
#Unsupervised random forest classification using Kmeans clustering and out-of-bag (OOB) prediction
# NOTE: X must contain complete cases only

rfClassUnsupervised <- function(X,mtry){
  library(randomForest)
  library(caret)
  library(AUC)
  library(cluster)
  set.seed(100)
  nsubs=nrow(X)
  rfprox <<- randomForest(X,mtry=mtry,ntree=500,proximity=TRUE) #no outcome specified so RF will run unsupervised
  Crf <<- kmeans(rfprox$proximity,2,iter.max = 100,nstart = 25) #create 2 groups based on kmeans clustering
  rf <- randomForest(X,as.factor(Crf$cluster),ntree = 500) #test how well the clustering assignments are predicted by X 
  predictions <- predict(rf) #if no testing data is specified, the OOB prediction is given
  probs <<- predict(rf,type="prob")[,2]
  CM<-confusionMatrix(predictions,as.factor(Crf$cluster))
  myclass<-as.vector(CM$table)
  nsucc<-(myclass[1]+myclass[4]) # true negatives + true positives
  sens<-(myclass[4]/(myclass[4]+myclass[3]))
  spec<-(myclass[1]/(myclass[1]+myclass[2]))
  acc<-nsucc/nsubs
  binomtest <- binom.test(nsucc,nsubs,.5, alternative = "two.sided",conf.level = 0.95)
  rfauc <- auc(roc(probs,as.factor(Crf$cluster)))
  plot(roc(probs,as.factor(Crf$cluster)),col="red")
  S <-list("acc"=acc,"pval"=binomtest$p.value,"AUC"=rfauc,"sens"=sens,"spec"=spec)
  return(S)
}
