#Shelli Kesler 10/20/15
#Random forest regression using out-of-bag (OOB) prediction
# NOTE: the outcome variable must be in the last column of the dataframe
# NOTE: dataframe must contain complete cases only
# Example Usage: rfReg(mydata,'Outcome1')

rfReg <- function(data,title){
  library(randomForest)
  library(caret)
  library(miscTools)
  library(pRF)
  library(multtest)
  set.seed(100)
  nsubs=nrow(data)
  nfeat=ncol(data)-1
  X <- as.matrix(data[,-ncol(data)]); Y <- data[,ncol(data)]
  mtry <- round(sqrt(ncol(X)))
  rfFitreg <<- randomForest(X,Y,mtry=mtry,ntree=500,replace=TRUE,importance=TRUE)
  p.test <- pRF(response=Y,predictors=X,n.perms=100,mtry=mtry,type="regression",alpha=0.05,ntree=500,seed=100)
  save(rfFitreg,file=paste(title,'_RFRfit.RData',sep =""))
  # varImpPlot(rfFitreg,type=1,scale=F,main=title)
  # abline(v=mean(rfFitreg$importance[,1])+sd(rfFitreg$importance[,1]),col="red")
  pvals <-cbind(p.test$Res.table,p.test$obs)
  r2 <- rSquared(Y, Y-predict(rfFitreg, X))
  adjR2 <- 1-(1-r2)*((nsubs-1)/(nsubs-nfeat-1))
  Fstat <- (r2/(1-r2))*((nsubs-nfeat-1)/(nfeat))
  fsquared <- (adjR2/(1-adjR2)) #effect size
  df2 <- (nsubs-nfeat-1)
  #pvalF <- pf(Fstat,nfeat,df2,lower.tail=F) this approach will give an erroneous p value with F stat is high
  pvalF <- 1-pf(Fstat,nfeat,df2)
  labels <- c('R2','adjR2','F2','F','p')
  S <-c(r2,adjR2,fsquared,Fstat,pvalF)
  stats <- data.frame(matrix(ncol = 5, nrow = 0))
  colnames(stats)=labels; stats[1,]=S
  results <- list(stats,pvals,rfFitreg$importance)
  write.csv(results,file=paste(title,'_Results.csv',sep = ""))
  return(results)
}
