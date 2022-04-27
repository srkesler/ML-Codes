#Shelli Kesler
#10/16/19
#Cox Proportional Hazards with 10-folds CV
#NOTE: dataset requires "time" and "status" vars  
#rselect = 1 for ridge penalty, 0 for no penalty
#thresh = p value threshold for significant covariates

cox10 <- function(data,rselect,thresh){
  library(survAUC)
  
  set.seed(1234)
  #storage variables
  cphauc <- matrix(data=NA,nrow=10,ncol=1)
  pf <- matrix(data=NA,nrow=ncol(data)-2,ncol=10)
  
  #Randomly shuffle the data
  rdata<-data[sample(nrow(data)),]
  folds <- cut(seq(1,nrow(rdata)),breaks=10,labels=FALSE)

  #10-fold CV
    for(i in 1:10){
      inTest <- which(folds==i,arr.ind=TRUE)
      testData <- rdata[inTest, ]
      trainData <- rdata[-inTest, ]
      
      sobj <- with(trainData,Surv(time,status == 1))
      tempfeats = as.matrix(subset(trainData, select=-c(time,status)))
      
     if (rselect == 1) {
      # do feature selection/reduction
      # tempf = as.formula(paste("sobj", paste(names(tempfeats), collapse = " + "), sep = " ~ "))
      # temp.fit <- coxph(tempf,data = trainData)
      #    # get coefficient pvals and subset the significant ones
      #    coeffs <- coef(summary(temp.fit))
      #    pvals <- as.matrix(coeffs[,5])
      #    selfeats = rownames(subset(pvals,pvals < .05))
      #    if (length(selfeats)>0) {
      #      selfeats = selfeats
      #    } else {
      #      selfeats = tempfeats
      #    }

      #regression with ridge penalty
       
      #f = as.formula(paste("sobj",paste("ridge",paste("(",paste(tempfeats,collapse = ","),",theta=1)")),sep="~"))
       f = as.formula(paste("sobj ~ ridge(tempfeats,theta=1)"))
       train.fit  <- coxph(f,data = trainData)
       coeffs <- coef(summary(train.fit))
       pf[,i] <- as.matrix(coeffs[,6])
     } else {
      #f = as.formula(paste("sobj", paste(tempfeats, collapse = " + "), sep = " ~ "))
       f = as.formula(paste("sobj ~ tempfeats"))
       train.fit  <- coxph(f,data = trainData)
       coeffs <- coef(summary(train.fit))
       pf[,i] <- as.matrix(coeffs[,5])
     }
      
      lp <- predict(train.fit)
      lpnew <- predict(train.fit, newdata=testData)
      Surv.rsp <- Surv(trainData$time, trainData$status)
      Surv.rsp.new <- Surv(testData$time, testData$status)
      t = round(range(trainData$time))
      times <- seq(t[1],t[2],5)
      aucdat = AUC.cd(Surv.rsp, Surv.rsp.new, lp, lpnew, times)
      cphauc[i,] <- aucdat$iauc
    }
  rownames(pf) <- colnames(tempfeats)
  pf <- cbind(pf,rowMeans(pf))
  colnames(pf) <- c("F1","F2","F3","F4","F5","F6","F7","F8","F9","F10","meanP")
  coxPH10aucs <<- cphauc
  covpvals <<- pf
  sel = rownames(subset(pf,pf[,11] < thresh))
  meanAUC <- mean(cphauc)
  results <- list("meanAUC" = meanAUC, "sign. covars" = sel)
  return(results)
}