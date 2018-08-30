#Shelli Kesler 03/26/18
#regularized regression
#categorical variables must be numerically coded

library(glmnet)
library(caret)
bc = read.csv('pathToData.csv')

set.seed(1)
X = as.matrix(bc[,c(2,3,5,7,10,12)])
Y = as.matrix(bc$MHD)

# look at unregularized regression
fit = lm(Y~X)
summary(fit)
par(mfrow=c(2,2));plot(fit,main="Regression Diagnostics")
# http://data.library.virginia.edu/diagnostic-plots/


inTrain = createDataPartition(y = bc$MHD,p=.80,list=FALSE)
testY = Y[-inTrain,]

a = 0 #regularization scheme: ridge = 0, lasso = 1, elasticnet = between 0 and 1 e.g. 0.4, 0.5, etc.

rfit = glmnet(X[inTrain,],Y[inTrain], alpha=a)
rcv = cv.glmnet(X[inTrain,],Y[inTrain],alpha=a)
plot(rcv)
lam = rcv$lambda.min
lam
rpred = predict(rfit,s=lam ,newx=X[-inTrain,])
sst <- sum((Y[inTrain] - mean(Y[inTrain]))^2)
sse <- sum((rpred - testY)^2)
rsq <- 1 - sse / sst
rsq 
Fstat <- (rsq/(1-rsq))*((nrow(X[-inTrain,])-ncol(X[-inTrain,])-1)/(ncol(X[-inTrain,])))
df2 <- nrow(X[-inTrain,])-ncol(X[-inTrain,])-1
pvalF <- 1-pf(Fstat,ncol(X[-inTrain,]),df2)
pvalF
pvalF/2

# get variables retained by lasso/elastic net
coef(rcv,s=rcv$lambda.min)

# for binary outcome
a = 1
fit = glmnet(X[inTrain,], as.factor(Y[inTrain]), family = "binomial", alpha=a)
logcv = cv.glmnet(X[inTrain,],as.factor(Y[inTrain]), family = "binomial", alpha=a)
plot(logcv)
lam = logcv$lambda.min
lam
rpred = predict(fit,s=lam,newx=X[-inTrain,],type="class")
confusionMatrix(rpred,testY)
coef(logcv,s=logcv$lambda.min)


