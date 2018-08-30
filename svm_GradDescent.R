# Shelli Kesler
# RJ Liu

# SVM with Gradient Descent

set.seed(1)
library(MLmetrics)

#import and clean data
train <-
  read.csv('./train.csv', header = FALSE)[, c(1, 3, 5, 11:13, 15)] #import only continuous vars
test <-
  read.csv('./test.csv', header = FALSE)[, c(1, 3, 5, 11:13, 15)]
adult <- rbind(train, test)
colnames(adult) <-
  c('age',
    'fnlwgt',
    'edyr',
    'capital_gain',
    'capital_loss',
    'hours_per_week',
    'income')
adult[adult == "?"] <- NA #recode missing data
adult <- adult[complete.cases(adult), ] #include only complete cases
adult$age <- as.numeric(adult$age)
#write.csv(file = './adult.csv', adult)

adultNormZ <- as.data.frame(scale(adult[1:6])) #normalize features
adultNormZ <- cbind(adultNormZ, adult$income)
colnames(adultNormZ)[7] <- "income"
#table(adultNormZ$income) # there are 4 levels instead of 2 so fix this
adultNormZ[adultNormZ == "<=50K."] <- "<=50K"
adultNormZ[adultNormZ == ">50K."] <- ">50K"
adultNormZ <- droplevels(adultNormZ)
sample.size <- dim(adultNormZ)[1]
#table(adultNormZ$income)

#convert class vector to -1, 1
adultNormZ$income = as.integer(adultNormZ$income)
adultNormZ$income[adultNormZ$income == 1] <- -1
adultNormZ$income[adultNormZ$income == 2] <- 1

#partition into 10% validation, 10% test, and 80% training
inTrain <- sample(1:sample.size, floor(sample.size*0.8), replace = F)
train.samp <- adultNormZ[inTrain, ] 
train.sample.size <- dim(train.samp)[1]
rest.samp <- adultNormZ[-inTrain, ]
rest.sample.size <- dim(rest.samp)[1]
inValid <- sample(1:rest.sample.size, floor(rest.sample.size*0.5), replace = F)
test.samp <- rest.samp[-inValid, ] 
valid.samp <- rest.samp[inValid, ]

# candidate regularization constants
lambda <- c(1e-3, 1e-2, 1e-1, 1) 

#search for best parameters with SGD
#some of this code was adapted from:
#https://github.com/faridcher/machine-learning-course/blob/master/Solutions/mlclass-ex6/svmTrain.R
#https://maviccprp.github.io/a-support-vector-machine-in-just-a-few-lines-of-python-code/

svm_sgd <- function(train.samp, lambda0) {
  #setting parameters in svm_sgd
  epochs <- 50 # number of epochs for each fixed lambda
  steps <- 300 # steps in each epoch
  acc.heldout <- rep(0, 10 * epochs) 
  w <- rep(0, 10 * epochs) 
  #initial values for w and b
  a <- matrix(0, nrow = 6, ncol = 1) 
  b <- 0
  
  for (e in 1:epochs) {
    #loop across epochs
    eta = 1 / (0.01 * e + 50) 
    inEpoch <-sample(1:train.sample.size, 50, replace = F) #randomly select 50 samples for evaluation
    X.heldout <-
      train.samp[inEpoch, ] 
    X.train <-
      train.samp[-inEpoch, ] 
    sampleN <- dim(X.train)[1]
    for (s in 1:steps) {
      #loop across descent steps
      sample_ID <- sample(1:sampleN, 1, replace = T) 
      x.s <- t(X.train[sample_ID, 1:6])
      y.s <- X.train[sample_ID, 7]
      grad <- y.s * (t(a) %*% x.s + b)
      if (grad >= 1) {
        a.new <- a - eta * lambda0 * a # update a
        b.new <- b # update b
      }
      else{
        a.new <- a - eta * (lambda0 * a - y.s * x.s)
        b.new <- b + eta * y.s
      }
      a <- a.new
      b <- b.new
      if (s %% 30 == 0) {
        w[floor(s / 30) + 10 * (e - 1)] <- t(a) %*% a
        predict.heldout <- sign(as.matrix(X.heldout[, 1:6]) %*% a + b)
        predict.heldout[which(predict.heldout<=0)] <- 0
        heldoutLabel <- unname(X.heldout[, 7])
        heldoutLabel[which(heldoutLabel<=0)] <- 0
        acc.heldout[floor(s / 30) + 10 * (e - 1)] <- Accuracy(predict.heldout, heldoutLabel)
      }
    }
  }
  
  model <- list(a, b, w, acc.heldout) # return model
  return(model)
}

#run svm_sgd on the training dataset
a_all <- matrix(0, nrow = 6, ncol = 4)
b_all <- rep(0, 4)
w_all <- matrix(0, nrow = 500, ncol = 4) # nrow=300/30*50
acc.heldout_all <- matrix(0, nrow = 500, ncol = 4)
acc.valid <- rep(0, 4)

for (l in 1:4) {
  lambda0 <- lambda[l]
  res <- svm_sgd(train.samp, lambda0)
  a_all[, l] <- res[[1]]
  b_all[l] <- res[[2]]
  w_all[, l] <- res[[3]]
  acc.heldout_all[, l] <- res[[4]]
  #run svm_sgd on the validation dataset
  valid.samp <- as.matrix(valid.samp)
  validLabel <- unname(valid.samp[, 7])
  validLabel[which(validLabel<=0)] <- 0
  predict.valid <- sign(as.matrix(valid.samp[, 1:6]) %*% a_all[, l] + b_all[l])
  predict.valid <- unname(predict.valid[, 1])
  predict.valid[which(predict.valid<=0)] <- 0
  acc.valid[l] <- Accuracy(predict.valid, validLabel)
}

#choose the optimal lambda and run the model on testing data
a.opt <- a_all[, which(acc.valid == max(acc.valid))]
b.opt <- b_all[which(acc.valid == max(acc.valid))]
test.samp <- as.matrix(test.samp)
predict.test <- sign(as.matrix(test.samp[, 1:6]) %*% a.opt + b.opt)
predict.test[which(predict.test<=0)] <- 0
testLabel <- unname(test.samp[, 7])
testLabel[which(testLabel<=0)] <- 0
acc.test <- Accuracy(predict.test, testLabel)


plot.idx <- seq(0.1, 50, length.out = 500)

#plot the accuracy every 30 steps, for each value of the regularization constant
plot(plot.idx, acc.heldout_all[,1], lty = 1, type = "l", pch = c(18, 19, 20),
  xlab = "Epoch", ylab = "Accuracy", ylim = c(0.2, 1), col = "red",lwd = 2)
par(new = TRUE)
plot(plot.idx, acc.heldout_all[,2], lty = 1, type = "l", pch = c(18, 19, 20),
     xlab = "Epoch", ylab = "Accuracy", ylim = c(0.2, 1), col = "green",lwd = 2)
par(new = TRUE)
plot(plot.idx, acc.heldout_all[,3], lty = 1, type = "l", pch = c(18, 19, 20),
     xlab = "Epoch", ylab = "Accuracy", ylim = c(0.2, 1), col = "black",lwd = 2)
par(new = TRUE)
plot(plot.idx, acc.heldout_all[,4], lty = 1, type = "l", pch = c(18, 19, 20),
     xlab = "Epoch", ylab = "Accuracy", ylim = c(0.2, 1), col = "blue",lwd = 2)
legend("bottomright",c("1e-3", "1e-2", "1e-1", "1"),
  bty = "n",col = c("red", "green", "black", "blue"),lwd = 1.5)

#plot the magnitude of the coefficient vector every 30 steps, 
#for each value of the regularization constant
plot(plot.idx, w_all[,1], lty = 1, type = "l", pch = c(18, 19, 20),
     xlab = "Epoch", ylab = "Magnitude", ylim = c(0, 6), col = "red",lwd = 2)
par(new = TRUE)
plot(plot.idx, w_all[,2], lty = 1, type = "l", pch = c(18, 19, 20),
     xlab = "Epoch", ylab = "Magnitude", ylim = c(0, 6), col = "green",lwd = 2)
par(new = TRUE)
plot(plot.idx, w_all[,3], lty = 1, type = "l", pch = c(18, 19, 20),
     xlab = "Epoch", ylab = "Magnitude", ylim = c(0, 6), col = "black",lwd = 2)
par(new = TRUE)
plot(plot.idx, w_all[,4], lty = 1, type = "l", pch = c(18, 19, 20),
     xlab = "Epoch", ylab = "Magnitude", ylim = c(0, 6), col = "blue",lwd = 2)
legend("topright",c("1e-3", "1e-2", "1e-1", "1"),
       bty = "n",col = c("red", "green", "black", "blue"),lwd = 1.5)