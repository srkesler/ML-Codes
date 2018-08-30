# Shelli Kesler
# https://www.rdocumentation.org/packages/h2o/versions/2.8.4.4/topics/h2o.randomForest
# https://github.com/h2oai/h2o-tutorials/blob/master/tutorials/gbm-randomforest/GBM_RandomForest_Example.R#L48

set.seed(1)

library(h2o)
h20.init()

train.h2o <- h2o.importFile(path = '/pathToData.csv')
test.h2o <- h2o.importFile(path = '/pathToData.csv')

train.h2o[,1] <- as.factor(train.h2o[,1])
test.h2o[,1] <- as.factor(test.h2o[,1])

rf1 <- h2o.randomForest(x = 2:ncol(train.h2o), y = 1, training_frame = train.h2o, validation_frame = test.h2o, ntree = 10, max_depth = 4)
h2o.hit_ratio_table(rf1,valid = T)[1,2]

rf2 <- h2o.randomForest(x = 2:ncol(train.h2o), y = 1, training_frame = train.h2o, validation_frame = test.h2o, ntree = 10, max_depth = 8)
h2o.hit_ratio_table(rf2,valid = T)[1,2]

rf3 <- h2o.randomForest(x = 2:ncol(train.h2o), y = 1, training_frame = train.h2o, validation_frame = test.h2o, ntree = 10, max_depth = 16)
h2o.hit_ratio_table(rf3,valid = T)[1,2]

rf4 <- h2o.randomForest(x = 2:ncol(train.h2o), y = 1, training_frame = train.h2o, validation_frame = test.h2o, ntree = 20, max_depth = 4)
h2o.hit_ratio_table(rf4,valid = T)[1,2]

rf5 <- h2o.randomForest(x = 2:ncol(train.h2o), y = 1, training_frame = train.h2o, validation_frame = test.h2o, ntree = 20, max_depth = 8)
h2o.hit_ratio_table(rf5,valid = T)[1,2]

rf6 <- h2o.randomForest(x = 2:ncol(train.h2o), y = 1, training_frame = train.h2o, validation_frame = test.h2o, ntree = 20, max_depth = 16)
h2o.hit_ratio_table(rf6,valid = T)[1,2]

rf7 <- h2o.randomForest(x = 2:ncol(train.h2o), y = 1, training_frame = train.h2o, validation_frame = test.h2o, ntree = 30, max_depth = 4)
h2o.hit_ratio_table(rf7,valid = T)[1,2]

rf8 <- h2o.randomForest(x = 2:ncol(train.h2o), y = 1, training_frame = train.h2o, validation_frame = test.h2o, ntree = 30, max_depth = 8)
h2o.hit_ratio_table(rf8,valid = T)[1,2]

rf9 <- h2o.randomForest(x = 2:ncol(train.h2o), y = 1, training_frame = train.h2o, validation_frame = test.h2o, ntree = 30, max_depth = 16)
h2o.hit_ratio_table(rf9,valid = T)[1,2]
