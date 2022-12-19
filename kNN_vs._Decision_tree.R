rm(list=ls())
wd=(dirname(rstudioapi::getActiveDocumentContext()$path)) 
setwd(wd)

library(caret)

###Import our dataset####

rdata = read.csv('StudentsPerformance.csv')
rdata

str(rdata)
summary(rdata)
table(rdata$gender)

###mosaicplot to see relation between categorical features####

mosaicplot(gender ~ test.preparation.course, data=rdata, col=c('Blue', 'Red') )

mosaicplot(gender ~ race.ethnicity, data=rdata, col=c('Blue', 'Red', 'Yellow') )

mosaicplot(gender ~ parental.level.of.education,
           data=rdata, col=c('Blue', 'Red', 'Yellow'),
           cex.axis = 0.4)

mosaicplot(gender ~ lunch, data=rdata, col=c('Blue', 'Red', 'Yellow') )


###Oversampling to get balanced data####

rdata$gender = as.factor(rdata$gender)

set.seed(1)

brdata <- upSample(x = subset(rdata,select = -c(gender)),
                   y = rdata$gender, yname = "gender")

table(brdata$gender)

###boxplots####

ggplot(brdata, aes(x = gender, y = reading.score, fill = gender)) + 
  geom_boxplot()

ggplot(brdata, aes(x = gender, y = writing.score, fill = gender)) + 
  geom_boxplot()

ggplot(brdata, aes(x = gender, y = math.score, fill = gender)) + 
  geom_boxplot()

ggplot(brdata, aes(x = test.preparation.course
                   , y = math.score, fill = gender)) + 
  geom_boxplot()

ggplot(brdata, aes(x = test.preparation.course
                   , y = reading.score, fill = gender)) + 
  geom_boxplot()

ggplot(brdata, aes(x = test.preparation.course
                   , y = writing.score, fill = gender)) + 
  geom_boxplot()

###turning test.preparation.course to dummy####

brdata$test.preparation.course = 
  ifelse(brdata$test.preparation.course == "none", 0, 1) 

str(brdata)

###all features to dummy ####
#library(fastDummies)
#sbrdata <- subset(brdata, select = -c(gender))

#dbrdata <- dummy_cols(sbrdata) 
###delete some columns####
data = subset(brdata, select = -c(race.ethnicity,
                                 parental.level.of.education,
                                 lunch) )

data

set.seed(12)


train <- createDataPartition(data[,"gender"],p=0.8,list=FALSE)

data.trn <- data[train,]

data.tst <- data[-train,] 


ctrl  <- trainControl(method  = "cv", number= 10,
                      summaryFunction = multiClassSummary)
nrow(data)
sqrt(nrow(data))

###kNN####

fit.knn <- train(gender ~ ., data = data.trn, method = "knn",
  trControl = ctrl,
  preProcess = c("center","scale"), 
  tuneGrid =data.frame(k=seq(5,100,by=2)))
  

pred.knn <- predict(fit.knn, data.tst) 

pred.knn

data.tst$probs <- predict(fit.knn,
                          data.tst, type = 'prob')

data.tst$probs$male

data.tst$numeric.gender = 
  ifelse(data.tst$gender == "female", 0, 1)

kNN_RMSE <- RMSE(data.tst$probs$male, data.tst$numeric.gender)
kNN_RMSE

confusionMatrix(table(data.tst[,"gender"], pred.knn)) 

print(fit.knn) 
plot(fit.knn) 
plot(fit.knn,metric = "Specificity")
plot(fit.knn,metric = "Sensitivity")
plot(fit.knn,metric = "Kappa")
plot(fit.knn,metric = "F1")

###Decision tree####
library(lattice)

library(rpart.plot)

ctrl  <- trainControl(method  = "cv", number  = 10,
                      summaryFunction = multiClassSummary)

fit.tree <- train(gender ~ ., data = data.trn, method = "rpart",
                trControl = ctrl,
                tuneLength=25)

print(fit.tree)
plot(fit.tree)

pred_dt <- predict(fit.tree,data.tst) 
pred_dt

dt_probs <- predict(fit.tree,
                          data.tst, type = 'prob')

dt_probs
confusionMatrix(table(data.tst[,"gender"],pred_dt)) 

rpart.plot(fit.tree$finalModel)
##################################################3

dt_RMSE <- RMSE(dt_probs$male, data.tst$numeric.gender)
dt_RMSE

###Roc Curve####
library(pROC)
par(pty="s")
kNN_ROC <- roc(data.tst$gender ~ as.numeric(pred.knn), plot=TRUE,
               print.auc=TRUE,col="green",lwd =4,
               legacy.axes=TRUE, main="ROC Curves")


DT_ROC <- roc(data.tst$gender ~ as.numeric(pred_dt),
              plot=TRUE,print.auc=TRUE,
              col="blue",lwd = 4,print.auc.y=0.4,
              legacy.axes=TRUE,add = TRUE)

legend("bottomright",legend=c("kNN","DT"),
       col=c("green","blue"),lwd=4)

###DT with max-depth tunned####

fit.tree.max <- train(gender ~ ., data = data.trn, method = "rpart2",
                  trControl = ctrl,
                  tuneLength=25)

print(fit.tree.max)
plot(fit.tree.max)

pred_dt_mx <- predict(fit.tree.max,data.tst) 

confusionMatrix(table(data.tst[,"gender"],pred_dt_mx))

###ROC for tree with max-depth####
par(pty="s")
kNN_ROC <- roc(data.tst$gender ~ as.numeric(pred.knn), plot=TRUE,
               print.auc=TRUE,col="green",lwd =4,
               legacy.axes=TRUE, main="ROC Curves")


DT_ROC <- roc(data.tst$gender ~ as.numeric(pred_dt_mx),
              plot=TRUE,print.auc=TRUE,
              col="blue",lwd = 4,print.auc.y=0.4,
              legacy.axes=TRUE,add = TRUE)

legend("bottomright",legend=c("kNN","DT_with_MAX"),
       col=c("green","blue"),lwd=4)
