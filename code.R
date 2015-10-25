# code for  Coursera Machine Learning project.

#### plug right libraries
library(caret)
library(plyr)


####  download data
download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv",destfile="pml-training.csv")
Data <-read.csv("pml-training.csv",header = TRUE,sep = ",", stringsAsFactors=FALSE)
Data$classe <- as.factor(Data$classe)

download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", destfile="pml-testing.csv")
valData <- read.csv("pml-testing.csv",header = TRUE,sep = ",", stringsAsFactors=FALSE)

#### TODO:
# create lists of predictors and outcome to generalize experiments

# explore more optiosn about the multi-level factor outcomes.
# http://amunategui.github.io/multinomial-neuralnetworks-walkthrough/
  
# separate the training data into training and test. Treat original test data as validations.


#######Split training into training and test 
set.seed(1000)
inTrain = createDataPartition(Data$classe, p = 0.6)[[1]]

NZR <- nearZeroVar(Data) #define all data which might be useless because it's near zero

trData = Data[ inTrain,-NZR]
tstData = Data[-inTrain,-NZR]

#######identify predictors and outcomes

#full  predictors 
outcomeName_1 <- 'classe'
predictorNames_1 <- names(trData)[
  !names(trData) %in% c("X","user_name","raw_timestamp_part_1","raw_timestamp_part_2","cvtd_timestamp","num_window",outcomeName_1)]


# Impute NAs for measurements with median v alue.  
# this is to prepare data for models which cannot work well with NAs (lm,glm etc.)
impute.median=function(x){
  x<-as.numeric(as.character(x)) #first convert each column into numeric if it is from factor
  x[is.na(x)] =median(x, na.rm=TRUE) #convert the item with NA to median value from the column
  x #display the column
}
trData[,c(predictorNames_1)] <- data.frame(apply(trData[,c(predictorNames_1)],2,impute.median))
tstData[,c(predictorNames_1)] <- data.frame(apply(tstData[,c(predictorNames_1)],2,impute.median))

#TODO: removed insignificant predictors   


####### Pre-processing section

# TODO: right now no selection happened based on correlation.
# define parameters which are highly correlated
M <- abs(cor(trData_nzr[,-93], use="complete.obs"))
diag(M) <- 0
correlated_var <- which(M>0.8, arr.ind=T)



###### training gym

## GBM without any cross validation with full list of predictors.
trGBM <- train(trData[,predictorNames_1],trData[,outcomeName_1], method="gbm")  ### around 20 minutes of calculation!!!
pr_GBM <- predict(trGBM, tstData[,predictorNames_1])
confusionMatrix(pr_GBM,tstData[,outcomeName_1])
## accuracy=0.961

# save model
save(trGBM,file="simple_gbm.RData")
# load("simple_gbm.RData") # load model as needed.

### Identify the most important predictors.
PredImpGBM <- data.frame(predictors=rownames(varImp(trGBM)$importance),importance=varImp(trGBM)$importance$Overall)
predictorNames_2 <-as.character(PredImpGBM[PredImpGBM$importance>10,]$predictors)

#### GBM with cross validation with just first 13 most important predictors from previous data model
GBM_ctrl <- trainControl(classProbs = TRUE, method='cv', returnResamp = 'none',number=3)
trGBM_1 <- train(trData[,predictorNames_2],trData[,outcomeName_1], method="gbm", preProc=c("center","scale"), trControl=GBM_ctrl)
pr_GBM_1 <- predict(trGBM_1, tstData[,predictorNames_2])
confusionMatrix(pr_GBM_1,tstData[,outcomeName_1])
## accuracy with CV=0.9397

# save model
save(trGBM_1,file="gbm_cv_13pred.RData")
# load("simple_gbm.RData") # load model as needed.


#### GBM without cross validation with just first 13 most important predictors from previous data model
trGBM_2 <- train(trData[,predictorNames_2],trData[,outcomeName_1], method="gbm", preProc=c("center","scale"))
pr_GBM_2 <- predict(trGBM_2, tstData[,predictorNames_2])
confusionMatrix(pr_GBM_2,tstData[,outcomeName_1])
## accuracy withou CV = 0.94

save(trGBM_2,file="gbm_13pred.RData")
# load("simple_gbm.RData") # load model as needed.

############### Random Forrest
set.seed(1313)

## Random forrest with limited set of predictors (taken from GBM)
GBM_ctrl <- trainControl(classProbs = TRUE, method='cv', returnResamp = 'none',number=3)
trRF <- train(trData[,predictorNames_2],trData[,outcomeName_1], method="rf")
pr_RF <- predict(trRF,tstData[,predictorNames_2])
confusionMatrix(pr_RF,tstData[,outcomeName_1])
## accuracy 0.9875

save(trRF,file="rf_13pred.RData")
# load("simple_gbm.RData") # load model as needed.


## Random forrest with full set of predictors. not calculated.
trRF_1 <- train(trData[,predictorNames_1],trData[,outcomeName_1], method="rf")
pr_RF_1 <- predict(trRF_1,tstData[,predictorNames_2])


##### Neural networks
## with limited set of predictors
trNN <- train(trData[,predictorNames_2],trData[,outcomeName_1], method="nnet", maxit=1000, trace=T)
pr_NN <- predict(trNN,tstData[,predictorNames_2])
confusionMatrix(pr_NN,tstData[,outcomeName_1]) # accuracy= 0.6492

save(trNN,file="NN_13pred.RData")
# load("simple_gbm.RData") # load model as needed.


## NN with all predictors but with less iterations (100)
trNN_1 <- train(trData[,predictorNames_1],trData[,outcomeName_1], method="nnet", maxit=100, trace=T)
pr_NN_1 <- predict(trNN,tstData[,predictorNames_2])
confusionMatrix(pr_NN_1,tstData[,outcomeName_1]) # accuracy= 0.6492 - very weird result because it's exactly the same as fo rprevious experiment.

save(trNN_1,file="nn_allpred.RData")
# load("simple_gbm.RData") # load model as needed.



######Predictions on validation sample.
pred1 <- predict(trGBM,valData[,predictorNames_1]) # GBM full set of preditors, no CV
pred2 <- predict(trGBM_1,valData[,predictorNames_2]) # GBM limited predictors, CV
pred3 <- predict(trGBM_2,valData[,predictorNames_2])  #GBM  limited predictors, no CV

pred_4 <-predict(trRF,valData[,predictorNames_2]) # RF full set of preditors, no CV
pred_5 <- predict(trNN, valData[,predictorNames_2]) # NN with 1000 maxit, 13 predictors, no CV

# not calculated pred_5 <-predict(trRF,valData[,predictorNames_2]) # GBM full set of preditors, no CV











########## Binary experiment
#### let's pretend that we need to measure just binary outcome. A=1; B,C,D,E = 0
trData_nzr$outcome <- ifelse(trData$classe=='A',"A","other")
trData_nzr$outcome <- as.factor(trData_nzr$outcome)

# let's use GBM for binary outcomes
GBM_ctrl<- trainControl(method='cv', number=3, returnResamp='none', summaryFunction = twoClassSummary, classProbs = TRUE)
trGBM_bin <- train(outcome~.,data=trData_nzr[,-94], method="gbm", preProc=c("center","scale"), metric='ROC', trControl=GBM_ctrl)
pr_GBM_bin <- predict(trGBM_bin, tstData_nzr)


trGLM <- train(classe~., data=trData_nzr, family="gaussian", method="glm")
pr_GLM <- predict(trGLM,tstData_nzr)
