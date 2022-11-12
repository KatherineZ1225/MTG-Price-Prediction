library(data.table)
library(caret)
library(Metrics)
library(glmnet)
library(plotmo)
library(lubridate)
set.seed(7)


#read in data, notice the path will always look like this because the assumed working directory is the repo level folder
train<-fread("./project/volume/data/interim/train_mtg.csv")
test<-fread("./project/volume/data/interim/test_mtg.csv")
example_sub<-fread("./project/volume/data/raw/example_submission.csv")

##########################
# Prep Data for Modeling #
##########################

train$current_date<-as_date(train$current_date)
train<-train[order(-current_date)]

# subset out only the columns to model

drops<- c('id')

train<-train[, !drops, with = FALSE]
test<-test[, !drops, with = FALSE]


#keep<-c("current_price","future_price","rarity","Land","BS_199","BS_200","BS_201")

#train<-train[, keep, with = FALSE]
#test<-test[, keep, with = FALSE]

#save the response var because dummyVars will remove
train_y<-train$future_price

test$result <- 0.5
test_y<-test$result
test$future_price<-0

# work with dummies

dummies <- dummyVars(future_price ~ ., data = train)
train<-predict(dummies, newdata = train)
test<-predict(dummies, newdata = test)

train<-data.table(train)
test<-data.table(test)

########################
# Use cross validation #
########################
train<-as.matrix(train)
test<-as.matrix(test)

gl_model<-glmnet(train, train_y, alpha = 1,family="gaussian")
error_DT<-NULL
for (i in 1:length(unclass(gl_model)$lambda)){
  model_lambda<-unclass(gl_model)$lambda[i]
  pred<-predict(gl_model,s=model_lambda, newx = test,type="response")
  error<-mean(log(test_y,pred[,1]))
  new_row<-c(model_lambda,error)
  error_DT<-rbind(error_DT,new_row)
}

# ll() function will produce inf, I don't know how to deal with it, so I use log and use the minimum absolutely value which seems available
error_DT<-data.table(error_DT)
setnames(error_DT,c("V1","V2"),c("lambda","error"))

bestlam<-error_DT[abs(error)==min(abs(error_DT$error),na.rm = TRUE)]$lambda

####################################
# fit the model to all of the data #
####################################


#now fit the full model

#fit a logistic model
gl_model<-glmnet(train, train_y, alpha = 1,family="gaussian")

#save model
saveRDS(gl_model,"./project/volume/models/gl_model.model")

test<-as.matrix(test)

#use the full model
pred<-predict(gl_model,s=bestlam, newx = test)


predict(gl_model,s=bestlam, newx = test,type="coefficients")

# predict(gl_model,s=0.218231957, newx = test,type="coefficients")

#########################
# make a submision file #
#########################


#our file needs to follow the example submission file format.
#we need the rows to be in the correct order

example_sub$future_price<-pred


#now we can write out a submission
fwrite(example_sub,"./project/volume/data/processed/submit.csv")
