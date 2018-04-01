# PREDICT 422 Practical Machine Learning

# Course Project - Example R Script File

# OBJECTIVE: A charitable organization wishes to develop a machine learning
# model to improve the cost-effectiveness of their direct marketing campaigns
# to previous donors.

# 1) Develop a classification model using data from the most recent campaign that
# can effectively capture likely donors so that the expected net profit is maximized.

# 2) Develop a prediction model to predict donation amounts for donors - the data
# for this will consist of the records for donors only.

# load the data
charity <- read.csv('charity.csv') # load the "charity.csv" file

# predictor transformations

charity.t <- charity
charity.t$avhv <- log(charity.t$avhv)
#hist(charity.t$avhv)
charity.t$root_10_lgif <- charity.t$lgif^(1/10)
charity.t$log_root_3_lgif <- (log(1 + charity.t$lgif))^(1/3)

charity.t$log_incm <- log(1+charity.t$incm)
charity.t$log_inca <- log(1 + charity.t$inca)
charity.t$log_rgif <- log(1+charity.t$rgif)
charity.t$log_tgif <- log(1+charity.t$tgif)
charity.t$log_tlag <- log(1 + charity.t$tlag)
charity.t$log_agif <- log(1 + charity.t$agif)
charity.t$root_3_plow <- (charity.t$plow)^(1/3)

hist(charity.t$log_inca, xlab = "log_inca", ylab = "Frequency", main = "log_inca Variable Distribution")

# #charity.t$damt <- (charity.t$damt)^(1/5)
# hist((charity.t$tdon[charity.t['damt']>=1])^(1/2))
# par(mfrow=c(1,1))
# plot(jitter(charity.t$damt[charity.t['damt']>=1]), log(charity.t$tgif[charity.t['damt']>=1]))
# hist(log(charity.t$tgif[charity.t['damt']>=1]))
# test <-ifelse(charity.t$reg3[charity.t['damt']>=1]==1,1,0)
# test2 <- ifelse(charity.t$reg4[charity.t['damt']>=1]==1,1,0)
# test <- test + test2
# boxplot(charity.t$damt[charity.t['damt']>=1] ~ test2)
# plot(test, charity.t$damt[charity.t['damt']>=1])
# # plot(charity.t$z, charity.t$damt)
# # plot(jitter(charity.t$home), charity.t$damt)
# 
# plot(charity.t$plow[charity.t['damt']>=1], charity.t$damt[charity.t['damt']>=1])
# hist(charity.t$root_3_plow[charity.t['damt']>=1])
# 
# test <- charity.t
# transformation <- test$tlag[test['damt']>=1]
# transformation2 <- log(1 + test$tlag[test['damt']>=1])
# par(mfrow = c(1,2))
# hist(transformation)
# hist(transformation2)
# plot(transformation, test$damt[test['damt']>=1])
# library(cobs)
# cobs_spline <- cobs(transformation, test$damt[test['damt']>=1])
# plot(cobs_spline)
# test<-na.omit(test)
# transformation <- log(1 + test$hinc[test['damt']>=1])
# fit=smooth.spline(transformation,test$damt[test['damt']>=1],cv = TRUE)
# fit$df #2.001305
# fit <- gam(test$damt[test['damt']>=1]~s(transformation, 2.001305))
# fit$
# lines(fit,col="red",lwd=2)
# 
# cobs_spline$degree
# abline(lm(test$damt[test['damt']>=1]~transformation - 1))
# abline(lm(test$damt[test['damt']>=1]~transformation))
# model <- lm(test$damt[test['damt']>=1]~poly(transformation,2))
# abline(model)
# abline(h=15)
# plot(transformation2, test$damt[test['damt']>=1])
# abline(lm(test$damt[test['damt']>=1]~transformation2 - 1))
# abline(lm(test$damt[test['damt']>=1]~transformation2))
# par(mfrow = c(1,2))
# add further transformations if desired
# for example, some statistical methods can struggle when predictors are highly skewed

# set up data for analysis

data.train <- charity.t[charity$part=="train",]
x.train <- data.train[,c(2:21,25:33)]
c.train <- data.train[,22] # donr
n.train.c <- length(c.train) # 3984
y.train <- data.train[c.train==1,23] # damt for observations with donr=1
n.train.y <- length(y.train) # 1995

data.valid <- charity.t[charity$part=="valid",]
x.valid <- data.valid[,c(2:21,25:33)]
c.valid <- data.valid[,22] # donr
n.valid.c <- length(c.valid) # 2018
y.valid <- data.valid[c.valid==1,23] # damt for observations with donr=1
n.valid.y <- length(y.valid) # 999

data.test <- charity.t[charity$part=="test",]
n.test <- dim(data.test)[1] # 2007
x.test <- data.test[,c(2:21,25:33)]

x.train.mean <- apply(x.train, 2, mean)
x.train.sd <- apply(x.train, 2, sd)
x.train.std <- t((t(x.train)-x.train.mean)/x.train.sd) # standardize to have zero mean and unit sd
apply(x.train.std, 2, mean) # check zero mean
apply(x.train.std, 2, sd) # check unit sd
data.train.std.c <- data.frame(x.train.std, donr=c.train) # to classify donr
data.train.std.y <- data.frame(x.train.std[c.train==1,], damt=y.train) # to predict damt when donr=1

x.valid.std <- t((t(x.valid)-x.train.mean)/x.train.sd) # standardize using training mean and sd
data.valid.std.c <- data.frame(x.valid.std, donr=c.valid) # to classify donr
data.valid.std.y <- data.frame(x.valid.std[c.valid==1,], damt=y.valid) # to predict damt when donr=1

x.test.std <- t((t(x.test)-x.train.mean)/x.train.sd) # standardize using training mean and sd
data.test.std <- data.frame(x.test.std)

