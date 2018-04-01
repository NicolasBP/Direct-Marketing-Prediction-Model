source('Modeling.R')

# Prediction of donr variable in test set.

model.donr <- bagging.tree.fit(data.train.std.c, c('bagging.tree.fit1', 7))
prediction.valid <- prediction.bagging.tree(model.donr, data.valid.std.c, NULL)
prediction.test <- prediction.bagging.tree(model.donr, data.test.std) # post probs for test data

profit <- cumsum(14.5*c.valid[order(prediction.valid, decreasing=T)]-2)
plot(profit) # see how profits change as more mailings are made
n.mail.valid <- which.max(profit) # number of mailings that maximizes profits
c(n.mail.valid, max(profit)) # report number of mailings and maximum profit

cutoff <- sort(prediction.valid, decreasing=T)[n.mail.valid+1] # set cutoff based on n.mail.valid
chat.valid <- ifelse(prediction.valid>cutoff, 1, 0)

# Oversampling adjustment for calculating number of mailings for test set

n.mail.valid <- which.max(profit)
tr.rate <- .1 # typical response rate is .1
vr.rate <- .5 # whereas validation response rate is .5
adj.test.1 <- (n.mail.valid/n.valid.c)/(vr.rate/tr.rate) # adjustment for mail yes
adj.test.0 <- ((n.valid.c-n.mail.valid)/n.valid.c)/((1-vr.rate)/(1-tr.rate)) # adjustment for mail no
adj.test <- adj.test.1/(adj.test.1+adj.test.0) # scale into a proportion
n.mail.test <- round(n.test*adj.test, 0) # calculate number of mailings for test set

cutoff.test <- sort(prediction.test, decreasing=T)[n.mail.test+1] # set cutoff based on n.mail.test
chat.test <- ifelse(prediction.test>cutoff.test, 1, 0) # mail to everyone above the cutoff
table(chat.test)


# Prediction of damt variable in test set.

model.damt <- gam.fit.models(data.train.std.y, c('gam.fit.8'), NULL)
yhat.test <- prediction.gam(model.damt, data.test.std, c('gam.fit.8'))
hist(yhat.test)
mean(yhat.test)
# FINAL RESULTS

# Save final results for both classification and regression

length(chat.test) # check length = 2007
length(yhat.test) # check length = 2007
chat.test[1:10] # check this consists of 0s and 1s
yhat.test[1:10] # check this consists of plausible predictions of damt

ip <- data.frame(chat=chat.test, yhat=yhat.test) # data frame with two variables: chat and yhat
write.csv(ip, file="NBP.csv", row.names=FALSE)

