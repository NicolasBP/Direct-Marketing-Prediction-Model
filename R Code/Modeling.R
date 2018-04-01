library(MASS)
library(randomForest)
library(gbm)
library(class)
library(leaps)
library(tree)
library(rpart)
library(glmnet)
library(splines)
library(gam)
library(car)
library(boot)
library(pls)
library(subselect)
#source('clusterwise_regression.R')
globaldata = data.train.std.c

############# MODELS ###############

#Model-fitting functions----------------------

lda.fit2 = function(data,...){
  lda.fit2 <- lda(donr ~ avhv, 
                  data)
  return(lda.fit2)
}
# k.fold.tree.fit = rpart(data.train.std.c, control = rpart.control(cp=0, minsplit= 10, xval = 10, maxsurrogate = 0))
# printcp(k.fold.tree.fit)
# plotcp(k.fold.tree.fit, minline = TRUE, col= 4)
# complexity_value = 5.3536e-03
# tree.fit = decision.tree.fit(data.train.std.c, complexity_value)



#par(mai = c (0.1,0.1,0.1,0.1))
#plot(tree.fit, main = "Classification Tree: Donations Data", col = 3, compress=TRUE, branch=0.2, uniform = TRUE)
#text(tree.fit, cex = 0.6, col = 4, use.n = TRUE, fancy = TRUE, fwidth = 0.4, fheight = 0.4, bg = c(5))


# tree_train <- sample(1:nrow(data.train.std.c), nrow(data.train.std.c)/2)
# tree.fit <- tree(donr ~., data.train.std.c)
# cv.tree <- cv.tree(tree.fit)
# plot(cv.tree$size, cv.tree$dev, type = 'b')
# prune.tree <- prune.tree(tree.fit, best = 10)
# plot(prune.tree)
decision.tree.fit = function(data, tuning_params,...){
  decision.tree.fit1 <- function(data, tuning_params,...){
    set.seed(1)
    tree.fit <- tree(donr ~ chld + home + reg2 + reg1 + hinc + wrat, data)  
    decision.tree.fit1 <- prune.tree(tree.fit, best = strtoi(tuning_params[[2]]))
    return(decision.tree.fit1)
  }
  decision.tree.fit2 <- function(data, tuning_params,...){
    set.seed(1)
    tree.fit <- tree(donr ~ reg3 + reg4 + genf + avhv + incm + inca + plow + npro + tgif + lgif + rgif + tdon + tlag + agif, data)  
    decision.tree.fit2 <- prune.tree(tree.fit, best = strtoi(tuning_params[[2]]))
    return(decision.tree.fit2)
  }
  decision.tree.fun <- get(tuning_params[[1]])
  decision.tree.fit <- decision.tree.fun(data,tuning_params, ...)
  return(decision.tree.fit)
}

prediction.decision.tree = function(tree, test_data, tuning_params = NULL){
  yhat <- predict(tree, newdata = test_data)
  return(yhat)
}

bagging.tree.fit = function(data, tuning_params,...){
  bagging.tree.fit1 <- function(data, tuning_params,...){
    set.seed(1)
    data[['donr']]= as.factor(data[['donr']])
    bag.fit = randomForest(donr ~ reg2 + reg3 + reg4 + home + chld + hinc + genf + wrat + avhv + incm + inca + plow + npro + tgif + lgif + rgif + tdon + tlag + agif, data = data, mtry = strtoi(tuning_params[[2]]), importance = TRUE)
    return(bag.fit)
  }
  bag.tree.fun <- get(tuning_params[[1]])
  bag.tree.fit <- bag.tree.fun(data,tuning_params, ...)
  return(bag.tree.fit)
}

prediction.bagging.tree = function(tree, test_data, tuning_params = NULL){
  yhat <- predict(tree, newdata = test_data, type="prob")
  #return(as.numeric(as.character(yhat)))
  return(yhat[,2])
}

boosting.tree.fit = function(data, tuning_params,...){ 
  boosting.tree.fit1 <- function(data, tuning_params,...){
    set.seed(1)
    boost.fit = gbm(donr ~., data = data, distribution = tuning_params[[2]], n.trees = strtoi(tuning_params[[3]]), interaction.depth = strtoi(tuning_params[4]), shrinkage = as.double(tuning_params[5]))
    return(boost.fit)
  }
  boosting.tree.fun <- get(tuning_params[[1]])
  boosting.tree.fit <- boosting.tree.fun(data,tuning_params, ...)
  return(boosting.tree.fit)
}

prediction.boosting.tree = function(tree, test_data, tuning_params = NULL){
  yhat <- predict(tree, newdata = test_data, n.trees = 5000, type = 'response')
  return(yhat)
}

lda.fit = function(data, tuning_params = NULL,...){
  lda.fit1 <- function(data, tuning_params = NULL,...){
    lda.fit1 <- lda(donr ~ ., data)
    return(lda.fit1)
  }
  #Found using the subselect package
  lda.fit2 <- function(data, tuning_params = NULL,...){
    lda.fit2 <- lda(donr ~ reg2 + reg4 + home + chld + wrat + incm + plow + npro + tdon + tlag, data)
    return(lda.fit2)
  }
  #Using the variables in the best decision tree found.
  lda.fit3 <- function(data, tuning_params = NULL,...){
    lda.fit3 <- lda(donr ~ reg1 + reg2 + chld + home + hinc + wrat, data)
    return(lda.fit3)
  }
  
  
  lda.fit.fun <- get(tuning_params[[1]])
  lda.fit <- lda.fit.fun(data,tuning_params, ...)
  return(lda.fit)
}

prediction.lda.fit <- function(lda.fit, test_data, tuning_params){
  predictions <- predict(lda.fit, test_data)
  predictions <- predictions$posterior[,1]
  predictions <- as.data.frame(predictions)
  return(predictions)
#   fun <- function(x){
#     ifelse(x >= tuning_params[1],1,0)
#   }
#   yhat <- apply(predictions, 1, fun)
#   return(yhat)
}

qda.fit = function(data, tuning_params,...){
  qda.fit1 <- function(data, tuning_params = NULL,...){
    qda.fit1 <- qda(donr ~ ., data)
    return(qda.fit1)
  }
  #Found using the subselect package for lda
  qda.fit2 <- function(data, tuning_params = NULL,...){
    qda.fit2 <- qda(donr ~ reg2 + reg4 + home + chld + wrat + incm + plow + npro + tdon + tlag, data)
    return(qda.fit2)
  }
  #Using variables found in decision tree
  qda.fit3 <- function(data, tuning_params = NULL,...){
    qda.fit3 <- qda(donr ~ reg1 + reg2 + chld + home + hinc + wrat, data)
    return(qda.fit3)
  }
  #Found using the subselect package for logistic regression
  qda.fit4 <- function(data, tuning_params = NULL,...){
    qda.fit4 <- qda(donr ~ reg1 + reg2 + home + chld + wrat + incm + plow + npro + tdon + tlag, data)
    return(qda.fit4)
  }
  #Found using the subselect package for logistic regression
  qda.fit5 <- function(data, tuning_params = NULL,...){
    qda.fit5 <- qda(donr ~ reg1 + reg2 + home + chld + wrat + incm + plow + npro + tgif + tdon + tlag, data)
    return(qda.fit5)
  }
  #Found using the subselect package for logistic regression
  qda.fit6 <- function(data, tuning_params = NULL,...){
    qda.fit6 <- qda(donr ~ reg1 + reg2 + home + chld + hinc + wrat + incm + plow + npro + tgif + tdon + tlag, data)
    return(qda.fit6)
  }
  #Found using the subselect package for logistic regression
  qda.fit7 <- function(data, tuning_params = NULL,...){
    qda.fit7 <- qda(donr ~ reg1 + reg2 + home + chld + hinc + wrat + avhv + incm + plow + npro + tgif + tdon + tlag, data)
    return(qda.fit7)
  }
  
  
  qda.fit.fun <- get(tuning_params[[1]])
  qda.fit <- qda.fit.fun(data,tuning_params, ...)
  return(qda.fit)
}

prediction.qda.fit <- function(lda.fit, test_data, tuning_params){
  predictions <- predict(lda.fit, test_data)
  predictions <- predictions$posterior[,1]
  predictions <- as.data.frame(predictions)
  return(predictions)
#   fun <- function(x){
#     ifelse(x >= tuning_params[1],1,0)
#   }
#   yhat <- apply(predictions, 1, fun)
#   #yhat <- rapply(as.list(as.numeric(predictions$posterior[,1])), get_xval_profits)
#   return(yhat)
}

log.reg.fit = function(data, tuning_params,...){
  log.reg.fit1 <- function(data, tuning_params,...){
    log.reg.fit1 <- glm(donr ~ ., data, family=binomial("logit"))
    return(log.reg.fit1)
  }
  log.reg.fit2 <- function(data, tuning_params,...){
    log.reg.fit2 <- glm(donr ~ reg1 + reg2 + home + chld + wrat + incm + plow + npro + tdon + tlag, data, family = binomial("logit"))
    return(log.reg.fit2)
  }
  log.reg.fit3 <- function(data, tuning_params,...){
    log.reg.fit3 <- glm(donr ~ reg1 + reg2 + home + chld + wrat + incm + plow + npro + tgif + tdon + tlag, data, family = binomial("logit"))
    return(log.reg.fit3)
  }
  log.reg.fit4 <- function(data, tuning_params,...){
    log.reg.fit4 <- glm(donr ~ reg1 + reg2 + home + chld + hinc + wrat + incm + plow + npro + tgif + tdon + tlag, data, family = binomial("logit"))
    return(log.reg.fit4)
  }
  log.reg.fit5 <- function(data, tuning_params,...){
    log.reg.fit5 <- glm(donr ~ reg1 + reg2 + home + chld + hinc + wrat + avhv + incm + plow + npro + tgif + tdon + tlag, data, family = binomial("logit"))
    return(log.reg.fit5)
  }
  
  log.reg.fit.fun <- get(tuning_params[[1]])
  log.reg.fit <- log.reg.fit.fun(data,tuning_params, ...)

  return(log.reg.fit)
}

prediction.log.reg <- function (log.reg.fit, test_data, tuning_params){
  predictions <- predict(log.reg.fit, newdata = test_data, type = "response")
  #yhat <- ifelse(predictions >= tuning_params[1],1,0)
  return(predictions)
}

knn.fit = function(train_data, tuning_params, test_data){
  train_data = as.data.frame(train_data)
  test_data = as.data.frame(test_data)
  #print(train.data[1,])
  train.X = train_data[, ! colnames(train_data) %in% c('donr')]
  test.X = test_data[, ! colnames(test_data) %in% c('donr')]
  train.Y = train_data$donr
  knn.post.prob =knn(train.X,test.X,train.Y,k=tuning_params[1], prob = TRUE)
  knn.pred =knn(train.X,test.X,train.Y,k=tuning_params[1], prob = FALSE)
  knn.post.prob = as.numeric(attr(knn.post.prob,'prob'))
  df <- data.frame(knn.pred, knn.post.prob)
  fun <- function(x){
    if(x['knn.pred'] == 0){
      return(1 - as.numeric(x['knn.post.prob']))
    }
    return(as.numeric(x['knn.post.prob']))
  }
  df['knn.post.prob'] <- apply(df, 1, fun)
  return(df[,'knn.post.prob'])
}

prediction.knn.fit <- function(knn.fit, test_data, tuning_params){
  return(knn.fit)
}

regression.tree.fit = function(data, tuning_params,...){
  regression.tree.fit1 <- function(data, tuning_params,...){
    set.seed(1)
    regression.tree.fit1 <- tree(damt ~ reg1 + reg2 + reg3 + reg4 + home + chld + hinc + genf + wrat + avhv + incm + inca + plow + npro + tgif + lgif + rgif + tdon + tlag + agif, data = data)
#     plot(regression.tree.fit1)
#     text(regression.tree.fit1, pretty = 0)
#     cv.regression.tree.fit1 <- cv.tree(regression.tree.fit1)
#     plot(cv.regression.tree.fit1$size, cv.regression.tree.fit1$dev, type = 'b') # tree with 10 nodes performs best
    regression.tree.fit1 <- prune.tree(regression.tree.fit1, best = 10)
    return(regression.tree.fit1)
  }
  regression.tree.fit2 <- function(data, tuning_params,...){
    set.seed(1)
    regression.tree.fit2 <- tree(damt ~ ., data = data)
#         plot(regression.tree.fit2)
#         text(regression.tree.fit2, pretty = 0)
#         cv.regression.tree.fit2 <- cv.tree(regression.tree.fit2)
#         plot(cv.regression.tree.fit2$size, cv.regression.tree.fit2$dev, type = 'b') # tree with 10 nodes performs best
    regression.tree.fit2 <- prune.tree(regression.tree.fit2, best = 10)
    return(regression.tree.fit2)
  }
  regression.tree.fit.fun <- get(tuning_params[[1]])
  regression.tree.fit <- regression.tree.fit.fun(data,tuning_params, ...)
  return(regression.tree.fit)
}

prediction.regression.tree.fit <- function(regression.tree.fit, test_data, tuning_params){
  prediction <- predict(regression.tree.fit, test_data)
  return(prediction)
}

ridge.reg.fit <- function(data, tuning_params,...){
  ridge.reg.fit1 <- function(data, tuning_params, ...){
    set.seed(1)
    X <- model.matrix(damt~reg1 + reg2 + reg3 + reg4 + home + chld + hinc + genf + wrat + avhv + incm + inca + plow + npro + tgif + lgif + rgif + tdon + tlag + agif + root_10_lgif + log_root_3_lgif + log_incm + log_inca + log_rgif + log_tgif + log_tlag + log_agif + root_3_plow, data = data)
    ridge.reg.fit1 = glmnet(X,data[,'damt'], family = "poisson", alpha = 0, lambda = as.double(tuning_params[[2]]))
    return(ridge.reg.fit1)
  }  
  
  ridge.reg.fit2 <- function(data, tuning_params, ...){
    set.seed(1)
    X <- model.matrix(damt~reg1 + reg2 + reg3 + reg4 + home + chld + hinc + genf + wrat + avhv + incm + inca + plow + npro + tgif + lgif + rgif + tdon + tlag + agif + root_10_lgif + log_root_3_lgif + log_incm + log_inca + log_rgif + log_tgif + log_tlag + log_agif + root_3_plow, data = data)
    ridge.reg.fit2 = glmnet(X,data[,'damt'], alpha = 0, lambda = as.double(tuning_params[[2]]))
    return(ridge.reg.fit2)
  }
  ridge.reg.fun <- get(tuning_params[[1]])
  ridge.reg.fit <- ridge.reg.fun(data,tuning_params, testData)
  return(ridge.reg.fit)
}

prediction.ridge.reg <- function(ridge.reg.fit, test_data, tuning_params){
  newX <- model.matrix(~.-damt,data=test_data)
  prediction <- predict(ridge.reg.fit, newx = newX, type = "response", s = strtoi(tuning_params[[2]]))
  return(prediction)
}

lasso.fit <- function(data, tuning_params,...){
  lasso.fit1 <- function(data, tuning_params, ...){
    set.seed(1)
    X <- model.matrix(damt~., data = data)
    lasso.fit1 = glmnet(X,data[,'damt'], alpha = 1, lambda = as.double(tuning_params[[2]]))
    return(lasso.fit1)
  }  
  
  # lasso.fit2 <- function(data, tuning_params, ...){
  #   set.seed(1)
  #   X <- model.matrix(damt~., data = data)
  #   lasso.fit2 = glmnet(X,data[,'damt'], alpha = 0, lambda = as.double(tuning_params[[2]]))
  #   return(lasso.fit2)
  # }
  lasso.fun <- get(tuning_params[[1]])
  lasso.fit <- lasso.fun(data,tuning_params, testData)
  return(lasso.fit)
}

prediction.lasso <- function(lasso.fit, test_data, tuning_params){
  newX <- model.matrix(damt~.,data=test_data)
  prediction <- predict(lasso.fit, newx = newX, type = "response", s = strtoi(tuning_params[[2]]))
  return(prediction)
}

gam.fit.models <- function(data, tuning_params, testData){
  gam.fit <- function(data, tuning_params, testData){
    model11 <- gam(damt ~ s(root_10_lgif, 7.378126) + s(log_incm, 4.186313) + s(avhv, 4.186313) + s(log_tgif, 3.713441), data = data)
    return(model11)
  }
  
  gam.fit.1<- function(data, tuning_params, testData){
    model11.1 <- gam(damt ~ s(root_10_lgif, 7.378126) + s(log_incm, 4.186313) + s(avhv, 4.186313), data = data)
    return(model11.1)
  }
  
  gam.fit.2<- function(data, tuning_params, testData){
    model11.2 <- gam(damt ~ s(root_10_lgif, 7.378126) + s(log_incm, 4.186313) + s(avhv, 4.186313) + s(log_tgif, 3.713441) + reg4, data = data)
    return(model11.2)
  }
  
  gam.fit.3<- function(data, tuning_params, testData){
    model11.3 <- gam(damt ~ s(root_10_lgif, 7.378126) + s(log_incm, 4.186313) + s(avhv, 4.186313) + 
                       s(log_tgif, 3.713441) + reg4 + home + chld + hinc + genf + wrat + inca + plow + tdon +
                       tlag + agif, data = data)
    return(model11.3)
  }
  
  gam.fit.4<- function(data, tuning_params, testData){
    model11.4 <- gam(damt ~ s(root_10_lgif, 7.378126) + s(log_incm, 4.186313) + s(avhv, 4.186313) + 
                       s(log_tgif, 3.713441) + reg4 + home + chld + hinc + genf + wrat + inca + plow + tdon +
                       tlag + s(log_agif,2.826113), data = data)
    return(model11.4)
  }
  
  gam.fit.5<- function(data, tuning_params, testData){
    model11.5 <- gam(damt ~ s(root_10_lgif, 7.378126) + s(log_incm, 4.186313) + s(avhv, 4.186313) + 
                       s(log_tgif, 3.713441) + reg4 + home + chld + hinc + genf + wrat + inca + plow + tdon +
                       s(log_agif,2.826113), data = data)
    return(model11.5)
  }
  
  gam.fit.6<- function(data, tuning_params, testData){
    model11.6 <- gam(damt ~ s(root_10_lgif, 7.378126) + s(log_incm, 4.186313) + s(avhv, 4.186313) + 
                       s(log_tgif, 3.713441) + reg4 + home + chld + hinc + genf + inca + plow +
                       s(log_agif,2.826113), data = data)
    return(model11.6)
  }
  
  gam.fit.7<- function(data, tuning_params, testData){
    model11.7 <- gam(damt ~ s(log_root_3_lgif, 2.700608) + s(log_incm, 4.186313) +  
                       s(log_tgif, 3.713441) + s(plow, 4.430904) + s(log_rgif, 2.001117) + 
                       s(tdon, 2.785614) + s(hinc, 2.785614) + s(log_tlag, 2.000029) + #s(log_inca, 4.161474) + hinc + s(wrat,2.562594) + reg3 + home
                       s(log_agif,2.826113) + reg1 + reg2 + reg4 + chld + genf, data = data)
    return(model11.7)
  }
  
  gam.fit.8<- function(data, tuning_params, testData){
    model11.8 <- gam(damt ~ s(lgif, 10.67876) + s(incm, 12.09151) +  
                   s(tgif, 5.120689) + s(wrat,2.562594) + s(plow, 4.430904) + s(rgif, 6.949567) + 
                   s(tdon, 2.785614) + s(hinc, 2.785614) + s(tlag, 2.000007) + #s(inca, 9.17457) + reg3 + home +  
                   s(agif,7.813879)  + reg1 + reg2 + reg4 + chld + genf, data = data)
    return(model11.8)
  }
  
  gam.model.fun <- get(tuning_params[[1]])
  gam.model.fit <- gam.model.fun(data,tuning_params, testData)
  return(gam.model.fit)
}

prediction.gam <-function(gam.fit, test_data, tuning_params){
  prediction <- predict(gam.fit, newdata = test_data)
  return(prediction)
}

lin.reg.models <- function(data, tuning_params, testData){
  lin.reg.fit1 <- function(data, tuning_params, testData){
    fit <- glm.nb(damt ~ ., data = data)
    lin.reg.fit1 <- stepAIC(fit, direction = tuning_params[[2]])
    return(lin.reg.fit1)
  }
  lin.reg.fit2<- function(data, tuning_params, testData){
    fit <- lm(damt ~ ., data = data)
    lin.reg.fit2<- stepAIC(fit, direction = tuning_params[[2]])
    return(lin.reg.fit2)
  }
  lin.reg.fit3 <- function(data, tuning_params, testData){
    number_of_predictors <- ncol(data)-1
    Hmat <- lmHmat(x = data[,1:number_of_predictors], y = data[,ncol(data)])
    solutions <- eleaps(Hmat$mat, H=Hmat$H, r = Hmat$r, criterion = "Wald", nsol = 10)
    cardinality <- strtoi(tuning_params[[2]])
    measurevar <- 'damt'
    predictors <- c(names(data)[solutions$bestsets[cardinality,1:cardinality]])
    string_formula <- paste(measurevar, paste(predictors, collapse=" + "), sep=" ~ ")
    lin.reg.fit3 <- lm(as.formula(string_formula), data = data)
    return(lin.reg.fit3)
  }
  lin.reg.fit4 <- function(data, tuning_params, testData){
    lin.reg.fit4 <- lm(damt ~ reg2 + reg3 + reg4 + home + chld + hinc + genf + lgif + tdon + log_root_3_lgif + log_incm + log_rgif + log_tgif + log_agif + root_3_plow, data = data)
    return(lin.reg.fit4)
  }
  lin.reg.fit5<- function(data, tuning_params, testData){
    lin.reg.fit5 <- lm(damt ~ reg2 + reg3 + reg4 + home + chld + hinc + genf + lgif + tdon + log_incm + log_rgif + log_tgif + log_agif + root_3_plow, data = data)
    return(lin.reg.fit5)
  }
  #Best Model (found using lin.reg.fit3)
  lin.reg.fit6 <- function(data, tuning_params, testData){
    lin.reg.fit6 <- lm(damt ~ reg1 + reg2 + reg3 + reg4 + home + chld + hinc + genf + 
                         wrat + avhv + incm + inca + plow + npro + tgif + lgif + rgif + 
                         tdon + tlag + agif + root_10_lgif + log_root_3_lgif + log_incm + 
                         log_inca + log_rgif + log_tgif + log_tlag + log_agif + root_3_plow, data = data)
    return(lin.reg.fit6)
  }
  lin.reg.fit.fun <- get(tuning_params[[1]])
  lin.model.fit <- lin.reg.fit.fun(data,tuning_params, testData)
  return(lin.model.fit)
}

prediction.lin.reg <- function(lin.reg.fit, test_data, tuning_params){
  prediction <- predict(lin.reg.fit, newdata = test_data)
  return(prediction)
}

pc.reg.fit <- function(data, tuning_params, testData){
  pcr.fit <- pcr(damt~., data = data, scale = FALSE, ncomp = strtoi(tuning_params[[1]]))
  return(pcr.fit)
}

prediction.pc.reg <- function(pc.reg.fit, test_data, tuning_params){
  pcr.pred = predict(pc.reg.fit, test_data, ncomp = strtoi(tuning_params[[1]]) )
  return(pcr.pred)
}

#Re-Sampling Functions----------------

calculate_max_profits = function(post.prob.data, Y, model.fit){
  profit.prob <- cumsum(14.5*Y[order(post.prob.data, decreasing=T)]-2)
  #plot(profit.lda1) # see how profits change as more mailings are made
  n.mail.valid <- which.max(profit.prob) # number of mailings that maximizes profits
  post.prob.data <- as.data.frame(post.prob.data)
  if(n.mail.valid + 1 > length(profit.prob)){
    cutoff <- sort(post.prob.data[,1], decreasing=T)[n.mail.valid] # set cutoff based on n.mail.valid
  }else{
    cutoff <- sort(post.prob.data[,1], decreasing=T)[n.mail.valid+1] # set cutoff based on n.mail.valid  
  }
  chat <- ifelse(post.prob.data>cutoff, 1, 0) # mail to everyone above the cutoff
  confusion.matrix = table(chat, Y, useNA = "always")
  #               c.valid
  #chat.valid.lda1   0   1
  #              0 675  14
  #              1 344 985
  # check n.mail.valid = 344+985 = 1329
  # check profit = 14.5*985-2*1329 = 11624.5
  return(14.5*confusion.matrix[2,2] - 2*(confusion.matrix[1,2]+confusion.matrix[2,2]))
}

calculate_mean_prediction_error <- function(yhat, y, model.fit){
  mean_prediction_error <- mean((y - yhat)^2)
  return(mean_prediction_error)
}

get_gam_AIC <- function(yhat, y, gam.fit){
  return(gam.fit$aic)
}

get_lin_model_aic <- function(yhat, y, lin.reg.fit){
  aic <- extractAIC(lin.reg.fit)[2]
  return(aic)
}

get_lin_model_vif <- function(yhat, y, lin.reg.fit){
  sqrt_vif <- sqrt(vif(linear.fit))
  return(sqrt_vif[which.max(sqrt_vif)][[1]])
}









k_fold_cross_validations = function(data, k = 10, fit.function, prediction.function, performance.function, prediction.var, tuning_params){
  set.seed(1)
  #Randomly shuffle the data
  data<-data[sample(nrow(data)),]
  
  #Create k equally size folds
  folds <- cut(seq(1,nrow(data)),breaks=k,labels=FALSE)
  #Create a list to collect performance accross folds.
  performance = list()
  for(i in 1:k){
    #Segement the data by fold
    testIndexes <- which(folds==i,arr.ind=TRUE)
    testData <- data[testIndexes, ]
    trainData <- data[-testIndexes, ]
    y <- testData[[prediction.var]]
    model.fit <- fit.function(trainData, tuning_params, testData)
    yhat <- prediction.function(model.fit, testData, tuning_params)
    if(length(yhat)!=length(y)){
      print('Different Lengths')
    }
    performance[i] <- performance.function(yhat, y, model.fit)
  }
  average_performance = mean(unlist(performance))
  return(average_performance)
}

################### MODEL EVALUTAION ######################

#Model 1: Decision Trees------------------------------------
data_training_c_1 <- data.train.std.c
fit.function <- c('decision.tree.fit1')
prune = c(2,3,4,5, 6, 7, 8, 9)
#prune = seq(1,1,1)
xval_profits = c(rep(0, length(prune)))
model_tuning <- data.frame(fit.function, prune, xval_profits, stringsAsFactors = FALSE)
get_xval_profits <- function(x){
  xvalmetric <- k_fold_cross_validations(data_training_c_1, 10, decision.tree.fit, prediction.decision.tree, calculate_max_profits, 'donr', c(x['fit.function'], x['prune']))
  return(xvalmetric)
}

# model_tuning['xval_profits'] <- apply(model_tuning, 1, get_xval_profits)

# fit.function prune xval_profits
# 1 decision.tree.fit1     2      1342.45
# 2 decision.tree.fit1     3      2045.70
# 3 decision.tree.fit1     4      2154.45
# 4 decision.tree.fit1     5      2147.20
# 5 decision.tree.fit1     6      2147.20
# 6 decision.tree.fit1     7      2174.75
# 7 decision.tree.fit1     8      2202.30
# 8 decision.tree.fit1     9      2270.45 Best Model


# plot(model_tuning[,'prune'], model_tuning[,'xval_profits']) #prune = 9 is best
# 
model1 <- decision.tree.fit(data_training_c_1, c('decision.tree.fit1', 9))
plot(model1)
text(model1 ,pretty=0)
# prediction <- prediction.decision.tree(model1, data.valid.std.c)

#Model 2: Random forest-----------------------------------------------------------
training_data_2 <- data.train.std.c
m_try = c(seq(5,10))
fit.function <- c('bagging.tree.fit1')
xval_profits = c(rep(0, length(m_try)))
model_tuning <- data.frame(fit.function, m_try, xval_profits, stringsAsFactors = FALSE)
get_xval_profits <- function(x){
  xvalmetric <- k_fold_cross_validations(training_data_2, 10, bagging.tree.fit, prediction.bagging.tree, calculate_max_profits, 'donr', c(x['fit.function'], x['m_try']))
}
# model_tuning['xval_profits'] <- apply(model_tuning, 1, get_xval_profits)

# fit.function m_try xval_profits
# 1 bagging.tree.fit1     5      2200.85
# 2 bagging.tree.fit1     6      2197.95
# 3 bagging.tree.fit1     7      2202.30 Best Model
# 4 bagging.tree.fit1     8      2193.60
# 5 bagging.tree.fit1     9      2179.10
# 6 bagging.tree.fit1    10      2186.35



# model2 <- bagging.tree.fit(training_data_2, c('bagging.tree.fit1', 7))
# prediction <- prediction.bagging.tree(model2, data.valid.std.c)
# plot(model2)
# importance(model2)
# varImpPlot(model2)
# 
# prediction <- predict(model2, data.valid.std.c)
# table(prediction_class, data.valid.std.c$donr)
# mean(prediction_class == data.valid.std.c$donr) # 0.8865213

#Model 3: Boosting-----------------------------------------------------------
fit.function <- c('boosting.tree.fit1')
response_distribution = c("bernoulli")
num_trees = c(5000)
depth_of_trees = c(2,3,4)
shrinkage = c(0.001, 0.001, 0.001, 0.2, 0.2, 0.2)
xval_profits = c(rep(0, length(num_trees)))
model_tuning <- data.frame(fit.function, response_distribution, num_trees, depth_of_trees, shrinkage, xval_profits, stringsAsFactors = FALSE)

get_xval_profits <- function(x){
  xvalmetric <- k_fold_cross_validations(data.train.std.c, 10, boosting.tree.fit, prediction.boosting.tree, calculate_max_profits, 'donr', c(x['fit.function'], x['response_distribution'], x['num_trees'], x['depth_of_trees'], x['shrinkage']))
}

model_tuning['xval_profits'] <- apply(model_tuning, 1, get_xval_profits)
# fit.function response_distribution num_trees depth_of_trees shrinkage xval_profits
# 1 boosting.tree.fit1             bernoulli      5000              2     0.001      2466.20
# 2 boosting.tree.fit1             bernoulli      5000              3     0.001      2460.40
# 3 boosting.tree.fit1             bernoulli      5000              4     0.001      2469.10
# 4 boosting.tree.fit1             bernoulli      5000              2     0.200      2461.85
# 5 boosting.tree.fit1             bernoulli      5000              3     0.200      2463.30
# 6 boosting.tree.fit1             bernoulli      5000              4     0.200      2457.50
# model3 <- boosting.tree.fit(data.train.std.c, c('boosting.tree.fit1', 'bernoulli', 5000, 4, 0.001))
# prediction <- prediction.boosting.tree(model3, data.valid.std.c, tuning_params = NULL)

#Model 4: LDA-----------------------------------------------------------
data_training_c_4 <- data.train.std.c
fit.functions <- c('lda.fit1','lda.fit2', 'lda.fit3')
#threshold = seq(0.1, 0.9, 0.1)
xval_profits = c(rep(0, length(fit.functions)))
model_tuning <- data.frame(fit.functions,xval_profits, stringsAsFactors = FALSE)

get_xval_profits <- function(x){
  xvalmetric <- k_fold_cross_validations(data.train.std.c, 10, lda.fit, prediction.lda.fit, calculate_max_profits, 'donr', c(x['fit.functions']))
}

# model_tuning['xval_profits'] <- apply(model_tuning, 1, get_xval_profits)

# fit.functions xval_profits
# 1      lda.fit1      2479.25
# 2      lda.fit2      2479.25 #Preferred model because it's not as complex as lda.fit1
# 3      lda.fit3      2470.55

model4 <- lda.fit(data_training_c_4, c('lda.fit2'))
summary(model4)
prediction <- prediction.lda.fit(model4, data.valid.std.c)
# table(prediction$class, data.valid.std.c$donr)
# mean(prediction$class == data.valid.std.c$donr) # 0.8250743

# 
# data(iris)
# ldaHmat(x=iris[,1:4], grouping=iris$Species)
# 
# hmat <- ldaHmat(x = data_training_c_4[,2:21], grouping=as.factor(data_training_c_4$donr))
# eleaps(hmat$mat,kmin=5,kmax=10,H=hmat$H,r=hmat$r,crit="ccr12")
# hmat$H

#Model 5: QDA-----------------------------------------------------------
threshold = seq(0.1, 0.9, 0.1)
xval_profits = c(rep(0, length(threshold)))
model_tuning <- data.frame(threshold, xval_profits)

data_training_c_5 <- data.train.std.c
fit.functions <- c('qda.fit1','qda.fit2', 'qda.fit3', 'qda.fit4', 'qda.fit5', 'qda.fit6', 'qda.fit7')
#threshold = seq(0.1, 0.9, 0.1)
xval_profits = c(rep(0, length(fit.functions)))
model_tuning <- data.frame(fit.functions,xval_profits, stringsAsFactors = FALSE)

get_xval_profits <- function(x){
  xvalmetric <- k_fold_cross_validations(data_training_c_5, 10, qda.fit, prediction.qda.fit, calculate_max_profits, 'donr', c(x['fit.functions']))
}

model_tuning['xval_profits'] <- apply(model_tuning, 1, get_xval_profits)
# 
# hmat <- ldaHmat(x = data_training_c_5[,2:21], grouping=as.factor(data_training_c_4$donr))
# eleaps(hmat$mat,kmin=5,kmax=10,H=hmat$H,r=hmat$r,crit="ccr12")
# hmat$H


plot(model_tuning[,'xval_profits'])

# fit.functions xval_profits
# 1      qda.fit1      2483.60
# 2      qda.fit2      2482.15
# 3      qda.fit3      2415.45
# 4      qda.fit4      2483.60 # Best model
# 5      qda.fit5      2483.60
# 6      qda.fit6      2483.60
# 7      qda.fit7      2483.60

# model5 <- qda.fit(data_training_c_5, c('qda.fit4'))
# prediction <- prediction.qda.fit(model5, data.valid.std.c)
# table(prediction$class, data.valid.std.c$donr)
# #     0    1
# # 0 1464  196
# # 1  525 1799
# mean(prediction$class == data.valid.std.c$donr) # 0.8190261

#Model 6: Logistic Regression-----------------------------------------------------------

# threshold = seq(0.1, 0.9, 0.1)
# xval_profits = c(rep(0, length(threshold)))
# model_tuning <- data.frame(threshold, xval_profits)
# training_data_6.1 <- data.train.std.c[,c('reg1', 'reg2', 'home', 'chld', 'wrat', 'incm', 'plow', 'npro', 'tgif', 'tdon', 'tlag', 'donr')]

training_data_6 <- data.train.std.c
fit.functions <- c('log.reg.fit1','log.reg.fit2', 'log.reg.fit3', 'log.reg.fit4', 'log.reg.fit5')
xval_profits = c(rep(0, length(fit.functions)))
model_tuning <- data.frame(fit.functions,xval_profits, stringsAsFactors = FALSE)

get_xval_profits <- function(x){
  xvalmetric <- k_fold_cross_validations(training_data_6, 10, log.reg.fit, prediction.log.reg, calculate_max_profits, 'donr', c(x['fit.functions']))
}

# model_tuning['xval_profits'] <- apply(model_tuning, 1, get_xval_profits)
# 
# model6 <- log.reg.fit(training_data_6, c('log.reg.fit2'))
# summary(model6)
# prediction <- prediction.log.reg(model6, data.valid.std.c)
# prediction_class <- ifelse(prediction>0.5,1,0)
# table(prediction_class, data.valid.std.c$donr)
# mean(prediction_class == data.valid.std.c$donr) # 0.8359762

# fit.functions xval_profits
# 1  log.reg.fit1      2454.60
# 2  log.reg.fit2      2463.30 Best Model
# 3  log.reg.fit3      2461.85
# 4  log.reg.fit4      2457.50
# 5  log.reg.fit5      2456.05

# logrfit <- glm(donr ~ . -root_10_lgif -log_incm -log_rgif -log_tgif -log_agif -root_3_plow, data = training_data_6, family = binomial)
# Hmat <- glmHmat(logrfit)
# Hmat$mat
# eleaps(Hmat$mat, H=Hmat$H, r = Hmat$r, criterion = "Wald", nsol = 10)

#Model 7: K-Nearest Neighbors-----------------------------------------------------------
training_data_7 <- data.train.std.c[,c(1:20, ncol(data.train.std.c))] #no variable transformations
#training_data_7 <- data.train.std.c #contains all variable transformations
neighbors = seq(50,150)
xval_profits = c(rep(0, length(neighbors)))
model_tuning <- data.frame(neighbors, xval_profits)
get_xval_profits <- function(x){
  xvalmetric <- k_fold_cross_validations(training_data_7, 10, knn.fit, prediction.knn.fit, calculate_max_profits, 'donr', c(x['neighbors']))
}

model_tuning['xval_profits'] <- apply(model_tuning, 1, get_xval_profits)

# neighbors xval_profits
# 1          50      2411.10
# 2          51      2416.90
# 3          52      2412.55
# 4          53      2418.35
# 5          54      2427.05
# 6          55      2414.00
# 7          56      2411.10
# 8          57      2428.50
# 9          58      2424.15
# 10         59      2431.40
# 11         60      2429.95
# 12         61      2421.25
# 13         62      2422.70
# 14         63      2432.85
# 15         64      2412.55
# 16         65      2421.25
# 17         66      2427.05
# 18         67      2421.25
# 19         68      2424.15
# 20         69      2421.25
# 21         70      2415.45
# 22         71      2415.45
# 23         72      2425.60
# 24         73      2424.15
# 25         74      2425.60
# 26         75      2421.25
# 27         76      2427.05
# 28         77      2427.05
# 29         78      2429.95
# 30         79      2427.05
# 31         80      2418.35
# 32         81      2435.75
# 33         82      2432.85
# 34         83      2434.30
# 35         84      2434.30
# 36         85      2432.85
# 37         86      2438.65
# 38         87      2438.65
# 39         88      2434.30
# 40         89      2444.45
# 41         90      2440.10
# 42         91      2429.95
# 43         92      2429.95
# 44         93      2431.40
# 45         94      2437.20
# 46         95      2427.05
# 47         96      2428.50
# 48         97      2437.20
# 49         98      2428.50
# 50         99      2437.20
# 51        100      2435.75
# 52        101      2444.45
# 53        102      2443.00
# 54        103      2437.20
# 55        104      2443.00
# 56        105      2431.40
# 57        106      2435.75
# 58        107      2428.50
# 59        108      2431.40
# 60        109      2440.10
# 61        110      2437.20
# 62        111      2434.30
# 63        112      2444.45
# 64        113      2441.55
# 65        114      2444.45
# 66        115      2448.80
# 67        116      2444.45
# 68        117      2444.45
# 69        118      2450.25
# 70        119      2451.70
# 71        120      2448.80
# 72        121      2437.20
# 73        122      2451.70
# 74        123      2444.45
# 75        124      2435.75
# 76        125      2435.75
# 77        126      2451.70
# 78        127      2458.95 Best Model
# 79        128      2445.90
# 80        129      2440.10
# 81        130      2451.70
# 82        131      2454.60
# 83        132      2444.45
# 84        133      2440.10
# 85        134      2443.00
# 86        135      2453.15
# 87        136      2445.90
# 88        137      2453.15
# 89        138      2457.50
# 90        139      2456.05
# 91        140      2458.95
# 92        141      2451.70
# 93        142      2445.90
# 94        143      2458.95
# 95        144      2454.60
# 96        145      2456.05
# 97        146      2453.15
# 98        147      2450.25
# 99        148      2444.45
# 100       149      2444.45
# 101       150      2445.90

# plot(model_tuning[,'neighbors'], model_tuning[,'xval_profits'], xlab = "Neighbors", ylab = "Cross-Validation Calculated Profits", main = "K-Nearest Neighbors Cross Validation Resutls", type = "l")
# points(127,model_tuning[78,'xval_profits'], col = "red", cex = 1.5)
#Fit the model using given all the training data and the optimal number of neighbors found
#using cross validation
# set.seed(1)
# training_data_7 <- na.omit(data.valid.std.c)
# training_data_7<-training_data_7[sample(nrow(training_data_7)),]
# training_data_7.subsets <- split(training_data_7, sample((rep(1:2, nrow(training_data_7)))))
# length_subsets <- c(nrow(training_data_7.subsets$`1`), nrow(training_data_7.subsets$`2`))
# min_length <- length_subsets[which.min(length_subsets)]
# training_data_7.train <- training_data_7.subsets$`1`[1:min_length,]
# training_data_7.test <- training_data_7.subsets$`2`[1:min_length,]
# 
# model7 <- knn.fit(train_data = training_data_7.train, test_data = training_data_7.test, tuning_params = c(10))
# prediction_class <- ifelse(model7 > 0.5, 1, 0)
# confusion.matrix <- table(prediction_class, training_data_7.test$donr)
# mean(prediction_class == training_data_7.test$donr) #0.7425641

# --------------------PREDICTION MODELS --------------------

# Model 8: Regression Tree-------------------
training_data_8 <- data.train.std.y
fit.functions <- c('regression.tree.fit2', 'regression.tree.fit2')
xval_errors = c(rep(0, length(fit.functions)))
model_tuning <- data.frame(fit.functions,xval_errors, stringsAsFactors = FALSE)

get_xval_errors <- function(x){
  xvalmetric <- k_fold_cross_validations(training_data_8, 10, regression.tree.fit, prediction.regression.tree.fit, calculate_mean_prediction_error, 'damt', c(x['fit.functions']))
}

#model_tuning['xval_errors'] <- apply(model_tuning, 1, get_xval_errors)

# fit.functions xval_errors
# 1 regression.tree.fit1    2.044899
# 2 regression.tree.fit2    2.044899

# model8 <- regression.tree.fit(training_data_8, c('regression.tree.fit1'))
# 
# plot(model8, main = "Model 8")
# text(model8, pretty = 0)
# summary(model8)
# yhat <- predict(model8, data.valid.std.y)
# error <- calculate_mean_prediction_error(data.valid.std.y$damt, yhat) #2.264245
# 
# summary(model8)
# plot(model8)
# text(model8 ,pretty=0)
# yhat = predict(model8, newdata = data.valid.std.y)
# residuals = yhat - data.valid.std.y[,'damt']
# plot(yhat,residuals)
# plot(yhat, data.valid.std.y[,'damt'])

# Model 9: Ridge Regression------------
training_data_9 <- data.train.std.y
models = c('model9', 'model9.1')
ridge.reg.fit.functions = c('ridge.reg.fit1', 'ridge.reg.fit2')
lambda.vals = c('0.2207023', '0.126294')
xval_errors = c(rep(0, length(models)))
model_tuning <- data.frame(models, ridge.reg.fit.functions, lambda.vals, xval_errors)
get_xval_errors <- function(x){
  xvalmetric <- k_fold_cross_validations(training_data_9, 10, ridge.reg.fit, prediction.ridge.reg, calculate_mean_prediction_error, 'damt', c(x['ridge.reg.fit.functions'], x['lambda.vals']))
}

model_tuning['xval_errors'] <- apply(model_tuning, 1, get_xval_errors)
model9 <- ridge.reg.fit(training_data_9, c('ridge.reg.fit2', 0.123802))

hist(training_data_9$dam, main = "Distribution of damt variable", xlab = "Damt")

# models ridge.reg.fit.functions lambda.vals xval_errors
# 1   model9          ridge.reg.fit1   0.2207023    1.386301
# 2 model9.1          ridge.reg.fit2    0.126294    1.377597 Best Model

# set.seed(1)
# X <- model.matrix(damt~reg1 + reg2 + reg3 + reg4 + home + chld + hinc + genf + wrat + avhv + incm + inca + plow + npro + tgif + lgif + rgif + tdon + tlag + agif + root_10_lgif + log_root_3_lgif + log_incm + log_inca + log_rgif + log_tgif + log_tlag + log_agif + root_3_plow, data = data.train.std.y)
# model9 = glmnet(X,data.train.std.y[,'damt'], family = "poisson", alpha = 0, lambda = 0.2422205)#0.126294
# 
# model9 = cv.glmnet(X,data.train.std.y[,'damt'], family = "poisson", type.measure = "mse", alpha = 0)
# plot(model9)
# opt.lam9 = c(model9$lambda.min, model9$lambda.1se)
# coef(model9, s = opt.lam9)
# 
# newX <- model.matrix(damt~reg1 + reg2 + reg3 + reg4 + home + chld + hinc + genf + wrat + avhv + incm + inca + plow + npro + tgif + lgif + rgif + tdon + tlag + agif + root_10_lgif + log_root_3_lgif + log_incm + log_inca + log_rgif + log_tgif + log_tlag + log_agif + root_3_plow,data=data.valid.std.y)
# yhat <- predict(model9, newx = newX, type = "response", s = 0.2422205)
# ##yhat <- round(yhat, digits = 0) rounding to nearest integer
# mse <- mean((data.valid.std.y[,'damt'] - yhat)^2)
# plot(yhat,data.valid.std.y[,'damt'])
# residuals <- data.valid.std.y[,'damt'] - yhat
# plot(yhat, residuals)


# set.seed(1)
# model9.1=cv.glmnet(X,data.train.std.y[,'damt'],alpha=0)
# plot(model9.1)
# opt.lam9.1 = c(model9.1$lambda.min, 0.123802)
# coef(model9.1, s = opt.lam9.1)
# 
# newX <- model.matrix(~.-damt,data=data.valid.std.y)
# yhat <- predict(model9.1, newx = newX, type = "response", s = model9.1$lambda.min)
# mse <- mean((data.valid.std.y[,'damt'] - yhat)^2)
# plot(yhat,data.valid.std.y[,'damt'])
# plot(yhat - data.valid.std.y[,'damt'], yhat)

# Model 10: The Lasso---------------------
training_data_10 <- data.train.std.y
models = c('model10', 'model10.1')
lasso.fit.functions = c('lasso.fit1', 'lasso.fit1')
#For each model, the first lambda is the minimum lambda and the second is a lambda within one standard deviation.
lambda.vals = c('0.007571124', '0.1124288')
xval_errors = c(rep(0, length(models)))
model_tuning <- data.frame(models, lasso.fit.functions, lambda.vals, xval_errors)
get_xval_errors <- function(x){
  xvalmetric <- k_fold_cross_validations(training_data_10, 10, lasso.fit, prediction.lasso, calculate_mean_prediction_error, 'damt', c(x['lasso.fit.functions'], x['lambda.vals']))
}

model_tuning['xval_errors'] <- apply(model_tuning, 1, get_xval_errors)

# models lasso.fit.functions lambda.vals xval_errors
# 1   model10          lasso.fit1 0.007571124    1.369195 Best Model
# 2 model10.1          lasso.fit1   0.1124288    1.518502



# set.seed(1)
# X <- model.matrix(damt~., data = data.train.std.y)
# model10 = cv.glmnet(X,data.train.std.y[,'damt'], type.measure = "mse", alpha = 1)
# plot(model10)
# opt.lam10 = c(model10$lambda.min, model10$lambda.1se)
# coef(model10, s = opt.lam10)
# 
# model10 = glmnet(X,data.train.std.y[,'damt'], family = "poisson", alpha = 1, lambda = 0.007571124)
# newX <- model.matrix(~.-damt,data=data.valid.std.y)
# yhat <- predict(model10, newx = newX, type = "response", s = 0.007571124)
# mse <- mean((data.valid.std.y[,'damt'] - yhat)^2) #1.545435
# residuals <- data.valid.std.y[,'damt'] - yhat
# plot(yhat, residuals)
# plot(yhat,data.valid.std.y[,'damt'])
# 
# set.seed(1)
# model10.1=cv.glmnet(X,data.train.std.y[,'damt'],alpha=1)
# plot(model10.1)
# opt.lam10.1 = c(model10.1$lambda.min, model10.1$lambda.1se)
# coef(model10.1, s = opt.lam10.1)
# 
# model10.2 = glmnet(X,data.train.std.y[,'damt'], alpha = 1, lambda = 0.006762406)
# newX <- model.matrix(~.-damt,data=data.valid.std.y)
# yhat <- predict(model10.2, newx = newX, type = "response", s = 0.006762406)
# mse <- mean((data.valid.std.y[,'damt'] - yhat)^2) #1.554611
# residuals <- data.valid.std.y[,'damt'] - yhat
# plot(yhat, residuals)
# plot(yhat,data.valid.std.y[,'damt'])
# 
# model10.3 = glmnet(X,data.train.std.y[,'damt'], alpha = 1, lambda = 0.0833701)
# newX <- model.matrix(~.-damt,data=data.valid.std.y)
# yhat <- predict(model10.3, newx = newX, type = "response", s = 0.0833701)
# mse <- mean((data.valid.std.y[,'damt'] - yhat)^2) #1.695292
# residuals <- data.valid.std.y[,'damt'] - yhat
# plot(yhat, residuals)
# plot(yhat,data.valid.std.y[,'damt'])

# Model 11: Generalized Additive Models--------------
training_data11 <- data.train.std.y
model11 <- gam.fit.models(training_data11,'gam.fit',NULL)
model11.1 <- gam.fit.models(training_data11,'gam.fit.1',NULL)
model11.2 <- gam.fit.models(training_data11,'gam.fit.2',NULL)
model11.3 <- gam.fit.models(training_data11,'gam.fit.3',NULL)
model11.4 <- gam.fit.models(training_data11,'gam.fit.4',NULL)
model11.5 <- gam.fit.models(training_data11,'gam.fit.5',NULL)
model11.6 <- gam.fit.models(training_data11,'gam.fit.6',NULL)
model11.7 <- gam.fit.models(training_data11,'gam.fit.7',NULL)
model11.8 <- gam.fit.models(training_data11,'gam.fit.8',NULL)

summary(model11.8)

# anova(model11.6, model11, model11.1, model11.2, model11.3, model11.4, model11.5, test = "F") #Best GAM Model: model11.6

models = c('model11', 'model11.1', 'model11.2', 'model11.3', 'model11.4', 'model11.5', 'model11.6', 'model11.7', 'model11.8')
gam.fit.functions = c('gam.fit', 'gam.fit.1', 'gam.fit.2', 'gam.fit.3', 'gam.fit.4', 'gam.fit.5', 'gam.fit.6', 'gam.fit.7', 'gam.fit.8')
xval_errors = c(rep(0, length(models)))
xval_aic = c(rep(0, length(models)))
model_tuning <- data.frame(models, gam.fit.functions, xval_errors, xval_aic)
get_xval_errors <- function(x){
  xvalmetric <- k_fold_cross_validations(training_data11, 10, gam.fit.models, prediction.gam, calculate_mean_prediction_error, 'damt', c(x['gam.fit.functions']))
}
get_xval_aic <- function(x){
  xvalmetric <- k_fold_cross_validations(training_data11, 10, gam.fit.models, prediction.gam, get_gam_AIC, 'damt', c(x['gam.fit.functions']))
}

# model_tuning['xval_errors'] <- apply(model_tuning, 1, get_xval_errors)
# model_tuning['xval_aic'] <- apply(model_tuning, 1, get_xval_aic)

#      models gam.fit.functions xval_errors xval_aic
# 1   model11           gam.fit    2.358014 6636.708
# 2 model11.1         gam.fit.1    2.390891 6662.187
# 3 model11.2         gam.fit.2    1.988786 6326.889
# 4 model11.3         gam.fit.3    1.519128 5844.453
# 5 model11.4         gam.fit.4    1.514052 5839.522
# 6 model11.5         gam.fit.5    1.512961 5837.950
# 7 model11.6         gam.fit.6    1.513287 5836.696
# 8 model11.7         gam.fit.7    1.405116 5709.189
# 9 model11.8         gam.fit.8    1.324305 5593.727 # Best Model

# model_tuning <- model_tuning[with(model_tuning, order(xval_errors)),]
# plot(model_tuning[,'xval_errors'], type = 'l', col = 'blue')
# plot(model_tuning[,'xval_aic'], type = 'l', col = 'green')
# 
# model11.8 <- gam(damt ~ s(lgif, 10.67876) + s(incm, 12.09151) +  
#                    s(tgif, 5.120689) + s(wrat,2.562594) + s(plow, 4.430904) + s(rgif, 6.949567) + 
#                    s(tdon, 2.785614) + s(hinc, 2.785614) + s(tlag, 2.000007) + s(inca, 9.17457) +
#                    s(agif,7.813879)  + reg1 + reg2 + reg3 + reg4 + home + chld + genf, data = training_data11)
# 
# fit <- smooth.spline(training_data11$log_agif, training_data11$damt, cv = TRUE)
# fit$df #2.785614
# plot(training_data11$log_agif,training_data11$damt,xlim= range(training_data11$log_agif),col="darkgrey", xlab = "log_agif", ylab = "damt")
# lines(fit,col="red",lwd=2)
# title (" Transformation ")
# mean((predict(fit, training_data11)$y-training_data11$damt)^(2))

--

# par(mfrow=c(4,1))
# plot(model11, se = TRUE, col = "blue")
# plot(data.train.std.y$damt, data.train.std.y$root_10_lgif, add = TRUE)
#pairs(training_data11[,c('damt','incm','npro', 'tgif', 'tdon', 'agif','home')])
###lgif
# plot(training_data11$root_10_lgif, training_data11$damt)
# hist(training_data11$root_10_lgif)
# fit=smooth.spline(training_data11$root_10_lgif,training_data11$damt,cv = TRUE)
# fit$df #7.378126
# 
# plot(training_data11$root_10_lgif,training_data11$damt,xlim= range(training_data11$root_10_lgif),col="darkgrey")
# lines(fit,col="red",lwd=2)
# title (" Smoothing Spline ")
####incm
# plot(training_data11$log_incm, training_data11$damt)
# hist(training_data11$log_incm)
# fit=smooth.spline(training_data11$log_incm,training_data11$damt,cv = TRUE)
# fit$df #4.186313
# 
# plot(training_data11$log_incm,training_data11$damt,xlim= range(training_data11$log_incm),col="darkgrey")
# lines(fit,col="red",lwd=2)
# title (" Smoothing Spline ")

# plot(training_data11$avhv, training_data11$damt)
# hist(training_data11$avhv)
# fit=smooth.spline(training_data11$avhv,training_data11$damt,cv = TRUE)
# fit$df #4.186313
# 
# plot(training_data11$avhv,training_data11$damt,xlim= range(training_data11$avhv),col="darkgrey")
# lines(fit,col="red",lwd=2)
# title (" Smoothing Spline ")

# 
# plot(jitter(training_data11$log_rgif), jitter(training_data11$damt))
# hist(training_data11$log_rgif)
# boxplot(training_data11$damt~training_data11$log_rgif)
# fit=smooth.spline(training_data11$log_rgif,training_data11$damt,cv = TRUE)
# fit$df #2.001117
# 
# plot(training_data11$log_rgif,training_data11$damt,xlim= range(training_data11$log_rgif),col="darkgrey")
# lines(fit,col="red",lwd=2)
# title (" Smoothing Spline ")

# plot(jitter(training_data11$log_tgif), jitter(training_data11$damt))
# hist(training_data11$log_tgif)
# fit=smooth.spline(training_data11$log_tgif,training_data11$damt,cv = TRUE)
# fit$df #3.713441
# 
# plot(training_data11$log_tgif,training_data11$damt,xlim= range(training_data11$log_tgif),col="darkgrey")
# lines(fit,col="red",lwd=2)
# title (" Smoothing Spline ")

# fit=smooth.spline(training_data11$log_agif,training_data11$damt,cv = TRUE)
# fit$df #2.826113
# 
# plot(training_data11$log_agif,training_data11$damt,xlim= range(training_data11$log_agif),col="darkgrey")
# lines(fit,col="red",lwd=2)
# title (" Smoothing Spline ")

# fit <- smooth.spline(training_data11$log_root_3_lgif, training_data11$damt, cv = TRUE)
# fit$df #2.700608
# plot(training_data11$log_root_3_lgif,training_data11$damt,xlim= range(training_data11$log_root_3_lgif),col="darkgrey")
# lines(fit,col="red",lwd=2)
# title (" Smoothing Spline ")
# mean((predict(fit, training_data11)$y-training_data11$damt)^(2))

# fit <- smooth.spline(training_data11$wrat, training_data11$damt, cv = TRUE)
# fit$df #2.562594
# plot(training_data11$wrat,training_data11$damt,xlim= range(training_data11$wrat),col="darkgrey")
# lines(fit,col="red",lwd=2)
# title (" Smoothing Spline ")
# mean((predict(fit, training_data11)$y-training_data11$damt)^(2))

# fit <- smooth.spline(training_data11$plow, training_data11$damt, cv = TRUE)
# fit$df #4.430904
# plot(training_data11$plow,training_data11$damt,xlim= range(training_data11$plow),col="darkgrey")
# lines(fit,col="red",lwd=2)
# title (" Smoothing Spline ")
# mean((predict(fit, training_data11)$y-training_data11$damt)^(2))

# fit <- smooth.spline(training_data11$log_rgif, training_data11$damt, cv = TRUE)
# fit$df #2.001117
# plot(training_data11$log_rgif,training_data11$damt,xlim= range(training_data11$log_rgif),col="darkgrey")
# lines(fit,col="red",lwd=2)
# title (" Smoothing Spline ")
# mean((predict(fit, training_data11)$y-training_data11$damt)^(2))

# fit <- smooth.spline(training_data11$log_inca, training_data11$damt, cv = TRUE)
# fit$df #2.785614
# plot(training_data11$log_inca,training_data11$damt,xlim= range(training_data11$log_inca),col="darkgrey")
# lines(fit,col="red",lwd=2)
# title (" Smoothing Spline ")
# mean((predict(fit, training_data11)$y-training_data11$damt)^(2))

# Model 12: Linear Regression-----------------------------
training_data12 <- data.train.std.y

# Hmat <- lmHmat(x = training_data12[,1:29], y = training_data12[,30])
# solutions <- eleaps(Hmat$mat, H=Hmat$H, r = Hmat$r, criterion = "Wald", nsol = 10)
# solutions$bestsets
# cardinality <- 18
# measurevar <- 'damt'
# predictors <- c(names(training_data12)[solutions$bestsets[cardinality,1:cardinality]])
# string_formula <- paste(measurevar, paste(predictors, collapse=" + "), sep=" ~ ")
# linear.fit <- lm(as.formula(string_formula), data = training_data12)
# summary(linear.fit)


model12.14 <- lin.reg.models(training_data12, c('lin.reg.fit.3', 10))
summary(model12.14)
extractAIC(model12.14)[2]

models = c('model12.1', 'model12.2', 'model12.3', 'model12.4', 'model12.5', 'model12.6', 'model12.7', 'model12.8', 'model12.9', 'model12.10', 'model12.11', 'model12.12', 'model12.13', 'model12.14', 'model12.15', 'model12.16')
lin.reg.functions = c('lin.reg.fit1', 'lin.reg.fit1', 'lin.reg.fit1', 'lin.reg.fit2', 'lin.reg.fit2', 'lin.reg.fit2', 'lin.reg.fit3', 'lin.reg.fit3', 'lin.reg.fit3', 'lin.reg.fit3', 'lin.reg.fit3', 'lin.reg.fit3', 'lin.reg.fit3', 'lin.reg.fit3', 'lin.reg.fit4', 'lin.reg.fit5')
tuning_params = c('forward', 'backward', 'both', 'forward', 'backward', 'both', 8,9,10,11,12,13,14,15,0,0)

length(lin.reg.functions)
xval_errors = c(rep(0, length(models)))
xval_aic = c(rep(0, length(models)))
xval_vif = c(rep(0, length(models)))
model_tuning <- data.frame(models, lin.reg.functions, tuning_params, xval_errors, xval_aic, xval_vif)
get_xval_errors <- function(x){
  xvalmetric <- k_fold_cross_validations(training_data12, 10, lin.reg.models, prediction.lin.reg, calculate_mean_prediction_error, 'damt', c(x['lin.reg.functions'], x['tuning_params']))
}
get_xval_aic <- function(x){
  xvalmetric <- k_fold_cross_validations(training_data12, 10, lin.reg.models, prediction.lin.reg, get_lin_model_aic, 'damt', c(x['lin.reg.functions'], x['tuning_params']))
}
get_xval_vif <- function(x){
  xvalmetric <- k_fold_cross_validations(training_data12, 10, lin.reg.models, prediction.lin.reg, get_lin_model_vif, 'damt', c(x['lin.reg.functions'], x['tuning_params']))
}

# model_tuning['xval_errors'] <- apply(model_tuning, 1, get_xval_errors)
# model_tuning['xval_aic'] <- apply(model_tuning, 1, get_xval_aic)
# #model_tuning['xval_vif'] <- apply(model_tuning, 1, get_xval_vif)
# model_tuning_save <- model_tuning
# model12.9 <- lin.reg.models(training_data12, c('lin.reg.fit3',10))
# summary(model12.9)
# extractAIC(model12.9)[2]
# sqrt(vif(model12.9))

# models lin.reg.functions tuning_params xval_errors  xval_aic xval_vif
# 1   model12.1      lin.reg.fit1       forward  143.376991 8327.3491 1.619975
# 2   model12.2      lin.reg.fit1      backward  143.381325 8296.6001 1.619975
# 3   model12.3      lin.reg.fit1          both  143.381325 8296.6001 1.619975
# 4   model12.4      lin.reg.fit2       forward    1.372432  565.5078 1.619975
# 5   model12.5      lin.reg.fit2      backward    1.371267  549.8216 1.619975
# 6   model12.6      lin.reg.fit2          both    1.372673  549.7746 1.619975
# 7   model12.7      lin.reg.fit3             8    1.496311  672.6379 1.619975
# 8   model12.8      lin.reg.fit3             9    1.453629  630.7700 1.619975
# 9   model12.9      lin.reg.fit3            10    1.392712  590.1913 1.619975 #Possibly best model, but there may be some collinearity issues.
# 10 model12.10      lin.reg.fit3            11    1.393049  574.4368 1.619975
# 11 model12.11      lin.reg.fit3            12    1.375857  563.7209 1.619975
# 12 model12.12      lin.reg.fit3            13    1.369321  556.4909 1.619975
# 13 model12.13      lin.reg.fit3            14    1.374198  553.0293 1.619975
# 14 model12.14      lin.reg.fit3            15    1.370206  550.9578 1.619975
# 15 model12.15      lin.reg.fit4             0    1.415675  618.2887 1.619975
# 16 model12.16      lin.reg.fit5             0    1.455443  668.6781 1.619975

# Model 13: PCA Regression---------------------------------

training_data13 <- data.train.std.y[,c(1:20,ncol(data.train.std.y))]
# library(Hmisc)
# rcorr(as.matrix(training_data13[,c('incm','inca','tgif','lgif','rgif','agif','tdon','tlag')]), type="pearson")
names(training_data13)
names(data.train.std.y)

models = c('model13', 'model13.1', 'model13.2', 'model13.3', 'model13.4', 'model13.5', 'model13.6', 'model13.7', 'model13.8', 'model13.9', 'model13.10', 'model13.11', 'model13.12', 'model13.13', 'model13.14', 'model13.15', 'model13.16', 'model13.17', 'model13.18', 'model13.19')
ncomp = c(seq(1,20))
xval_errors = c(rep(0, length(models)))
model_tuning <- data.frame(models, ncomp, xval_errors)
get_xval_errors <- function(x){
  xvalmetric <- k_fold_cross_validations(training_data13, 10, pc.reg.fit, prediction.pc.reg, calculate_mean_prediction_error, 'damt', c(x['ncomp']))
}

model_tuning['xval_errors'] <- apply(model_tuning, 1, get_xval_errors) #Best model: model13.6, compromises complexity and predictability.
plot(model_tuning$ncomp, model_tuning$xval_errors, type = "l", xlab = "Number of Components", ylab = "Cross Validation MSE", main = "Performance by Number of Components")

model13.13 <- pcr(damt~., data = training_data13, scale = FALSE, ncomp = 14)
plot(model13.13$fitted.values, model13.13$residuals)
#plot(model_tuning[,'ncomp'], model_tuning[,'xval_errors'], type = "l", xlab = "Number of components", ylab = "CV Error", main = "CV Error for each Number of Components")

#        models ncomp xval_errors
# 1     model13     1    3.750555
# 2   model13.1     2    2.713785
# 3   model13.2     3    2.681052
# 4   model13.3     4    2.597513
# 5   model13.4     5    2.182591
# 6   model13.5     6    2.137410
# 7   model13.6     7    2.086242
# 8   model13.7     8    2.048850
# 9   model13.8     9    2.035330
# 10  model13.9    10    2.011769
# 11 model13.10    11    1.858109
# 12 model13.11    12    1.862013
# 13 model13.12    13    1.751263
# 14 model13.13    14    1.689775 Preferred Model
# 15 model13.14    15    1.692004
# 16 model13.15    16    1.693280
# 17 model13.16    17    1.690344
# 18 model13.17    18    1.693866
# 19 model13.18    19    1.646251
# 20 model13.19    20    1.645559

# Model 14: Ensemble model----------------------
training_data_14 <- data.train.std.y

ensemble.models <- function(data, tuning_params, testData){
  ensemble.fit1 <- function(data, tuning_params, testData){
    #data <- training_data_14
    regression_tree <- regression.tree.fit(data, c('regression.tree.fit1'))
    regression_tree_prediction <- prediction.regression.tree.fit(regression_tree, data, NULL)
    
    ridge.fit <- ridge.reg.fit(data, c('ridge.reg.fit2', 0.123802))
    ridge.prediction <- prediction.ridge.reg(ridge.fit, data, c('ridge.reg.fit2', 0.123802))
    
    lassofit <- lasso.fit(data, c('lasso.fit2', 0.0833701))
    lasso.prediction <- prediction.lasso(lassofit, data, c('lasso.fit2', 0.0833701))
    
    gam.fit <- gam.fit.models(data, c('gam.fit.8'), testData)
    gam.prediction <- prediction.gam(gam.fit, data, c('gam.fit.8'))
    
    lin.fit <- lin.reg.models(data, c('lin.reg.fit3', 10), testData)
    lin.prediction <- prediction.lin.reg(lin.fit, data, c('lin.reg.fit3', 10))
    
    pcr.fit <- pc.reg.fit(data, c(14), testData)
    pcr.prediction <- prediction.pc.reg(pcr.fit, data, c(14))
    
    variables <- data.frame(regression_tree_prediction, ridge.prediction, lasso.prediction, gam.prediction, lin.prediction, pcr.prediction, data$damt)
    colnames(variables) <- c('regression_tree', 'ridge_regression', 'lasso', 'gam', 'linear_regression', 'pcr', 'damt')
    ensemble.fit1 <- lm(damt ~., data = variables)
    return(ensemble.fit1)
  }
  
  ensemble.fit2 <- function(data, tuning_params, testData){
    #data <- training_data_14
    regression_tree <- regression.tree.fit(data, c('regression.tree.fit1'))
    regression_tree_prediction <- prediction.regression.tree.fit(regression_tree, data, NULL)
    
    gam.fit <- gam.fit.models(data, c('gam.fit.8'), testData)
    gam.prediction <- prediction.gam(gam.fit, data, c('gam.fit.8'))
 
    pcr.fit <- pc.reg.fit(data, c(14), testData)
    pcr.prediction <- prediction.pc.reg(pcr.fit, data, c(14))
    
    variables <- data.frame(regression_tree_prediction, gam.prediction, pcr.prediction, data$damt)
    colnames(variables) <- c('regression_tree', 'gam', 'pcr', 'damt')
    ensemble.fit2 <- lm(damt ~., data = variables)
    return(ensemble.fit2)
  }
  
  ensemble.fit3 <- function(data, tuning_params, testData){
    #data <- training_data_14
    regression_tree <- regression.tree.fit(data, c('regression.tree.fit1'))
    regression_tree_prediction <- prediction.regression.tree.fit(regression_tree, data, NULL)
    
    gam.fit <- gam.fit.models(data, c('gam.fit.8'), testData)
    gam.prediction <- prediction.gam(gam.fit, data, c('gam.fit.8'))
    
    variables <- data.frame(regression_tree_prediction, gam.prediction, data$damt)
    colnames(variables) <- c('regression_tree', 'gam', 'damt')
    ensemble.fit2 <- lm(damt ~., data = variables)
    return(ensemble.fit3)
  }
  
  
  lin.reg.fit.fun <- get(tuning_params[[1]])
  lin.model.fit <- lin.reg.fit.fun(data,tuning_params, testData)
  return(lin.model.fit)
}

models = c('model14')
ensemble.fit.functions = c('ensemble.fit1')
xval_errors = c(rep(0, length(models)))
model_tuning <- data.frame(models, ensemble.fit.functions, xval_errors)
get_xval_errors <- function(x){
  xvalmetric <- k_fold_cross_validations(training_data11, 10, ensemble.models, prediction.lin.reg, calculate_mean_prediction_error, 'damt', c(x['ensemble.fit.functions']))
}
# get_xval_aic <- function(x){
#   xvalmetric <- k_fold_cross_validations(training_data11, 10, gam.fit.models, prediction.gam, get_gam_AIC, 'damt', c(x['gam.fit.functions']))
# }

#model_tuning['xval_errors'] <- apply(model_tuning, 1, get_xval_errors)
model14 <- ensemble.models(data.train.std.y, c('ensemble.fit3'), data.train.std.y)
plot(model14$fitted.values, model14$residuals)
summary(model14)
sqrt(vif(model14))
