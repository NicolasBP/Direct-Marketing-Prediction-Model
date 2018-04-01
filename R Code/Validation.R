library(boot)
source('Modeling.R')

do_bootstrap = function(data, statistic.function){
  index <- 1:nrow(data)
  statistic.function(data, index)
  boot(data, statistic.function, R = 1000)
}

#Statistic Functions----------------------

calculate_statistic_model1 <- function(data, index){
  data = na.omit(data)
  subset = data[index,]
  Y = subset[["donr"]]
  model1 <- decision.tree.fit(subset, c('decision.tree.fit1', 9))
  prediction <- prediction.decision.tree(model1, subset, NULL)
  return(calculate_max_profits(prediction, Y))
}

calculate_statistic_model2 <- function(data, index){
  data = na.omit(data)
  subset = data[index,]
  Y = subset[["donr"]]
  model2 <- bagging.tree.fit(subset, c('bagging.tree.fit1', 7))
  prediction <- prediction.bagging.tree(model2, subset, NULL)
  return(calculate_max_profits(prediction, Y))
}

calculate_statistic_model3 <- function(data, index){
  data = na.omit(data)
  subset = data[index,]
  Y = subset[["donr"]]
  model3 <- boosting.tree.fit(subset, c('boosting.tree.fit1', 'bernoulli', 5000, 4, 0.001))
  prediction <- prediction.boosting.tree(model3, subset, NULL)
  return(calculate_max_profits(prediction, Y))
}

calculate_statistic_model4 = function(data, index){
  data = na.omit(data)
  subset = data[index,]
  Y = subset[["donr"]]
  model4 <- lda.fit(subset, c('lda.fit2'))
  prediction <- predict(model4, subset)$posterior[,2] # n.valid.c post probs
  return(calculate_max_profits(prediction, Y))
}

calculate_statistic_model5 = function(data, index){
  data = na.omit(data)
  subset = data[index,]
  Y = subset[["donr"]]
  model5 <- qda.fit(subset, c('qda.fit4'))
  prediction <- predict(model5, subset)$posterior[,2] # n.valid.c post probs
  return(calculate_max_profits(prediction, Y))
}

calculate_statistic_model6 = function(data, index){
  data = na.omit(data)
  subset = data[index,]
  Y = subset[["donr"]]
  model6 <- log.reg.fit(subset, c('log.reg.fit2'))
  prediction <- predict(model6, subset, type="response")
  return(calculate_max_profits(prediction, Y))
  
}

calculate_statistic_model7 = function(data, index){
  data = na.omit(data)
  subset = data[index,]
  set.seed(1)
  subset <- split(subset, sample((rep(1:2, nrow(subset)))))
  length_subsets <- c(nrow(subset$`1`), nrow(subset$`2`))
  min_length <- length_subsets[which.min(length_subsets)]
  subset.train <- subset$`1`[1:min_length,]
  subset.test <- subset$`2`[1:min_length,]
  Y = subset.test[["donr"]]
  prediction <- knn.fit(train_data = subset.train, test_data = subset.test, tuning_params = c(127))
  return(calculate_max_profits(prediction, Y))
}

calculate_statistic_model8 = function(data, index){
  data = na.omit(data)
  subset = data[index,]
  Y = subset[["damt"]]
  model8 <- regression.tree.fit(data, c('regression.tree.fit'))
  prediction <- prediction.regression.tree.fit(model8, subset, c('regression.tree.fit1'))
  return(calculate_mean_prediction_error(prediction, Y, model8))
}

calculate_statistic_model9 = function(data, index){
  data = na.omit(data)
  subset = data[index,]
  Y = subset[["damt"]]
  model9 <- ridge.reg.fit(subset, c('ridge.reg.fit2', 0.126294))
  prediction <- prediction.ridge.reg(model9, subset, c('ridge.reg.fit2', 0.126294))
  return(calculate_mean_prediction_error(prediction, Y, model9))
}

calculate_statistic_model10 = function(data, index){
  data = na.omit(data)
  subset = data[index,]
  Y = subset[["damt"]]
  model10 <- lasso.fit(subset, c('lasso.fit1', 0.007571124))
  prediction <- prediction.lasso(model10, subset, c('lasso.fit1', 0.007571124))
  return(calculate_mean_prediction_error(prediction, Y, model10))
}

calculate_statistic_model11 = function(data, index){
  data = na.omit(data)
  subset = data[index,]
  Y = subset[["damt"]]
  model11 <- gam.fit.models(subset, c('gam.fit.8'), NULL)
  prediction <- prediction.gam(model11, subset, c('gam.fit.8'))
  return(calculate_mean_prediction_error(prediction, Y, model11))
}

calculate_statistic_model12 = function(data, index){
  data = na.omit(data)
  subset = data[index,]
  Y = subset[["damt"]]
  model12 <- lin.reg.models(subset, c('lin.reg.fit6'), NULL)
  prediction <- prediction.lin.reg(model12, subset, c('lin.reg.fit6'))
  return(calculate_mean_prediction_error(prediction, Y, model12))
}

calculate_statistic_model13 = function(data, index){
  data = na.omit(data)
  subset = data[index,]
  Y = subset[["damt"]]
  model13 <- pc.reg.fit(subset, c(14), NULL)
  prediction <- prediction.pc.reg(model12, subset, c(14))
  return(calculate_mean_prediction_error(prediction, Y, model13))
}

#Bootstrap Calls----------

#Model1: Decision Trees----------
# decision.tree.boot <- do_bootstrap(data.valid.std.c, calculate_statistic_model1)

# Call:
#   boot(data = data, statistic = statistic.function, R = 1000)
# 
# 
# Bootstrap Statistics :
#   original  bias    std. error
# t1*    11632 -14.002    510.8235

#Model2: Random Forest----------
random.forest.boot <- do_bootstrap(data.valid.std.c, calculate_statistic_model2)

# Call:
#   boot(data = data, statistic = statistic.function, R = 1000)
# 
# 
# Bootstrap Statistics :
#   original  bias    std. error
# t1*  12487.5     5.3    283.8391

#Model3: Bootsing-----------

# boosting.boot <- do_bootstrap(data.valid.std.c, calculate_statistic_model3)

# Call:
#   boot(data = data, statistic = statistic.function, R = 1000)
# 
# 
# Bootstrap Statistics :
#   original  bias    std. error
# t1*    12328 60.2895    292.1754

#Model4: LDA--------------
# lda.boot <- do_bootstrap(data.valid.std.c, calculate_statistic_model4)

# Call:
#   boot(data = data, statistic = statistic.function, R = 1000)
# 
# 
# Bootstrap Statistics :
#   original  bias    std. error
# t1*    12357 -35.736    314.0321

#Model5: QDA--------------
# qda.boot <- do_bootstrap(data.valid.std.c, calculate_statistic_model5)

# Call:
#   boot(data = data, statistic = statistic.function, R = 1000)
# 
# 
# Bootstrap Statistics :
#   original  bias    std. error
# t1*  11994.5 48.6895     297.165

#Model 6: Logistic Regression-----------

# logistic.regression <- do_bootstrap(data.valid.std.c, calculate_statistic_model6)

# Call:
#   boot(data = data, statistic = statistic.function, R = 1000)
# 
# 
# Bootstrap Statistics :
#   original  bias    std. error
# t1*  12226.5  26.685    310.7822

#Model 7: K-Nearest Neighbors-----------

knn.boot <- do_bootstrap(data.valid.std.c, calculate_statistic_model7)

# Call:
#   boot(data = data, statistic = statistic.function, R = 1000)
# 
# 
# Bootstrap Statistics :
#   original   bias    std. error
# t1*     6415 -344.541    223.4474


set.seed(123)
Model1 <- rnorm(1000, mean = 11632, sd = 510.8235 * sqrt(1000))
Model2 <- rnorm(1000, mean = 12487.5, sd = 283.8391 * sqrt(1000))
Model3 <- rnorm(1000, mean = 12328, sd = 292.1754 * sqrt(1000))
Model4 <- rnorm(1000, mean = 12357, sd = 314.0321 * sqrt(1000))
Model5 <- rnorm(1000, mean = 11994.5, sd = 297.165 * sqrt(1000))
Model6 <- rnorm(1000, mean = 12226.5, sd = 310.7822 * sqrt(1000))
# Model7 <- rnorm(1000, mean = 6415, sd = 223.4474 * sqrt(1000))
hist(Model2)
length(c(Model1, Model2, Model3, Model4, Model5, Model6, Model7))
anova_df <- data.frame(MSE = c(Model1, Model2, Model3, Model4, Model5, Model6), Group = c(rep("Model1", 1000), rep("Model2", 1000), rep("Model3", 1000), rep("Model4", 1000), rep("Model5", 1000), rep("Model6", 1000)))
names(anova_df)

anova <- aov(MSE ~ Group, data = anova_df)
summary(anova)
TukeyHSD(anova, conf.level = 0.95)

# --------------------PREDICTION MODELS --------------------

#Model 8: Regression Tree-----------

# reg.tree.boot <- do_bootstrap(data.valid.std.y, calculate_statistic_model8)
# Call:
#   boot(data = data, statistic = statistic.function, R = 1000)
# 
# 
# Bootstrap Statistics :
#   original      bias    std. error
# t1*  2.07776 -0.00710705   0.1548316

#Model 9: Ridge Regression-------------

# ridge.reg.boot <- do_bootstrap(data.valid.std.y, calculate_statistic_model9)
# Call:
#   boot(data = data, statistic = statistic.function, R = 1000)
# 
# 
# Bootstrap Statistics :
#   original      bias    std. error
# t1* 1.487828 -0.04405707   0.1510042

#Model 10: Lasso-----------------

# lasso.boot <- do_bootstrap(data.valid.std.y, calculate_statistic_model10)

# Call:
#   boot(data = data, statistic = statistic.function, R = 1000)
# 
# 
# Bootstrap Statistics :
#   original      bias    std. error
# t1* 1.474355 -0.04835028   0.1507193

#Model 11: Generalized Additive Models-----------------

# gam.boot <- do_bootstrap(data.valid.std.y, calculate_statistic_model11)

# Call:
#   boot(data = data, statistic = statistic.function, R = 1000)
# 
# 
# Bootstrap Statistics :
#   original      bias    std. error
# t1* 1.247932 -0.09445278   0.1162653

#Model 12: Linear Regression-----------------

# lin.reg.boot <- do_bootstrap(data.valid.std.y, calculate_statistic_model12)

# Call:
#   boot(data = data, statistic = statistic.function, R = 1000)
# 
# 
# Bootstrap Statistics :
#   original      bias    std. error
# t1* 1.459581 -0.04697706   0.1504346

#Model 13: PCA Regression-----------------

# pcar.boot <- do_bootstrap(data.valid.std.y, calculate_statistic_model13)

# Call:
#   boot(data = data, statistic = statistic.function, R = 1000)
# 
# 
# Bootstrap Statistics :
#   original    bias    std. error
# t1* 140.7616 -0.032573    1.616515

set.seed(123)
Model8 <- rnorm(1000, mean = 2.07776, sd = 0.1548316 * sqrt(1000))
Model9 <- rnorm(1000, mean = 1.487828, sd = 0.1510042 * sqrt(1000))
Model10 <- rnorm(1000, mean = 1.474355, sd = 0.1507193 * sqrt(1000))
Model11 <- rnorm(1000, mean = 1.247932, sd = 0.1162653 * sqrt(1000))
Model12 <- rnorm(1000, mean = 1.459581, sd = 0.1504346 * sqrt(1000))
# Model13 <- rnorm(1000, mean = 140.7616, sd = 1.616515 * sqrt(1000))
hist(Model11)
length(c(Model8, Model9, Model10, Model11, Model12, Model13))
anova_df <- data.frame(MSE = c(Model8, Model9, Model10, Model11, Model12), Group = c(rep("Model8", 1000), rep("Model9", 1000), rep("Model10", 1000), rep("Model11", 1000), rep("Model12", 1000)))
names(anova_df)

anova <- aov(MSE ~ Group, data = anova_df)
summary(anova)
TukeyHSD(anova, conf.level = 0.95)





