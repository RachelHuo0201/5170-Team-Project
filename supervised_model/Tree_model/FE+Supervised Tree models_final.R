library(readr)
library(dplyr)
library(caret)
library(rpart)
library(randomForest)
library(xgboost)
library(gains)
library(rpart.plot)
library(ggplot2)
library(corrplot)

# Read dataset, combine them and feature engineering

red <- read_delim("winequality-red.csv",
                  delim = ";", escape_double = FALSE, trim_ws = TRUE)
white <- read_delim("winequality-white.csv",
                    delim = ";", escape_double = FALSE, trim_ws = TRUE)
names(red) <- make.names(names(red))
names(white)  <- make.names(names(white))

red$wine_type_bin <- 0
white$wine_type_bin <- 1
wine_all <- bind_rows(red, white)
wine_all_nodup <- wine_all %>% distinct()

### heatmap
num_cols <- wine_all_nodup[, sapply(wine_all_nodup, is.numeric)]
corr_matrix <- cor(num_cols, use = "complete.obs")
corrplot(corr_matrix,
         method = "color",
         tl.col = "black",
         tl.srt = 45,
         addCoef.col = "black",
         number.cex = 0.7,
         col = colorRampPalette(c("green", "white", "red"))(200),
         title = "Correlation Heatmap of wine",
         mar = c(0,0,2,0))


wine_fe <- wine_all_nodup %>%
  select(-total.sulfur.dioxide, -density) %>%
  
  mutate(
    acidity_ratio = fixed.acidity / (volatile.acidity + citric.acid + 1e-6),
    alcohol_va_ratio = alcohol / (volatile.acidity + 1e-6),
    sugar_body_ratio = residual.sugar / (1/alcohol + 1e-6),
    sulfur_efficiency = free.sulfur.dioxide / (fixed.acidity + 1e-6)
  )

wine_FE <- wine_fe

wine_FE$label <- factor(ifelse(wine_FE$quality >= 6, "high", "low"),
                        levels = c("low", "high"))

wine_FE$quality <- NULL


# Split data

set.seed(42)
train_idx <- sample(1:nrow(wine_FE), 0.6 * nrow(wine_FE))
train <- wine_FE[train_idx, ]
test  <- wine_FE[-train_idx, ]


# Compute F1 and G-means

compute_metrics <- function(pred, truth) {
  cm <- confusionMatrix(pred, truth, positive = "high")
  acc <- as.numeric(cm$overall["Accuracy"])
  precision <- as.numeric(cm$byClass["Precision"])
  recall   <- as.numeric(cm$byClass["Recall"])
  F1  <- 2 * precision * recall / (precision + recall)
  G   <- sqrt(as.numeric(cm$byClass["Sensitivity"]) * as.numeric(cm$byClass["Specificity"]))
  c(Accuracy = acc, Precision = precision, Recall = recall, F1 = F1, Gmean = G)
}


# 5-fold CV of train

folds <- createFolds(train$label, k = 5, list = TRUE)
K <- length(folds)

res_dt  <- matrix(NA, K, 5); colnames(res_dt)  <- c("Accuracy","Precision","Recall","F1","Gmean")
res_rf  <- matrix(NA, K, 5); colnames(res_rf)  <- c("Accuracy","Precision","Recall","F1","Gmean")
res_xgb <- matrix(NA, K, 5); colnames(res_xgb) <- c("Accuracy","Precision","Recall","F1","Gmean")

# (1). CV of Decision Tree

for (i in 1:K) {
  val_idx    <- folds[[i]]
  train_data <- train[-val_idx, ]
  val_data   <- train[ val_idx, ]
  
  dt_model <- rpart(
    label ~ ., data = train_data, method = "class",
    parms  = list(split = "information"),
    control = rpart.control(minbucket = 50, maxdepth = 7, cp = 0.01)
  )
  rpart.plot(dt_model, extra = 106) 
  
  best_cp <- dt_model$cptable[which.min(dt_model$cptable[, "xerror"]), "CP"]
  dt_model <- prune(dt_model, cp = best_cp)
  
  pred_dt <- predict(dt_model, newdata = val_data, type = "class")
  res_dt[i, ] <- compute_metrics(pred_dt, val_data$label)
}


# (2). Random Forest

for (i in 1:K) {
  val_idx    <- folds[[i]]
  train_data <- train[-val_idx, ]
  val_data   <- train[ val_idx, ]
  
  rf_model <- randomForest(
    label ~ ., data = train_data,
    ntree = 500, mtry = floor(sqrt(ncol(train_data)-1))
  )
  
  pred_rf <- predict(rf_model, newdata = val_data)
  res_rf[i, ] <- compute_metrics(pred_rf, val_data$label)
}

# (3). XGBoost

for (i in 1:K) {
  val_idx    <- folds[[i]]
  train_data <- train[-val_idx, ]
  val_data   <- train[ val_idx, ]
  
  train_x <- model.matrix(label ~ ., train_data)[, -1]
  val_x   <- model.matrix(label ~ ., val_data)[, -1]
  train_y <- ifelse(train_data$label == "high", 1, 0)
  val_y   <- ifelse(val_data$label   == "high", 1, 0)
  
  dtrain <- xgb.DMatrix(train_x, label = train_y)
  dval   <- xgb.DMatrix(val_x,   label = val_y)
  
  params <- list(
    objective = "binary:logistic",
    max_depth = 4, 
    eta = 0.1,
    subsample = 0.8, 
    colsample_bytree = 0.8
  )
  
  xgb_model <- xgb.train(
    params = params, data = dtrain,
    nrounds = 100,
    watchlist = list(train = dtrain, val = dval),
    print_every_n = 10, early_stopping_rounds = 10
  )
  
  pred_prob   <- predict(xgb_model, val_x)
  pred_class  <- ifelse(pred_prob > 0.5, "high", "low")
  pred_factor <- factor(pred_class, levels = c("low","high"))
  
  res_xgb[i, ] <- compute_metrics(pred_factor, val_data$label)
}


# Summary

cv_summary <- rbind(
  DecisionTree = colMeans(res_dt),
  RandomForest = colMeans(res_rf),
  XGBoost      = colMeans(res_xgb)
)

cv_summary


# Test
## DT and Lift chart
dt_final <- rpart(
  label ~ ., data = train, method = "class",
  parms = list(split = "information"),
  control = rpart.control(minbucket = 50, maxdepth = 7, cp = 0.01)
)
rpart.plot(dt_final, extra = 106) 
best_cp <- dt_final$cptable[which.min(dt_final$cptable[, "xerror"]), "CP"]
dt_final <- prune(dt_final, cp = best_cp)

dt_pred_test <- predict(dt_final, newdata = test, type = "class")
dt_test_metrics <- compute_metrics(dt_pred_test, test$label)
dt_prob <- predict(dt_final, newdata = test, type = "prob")[, 2]

actual_dt <- as.numeric(test$label) - 1
g_dt <- gains(actual = actual_dt, predicted = dt_prob, groups = 10)
plot(g_dt,
     main  = "Lift Chart for Decision Tree (test set)",
     xlab  = "Depth of File (Deciles)",
     ylab  = "Mean Response",
     col   = c("red", "green", "skyblue"),
     lty   = c(1, 1, 1),
     legend = c("Mean Response",
                "Cumulative Mean Response",
                "Mean Predicted Response"))

## Random Forest test
rf_final <- randomForest(
  label ~ ., data = train,
  ntree = 500, mtry = floor(sqrt(ncol(train)-1))
)

rf_pred_test <- predict(rf_final, newdata = test)
rf_test_metrics <- compute_metrics(rf_pred_test, test$label)

### lift chart for rf
rf_prob <- predict(rf_final, newdata = test, type = "prob")[, 2]
actual_rf <- as.numeric(test$label) - 1
g_rf <- gains(actual = actual_rf, predicted = rf_prob, groups = 10)

plot(g_rf,
     main  = "Lift Chart for Random Forest (test set)",
     xlab  = "Depth of File (Deciles)",
     ylab  = "Mean Response",
     col   = c("red", "green", "skyblue"),
     lty   = c(1, 1, 1),
     legend = c("Mean Response",
                "Cumulative Mean Response",
                "Mean Predicted Response"))

## XGBoost test
train_x_full <- model.matrix(label ~ ., train)[, -1]
test_x_full  <- model.matrix(label ~ ., test)[, -1]
train_y_full <- ifelse(train$label == "high", 1, 0)
test_y_full  <- ifelse(test$label  == "high", 1, 0)

dtrain_full <- xgb.DMatrix(train_x_full, label=train_y_full)
dtest_full  <- xgb.DMatrix(test_x_full,  label=test_y_full)

xgb_final <- xgb.train(
  params = list(
    objective = "binary:logistic",
    max_depth = 4, eta = 0.1,
    subsample = 0.8, colsample_bytree = 0.8
  ),
  data = dtrain_full,
  nrounds = 100
)

final_prob   <- predict(xgb_final, test_x_full)
final_class  <- ifelse(final_prob > 0.5, "high", "low")
final_factor <- factor(final_class, levels = c("low", "high"))

xgb_test_metrics <- compute_metrics(final_factor, test$label)

### lift chart for xgb
xgb_prob <- final_prob
actual_xgb <- as.numeric(test$label) - 1
g_xgb <- gains(actual = actual_xgb, predicted = xgb_prob, groups = 10)

plot(g_xgb,
     main  = "Lift Chart for XGBoost (test set)",
     xlab  = "Depth of File (Deciles)",
     ylab  = "Mean Response",
     col   = c("red", "green", "skyblue"),
     lty   = c(1, 1, 1),
     legend = c("Mean Response",
                "Cumulative Mean Response",
                "Mean Predicted Response"))

# Test Summary

test_summary <- rbind(
  DecisionTree = dt_test_metrics,
  RandomForest = rf_test_metrics,
  XGBoost      = xgb_test_metrics
)

test_summary
