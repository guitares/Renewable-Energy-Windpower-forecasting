# Baseline
library(pacman)
p_load(dplyr, ggplot2, rpart, caret, ranger, Matrix, car, lmtest, doParallel, lubridate, forecast, viridis, tidyr, Metrics, data.table, readr)
setwd("F:/My Drive/97 Article") # User should set their own working directory where the data file is located
wp <- read_csv("T1.csv") # Load the data
n <- nrow(wp)
train_size <- floor(0.7 * n)
train <- wp[1:train_size, ]
test <- wp[(train_size + 1):n, ]
results_df <- data.frame(Model = character(), RMSE = numeric(), R2 = numeric(), stringsAsFactors = F)

# 1.1. ARIMAX----
set.seed(123)
predictor_vars <- c("Wind_Speed", "Theoretical_Power_Curve", "Wind_Direction")
y_train <- train$active_power
xreg_train <- train[, predictor_vars]
y_test <- test$active_power
xreg_test <- test[, predictor_vars]

arimax_model <- auto.arima(y_train, xreg = as.matrix(xreg_train), seasonal = T, stepwise = T, approximation = FALSE)

# 1.1.1 Train
train_predicted <- forecast(arimax_model, xreg = as.matrix(xreg_train), h = train_size)$mean
ARIMAX_1.1.1.rmse <- RMSE(train_predicted, train$active_power)
ARIMAX_1.1.1.r2 <- R2(train_predicted, train$active_power)
(results_df <- rbind(results_df, data.frame(Model = "1.1.1", RMSE = ARIMAX_1.1.1.rmse, R2 = ARIMAX_1.1.1.r2))) # Store results

# 1.1.2 Test
arimax_forecast <- forecast(arimax_model, xreg = as.matrix(xreg_test))
test_predicted <- as.numeric(arimax_forecast$mean)
ARIMAX_1.1.2.rmse <- RMSE(test_predicted, test$active_power)
ARIMAX_1.1.2.r2 <- R2(test_predicted, test$active_power)
(results_df <- rbind(results_df, data.frame(Model = "1.1.2", RMSE = ARIMAX_1.1.2.rmse, R2 = ARIMAX_1.1.2.r2)))

# 2.1 MLR----
# 2.1.1 MLR train
set.seed(123)
MLR_2.1 <- lm(active_power ~ ., data = train[, !names(train) %in% "Time"])
train_MLR_pred <- predict(MLR_2.1, newdata = train[, !names(train) %in% "Time"])
# Compute training performance
MLR_2.1.1_rmse <- RMSE(train_MLR_pred, train$active_power)
MLR_2.1.1_r2 <- R2(train_MLR_pred, train$active_power)
(results_df <- rbind(results_df, data.frame(Model = "2.1.1", RMSE = MLR_2.1.1_rmse, R2 = MLR_2.1.1_r2)))

# 2.1.2 MLR test
test_MLR_pred <- predict(MLR_2.1, newdata = test[, !names(test) %in% "Time"]) # Prediction
MLR_2.1.2_rmse <- RMSE(test_MLR_pred, test$active_power)
MLR_2.1.2_r2 <- R2(test_MLR_pred, test$active_power)
(results_df <- rbind(results_df, data.frame(Model = "2.1.2", RMSE = MLR_2.1.2_rmse, R2 = MLR_2.1.2_r2)))

# 3.1 DT----
cl <- makeCluster(detectCores() - 1)
registerDoParallel(cl)
train_control_ts <- trainControl(method = "timeslice", initialWindow = floor(0.8 * nrow(train)), horizon = 1, fixedWindow = F, skip = floor(nrow(train) * 0.01), verboseIter = T, allowParallel = T)
tune_grid <- expand.grid(cp = seq(0.001, 0.05, by = 0.0025))
control_params <- rpart.control(minsplit = 100, minbucket = 50, maxdepth = 4)
set.seed(123)
tuned_tree_model_3.1 <- train(active_power ~ ., data = train[, !names(train) %in% "Time"], method = "rpart", trControl = train_control_ts, tuneGrid = tune_grid, control = control_params)
stopCluster(cl)
registerDoSEQ()

# 3.1.1 DT Train
train_predictions <- predict(tuned_tree_model_3.1, newdata = train[, !names(train) %in% "Time"])
DT_3.1.1_rmse <- RMSE(train_predictions, train$active_power)
DT_3.1.1_train_r2 <- R2(train_predictions, train$active_power)
results_df <- rbind(results_df, data.frame(Model = "3.1.1", RMSE = DT_3.1.1_rmse, R2 = DT_3.1.1_train_r2))
print("Best tune for Decision Tree:")
print(tuned_tree_model_3.1$bestTune) # Print the best complexity parameter found during tuning.

# 3.1.2 DT Test
test_predictions <- predict(tuned_tree_model_3.1, newdata = test[, !names(test) %in% "Time"])
DT_3.1.2_rmse <- RMSE(test_predictions, test$active_power)
DT_3.1.2_r2 <- R2(test_predictions, test$active_power)
(results_df <- rbind(results_df, data.frame(Model = "3.1.2", RMSE = DT_3.1.2_rmse, R2 = DT_3.1.2_r2)))

# 4.1 RF----
num_cores <- detectCores() - 1
cl <- makeCluster(num_cores)
registerDoParallel(cl)
train_control_rf <- trainControl(method = "timeslice", initialWindow = floor(0.8 * nrow(train)), horizon = 1, fixedWindow = F, skip = floor(nrow(train) * 0.01), verboseIter = T, allowParallel = T)
tune_grid_rf <- expand.grid(mtry = c(3), splitrule = "variance", min.node.size = c(200))

# 4.1.1 RF train
set.seed(123)
ranger_model <- train(active_power ~ ., data = train[, !names(train) %in% "Time"], method = "ranger", trControl = train_control_rf, tuneGrid = tune_grid_rf, num.threads = num_cores)
stopCluster(cl) # Stop the parallel cluster once training is complete
registerDoSEQ() # Reset to sequential mode
rf_train_predictions <- predict(ranger_model, newdata = train[, !names(train) %in% "Time"])
RF_4.1.1_rmse <- RMSE(rf_train_predictions, train$active_power)
RF_4.1.1_r2 <- R2(rf_train_predictions, train$active_power)
(results_df <- rbind(results_df, data.frame(Model = "4.1.1", RMSE = RF_4.1.1_rmse, R2 = RF_4.1.1_r2)))
print("Best parameters for Random Forest:")
print(ranger_model$bestTune) # Print the best hyperparameters found.

# 4.1.2 RF test
rf_test_predictions <- predict(ranger_model, newdata = test[, !names(test) %in% "Time"])
RF_4.1.2_rmse <- RMSE(rf_test_predictions, test$active_power) # model's performance.
RF_4.1.2_r2 <- R2(rf_test_predictions, test$active_power)
(results_df <- rbind(results_df, data.frame(Model = "4.1.2", RMSE = RF_4.1.2_rmse, R2 = RF_4.1.2_r2)))

# 5.1 XGBoost----
num_cores <- detectCores() - 1
cl <- makeCluster(num_cores)
registerDoParallel(cl)
train_control_xgb <- trainControl(method = "timeslice", initialWindow = floor(0.8 * nrow(train)), horizon = 1, fixedWindow = F, skip = floor(nrow(train) * 0.01), verboseIter = T, allowParallel = T)
grid_xgb <- expand.grid(nrounds = 200, max_depth = 4, eta = 0.03, gamma = 0, colsample_bytree = 0.8, min_child_weight = 6, subsample = 0.8)

# 5.1.1 XGB train
set.seed(123)
xgb_tuned <- train(active_power ~ ., data = train[, !names(train) %in% "Time"], method = "xgbTree", trControl = train_control_xgb, tuneGrid = grid_xgb, metric = "RMSE")
stopCluster(cl)
registerDoSEQ()
xgb_train_predictions <- predict(xgb_tuned, newdata = train[, !names(train) %in% "Time"])
XGB_5.1.1_rmse <- RMSE(xgb_train_predictions, train$active_power)
XGB_5.1.1_r2 <- R2(xgb_train_predictions, train$active_power)
(results_df <- rbind(results_df, data.frame(Model = "5.1.1", RMSE = XGB_5.1.1_rmse, R2 = XGB_5.1.1_r2)))

cat("Best hyperparameters for XGBoost:\n")
print(xgb_tuned$bestTune) # Print the best hyperparameters.

# 5.1.2 XGB test
xgb_test_predictions <- predict(xgb_tuned, newdata = test[, !names(test) %in% "Time"])
XGB_5.1.2_rmse <- RMSE(xgb_test_predictions, test$active_power)
XGB_5.1.2_r2 <- R2(xgb_test_predictions, test$active_power)
(results_df <- rbind(results_df, data.frame(Model = "5.1.2", RMSE = XGB_5.1.2_rmse, R2 = XGB_5.1.2_r2)))
