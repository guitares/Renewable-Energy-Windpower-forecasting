# 0.3 Cyclical features
library(pacman)
p_load(dplyr, ggplot2, rpart, caret, ranger, Matrix, car, lmtest, doParallel, lubridate, forecast, viridis, tidyr, Metrics, data.table, readr)
setwd("F:/My Drive/97 Article") # User should set their own working directory where the data file is located
wp <- read_csv("T1.csv") # Load the data

wp_cyc <- wp
wp_cyc$Time <- as.POSIXct(wp$Time, format = "%d %m %Y %H:%M") # Convert to datetime
wp_cyc$hour <- as.numeric(format(wp_cyc$Time, "%H")) # Extract cyclical components
wp_cyc$day_of_week <- as.numeric(format(wp_cyc$Time, "%u")) # 1-7 (Mon-Sun)
wp_cyc$month <- as.numeric(format(wp_cyc$Time, "%m"))
wp_cyc$hour_sin <- sin(2 * pi * wp_cyc$hour / 24) # Apply cyclical transformations
wp_cyc$hour_cos <- cos(2 * pi * wp_cyc$hour / 24)
wp_cyc$dow_sin <- sin(2 * pi * wp_cyc$day_of_week / 7)
wp_cyc$dow_cos <- cos(2 * pi * wp_cyc$day_of_week / 7)
wp_cyc$month_sin <- sin(2 * pi * wp_cyc$month / 12)
wp_cyc$month_cos <- cos(2 * pi * wp_cyc$month / 12)
periods_per_day <- 24 * 6 # For 10-minute intra-day pattern: 6 periods/hour
period_count <- as.numeric(format(wp_cyc$Time, "%H")) * 6 + as.numeric(format(wp_cyc$Time, "%M")) / 10
wp_cyc$period_sin <- sin(2 * pi * period_count / periods_per_day)
wp_cyc$period_cos <- cos(2 * pi * period_count / periods_per_day)
wp_cyc$wind_dir_sin <- sin(2 * pi * wp_cyc$Wind_Direction / 360) # Wind direction is also cyclical (0-360 degrees)
wp_cyc$wind_dir_cos <- cos(2 * pi * wp_cyc$Wind_Direction / 360)
wp_cyc$hour <- NULL # Remove intermediate columns
wp_cyc$day_of_week <- NULL
wp_cyc$month <- NULL

n <- nrow(wp_cyc)
train_size <- floor(0.7 * n)

set.seed(123)
train <- wp_cyc[1:train_size, ]
test <- wp_cyc[(train_size + 1):n, ]

results_df <- data.frame(Model = character(), RMSE = numeric(), R2 = numeric(), stringsAsFactors = F)

# 1.3 ARIMAX----
predictor_vars_2 <- c("Wind_Speed", "Theoretical_Power_Curve", "Wind_Direction", "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos", "period_sin", "period_cos", "wind_dir_sin", "wind_dir_cos")
y_train <- train$active_power
xreg_train <- train[, predictor_vars_2]
y_test <- test$active_power
xreg_test <- test[, predictor_vars_2]
set.seed(123)
arimax_model <- auto.arima(y_train, xreg = as.matrix(xreg_train), seasonal = T, stepwise = T, approximation = FALSE)

# 1.3.1 Train
train_predicted <- forecast(arimax_model, xreg = as.matrix(xreg_train), h = train_size)$mean
ARIMAX_1.3.1.rmse <- RMSE(train_predicted, train$active_power)
ARIMAX_1.3.1.r2 <- R2(train_predicted, train$active_power)
(results_df <- rbind(results_df, data.frame(Model = "1.3.1", RMSE = ARIMAX_1.3.1.rmse, R2 = ARIMAX_1.3.1.r2)))

# 1.3.2 Test
arimax_forecast <- forecast(arimax_model, xreg = as.matrix(xreg_test))
test_predicted <- as.numeric(arimax_forecast$mean)
ARIMAX_1.3.2.rmse <- RMSE(test_predicted, test$active_power)
ARIMAX_1.3.2.r2 <- R2(test_predicted, test$active_power)
(results_df <- rbind(results_df, data.frame(Model = "1.3.2", RMSE = ARIMAX_1.3.2.rmse, R2 = ARIMAX_1.3.2.r2)))

# 2.3 MLR----
# 2.3.1 MLR train
set.seed(123)
MLR_2.3 <- lm(active_power ~ ., data = train[, !names(train) %in% "Time"])
train_MLR_pred <- predict(MLR_2.3, newdata = train[, !names(train) %in% "Time"])
MLR_2.3.1_rmse <- RMSE(train_MLR_pred, train$active_power)
MLR_2.3.1_r2 <- R2(train_MLR_pred, train$active_power)
(results_df <- rbind(results_df, data.frame(Model = "2.3.1", RMSE = MLR_2.3.1_rmse, R2 = MLR_2.3.1_r2)))

# 2.3.2 MLR test
test_MLR_pred <- predict(MLR_2.3, newdata = test[, !names(test) %in% "Time"]) # Prediction
MLR_2.3.2_rmse <- RMSE(test_MLR_pred, test$active_power)
MLR_2.3.2_r2 <- R2(test_MLR_pred, test$active_power)
(results_df <- rbind(results_df, data.frame(Model = "2.3.2", RMSE = MLR_2.3.2_rmse, R2 = MLR_2.3.2_r2)))

# 3.3 DT----
cl <- makeCluster(detectCores() - 1)
registerDoParallel(cl)
train_control_ts <- trainControl(method = "timeslice", initialWindow = floor(0.8 * nrow(train)), horizon = 1, fixedWindow = F, skip = floor(nrow(train) * 0.01), verboseIter = T, allowParallel = T)
tune_grid <- expand.grid(cp = seq(0.001, 0.05, by = 0.0025))
control_params <- rpart.control(minsplit = 100, minbucket = 50, maxdepth = 4)
set.seed(123)
tuned_tree_model_3.3 <- train(active_power ~ ., data = train[, !names(train) %in% "Time"], method = "rpart", trControl = train_control_ts, tuneGrid = tune_grid, control = control_params)
stopCluster(cl)
registerDoSEQ()

# 3.3.1 DT Train
train_predictions <- predict(tuned_tree_model_3.3, newdata = train[, !names(train) %in% "Time"])
DT_3.3.1_rmse <- RMSE(train_predictions, train$active_power)
DT_3.3.1_train_r2 <- R2(train_predictions, train$active_power)
results_df <- rbind(results_df, data.frame(Model = "3.3.1", RMSE = DT_3.3.1_rmse, R2 = DT_3.3.1_train_r2))

# 3.3.2 DT Test
test_predictions <- predict(tuned_tree_model_3.3, newdata = test[, !names(test) %in% "Time"])
DT_3.3.2_rmse <- RMSE(test_predictions, test$active_power)
DT_3.3.2_r2 <- R2(test_predictions, test$active_power)
(results_df <- rbind(results_df, data.frame(Model = "3.3.2", RMSE = DT_3.3.2_rmse, R2 = DT_3.3.2_r2)))


# 4.3 RF----
num_cores <- detectCores() - 1
cl <- makeCluster(num_cores)
registerDoParallel(cl)
train_control_rf <- trainControl(method = "timeslice", initialWindow = floor(0.8 * nrow(train)), horizon = 1, fixedWindow = F, skip = floor(nrow(train) * 0.01), verboseIter = T, allowParallel = T)
tune_grid_rf <- expand.grid(mtry = c(3), splitrule = "variance", min.node.size = c(200))

# 4.3.1 RF train
set.seed(123)
ranger_model <- train(active_power ~ ., data = train[, !names(train) %in% "Time"], method = "ranger", trControl = train_control_rf, tuneGrid = tune_grid_rf, num.threads = num_cores)
stopCluster(cl)
registerDoSEQ()
rf_train_predictions <- predict(ranger_model, newdata = train[, !names(train) %in% "Time"])
RF_4.3.1_rmse <- RMSE(rf_train_predictions, train$active_power)
RF_4.3.1_r2 <- R2(rf_train_predictions, train$active_power)
(results_df <- rbind(results_df, data.frame(Model = "4.3.1", RMSE = RF_4.3.1_rmse, R2 = RF_4.3.1_r2)))

# 4.3.2 RF test
rf_test_predictions <- predict(ranger_model, newdata = test[, !names(test) %in% "Time"])
RF_4.3.2_rmse <- RMSE(rf_test_predictions, test$active_power) # model's performance.
RF_4.3.2_r2 <- R2(rf_test_predictions, test$active_power)
(results_df <- rbind(results_df, data.frame(Model = "4.3.2", RMSE = RF_4.3.2_rmse, R2 = RF_4.3.2_r2)))

# 5.3 XGBoost----
num_cores <- detectCores() - 1
cl <- makeCluster(num_cores)
registerDoParallel(cl)
train_control_xgb <- trainControl(method = "timeslice", initialWindow = floor(0.8 * nrow(train)), horizon = 1, fixedWindow = F, skip = floor(nrow(train) * 0.01), verboseIter = T, allowParallel = T)
grid_xgb <- expand.grid(nrounds = 200, max_depth = 4, eta = 0.03, gamma = 0, colsample_bytree = 0.8, min_child_weight = 6, subsample = 0.8)

# 5.3.1 XGB train
set.seed(123)
xgb_tuned <- train(active_power ~ ., data = train[, !names(train) %in% "Time"], method = "xgbTree", trControl = train_control_xgb, tuneGrid = grid_xgb, metric = "RMSE")
stopCluster(cl)
registerDoSEQ()
xgb_train_predictions <- predict(xgb_tuned, newdata = train[, !names(train) %in% "Time"])
XGB_5.3.1_rmse <- RMSE(xgb_train_predictions, train$active_power)
XGB_5.3.1_r2 <- R2(xgb_train_predictions, train$active_power)
(results_df <- rbind(results_df, data.frame(Model = "5.3.1", RMSE = XGB_5.3.1_rmse, R2 = XGB_5.3.1_r2)))

cat("Best hyperparameters for XGBoost:\n")
print(xgb_tuned$bestTune) # Print the best hyperparameters.

# 5.3.2 XGB test
xgb_test_predictions <- predict(xgb_tuned, newdata = test[, !names(test) %in% "Time"])
XGB_5.3.2_rmse <- RMSE(xgb_test_predictions, test$active_power)
XGB_5.3.2_r2 <- R2(xgb_test_predictions, test$active_power)
(results_df <- rbind(results_df, data.frame(Model = "5.3.2", RMSE = XGB_5.3.2_rmse, R2 = XGB_5.3.2_r2)))
