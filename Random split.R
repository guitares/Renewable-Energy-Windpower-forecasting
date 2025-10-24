# R Code for Wind Power Forecasting
# Random train-test split


# --- SETUP: LOAD REQUIRED LIBRARIES ---
# if (!require("pacman")) install.packages("pacman")
library(pacman) # used to load and install other packages easily.

p_load(readxl, # To read Excel files
  dplyr, # For data manipulation
  ggplot2, # For plotting
  rpart, # For decision trees
  caret, # For model training and evaluation (classification and regression training)
  ranger, # A fast implementation of Random Forest
  xgboost, # For Extreme Gradient Boosting
  Matrix, # For creating matrix objects required by xgboost
  car, # For MLR diagnostic test (VIF)
  doParallel, # For parallel processing
  lmtest)# For MLR diagnostic tests (Breusch-Pagan and Durbin-Watson)

# --- DATA LOADING AND PRE-PROCESSING ---
# Note: The user should set their own working directory where the data file is located.
# setwd("path/to/your/directory")

windpower <- read_excel("WP.xlsx") # Load the dataset from an Excel file.
wp <- windpower %>% # Rename the columns for easier access in R.
  rename(active_power = "LV ActivePower (kW)",
    wind_speed = "Wind Speed (m/s)",
    power_curve = "Theoretical_Power_Curve (KWh)",
    wind_dir = "Wind Direction")
wp <- wp %>% dplyr::select(-"Date/Time") # Remove 'Date/Time' as it will not be used

# --- DATA PARTITIONING ---
set.seed(123)
train_index <- createDataPartition(wp$active_power, p = 0.7, list = FALSE)
training_data <- wp[train_index, ]
test_data <- wp[-train_index, ]

# Create an empty data frame to store the performance metrics (RMSE and R-squared) of each model.
results_df_random_split <- data.frame(Model = character(), RMSE = numeric(), R2 = numeric(), stringsAsFactors = F)

# --- MODEL TRAINING, TUNING, AND EVALUATION ---
####  Multiple Linear Regression ####
multiple_reg <- lm(active_power ~ ., data = training_data) # Fit MLR model
(dw_test <- dwtest(multiple_reg)) # 1. Independence of errors: Durbin-Watson test
(bp_test <- bptest(multiple_reg)) # 2. Homoscedasticity: Breusch-Pagan test
(vif_values <- vif(multiple_reg)) # 3. Multicollinearity: Variance Inflation Factor (VIF)
summary(multiple_reg)
test_data$predicted_power_lm <- predict(multiple_reg, newdata = test_data) # predictions on the test set.
lm_rmse <- RMSE(test_data$predicted_power_lm, test_data$active_power) # Calculate RMSE and R2
lm_r2 <- R2(test_data$predicted_power_lm, test_data$active_power)
(results_df_random_split <- rbind(results_df_random_split, data.frame(Model = "Linear Regression", RMSE = lm_rmse, R2 = lm_r2))) # Add the results to summary data frame.

#### Decision Tree ####
# Train a DT with hyperparameter tuning to find the best complexity parameter (cp).
train_control <- trainControl(method = "cv", number = 10)
tune_grid <- expand.grid(cp = seq(0.001, 0.02, by = 0.001))
control_params <- rpart.control(minsplit = 20, # min. no. of observations before a split
                                minbucket = 50, # minimum terminal node
                                maxdepth = 6) # max. depth of the tree
tuned_tree_model <- train(active_power ~ ., data = training_data, method = "rpart", trControl = train_control, tuneGrid = tune_grid, control = control_params)
print("Best tune for Decision Tree:")
print(tuned_tree_model$bestTune) # Print the best complexity parameter found during tuning.
tree_predictions_optimized <- predict(tuned_tree_model, newdata = test_data) # predictions using the tuned model
tree_rmse_optimized <- RMSE(tree_predictions_optimized, test_data$active_power)# performance metrics for the tuned DT
tree_r2_optimized <- R2(tree_predictions_optimized, test_data$active_power)
(results_df_random_split <- rbind(results_df_random_split, data.frame(Model = "Decision Tree", RMSE = tree_rmse_optimized, R2 = tree_r2_optimized))) # Add results to the summary data frame.

#### Random Forest ####
num_cores <- detectCores() - 1 # Detect the number of available CPU cores.
cl <- makeCluster(num_cores)
registerDoParallel(cl)
# Define the control parameters for training, enabling cross-validation
train_control_rf <- trainControl(method = "cv",
                                 number = 10, verboseIter = T, allowParallel = T) # Show progress and parallel processing.
# Define the hyperparameter grid to search, testing specific values.
tune_grid_rf <- expand.grid(mtry = c(2), # No. of variables randomly sampled as candidates at each split.
                            splitrule = c("variance"), # Splitting rule for regression.
                            min.node.size = c(1)) # Minimum size of terminal nodes.
# Train the Random Forest model using the (fast) 'ranger' method
set.seed(123)
ranger_model <- train(active_power ~ ., data = training_data, method = "ranger", trControl = train_control_rf, tuneGrid = tune_grid_rf, num.threads = num_cores)
stopCluster(cl) # Stop the parallel cluster once training is complete.
registerDoSEQ()
print("Best parameters for Random Forest:")
print(ranger_model$bestTune) # Print the best hyperparameters found.
rf_predictions <- predict(ranger_model, newdata = test_data) # predictions on the test data.
rf_rmse <- RMSE(rf_predictions, test_data$active_power) # model's performance.
rf_r2 <- R2(rf_predictions, test_data$active_power)
(results_df_random_split <- rbind(results_df_random_split, data.frame(Model = "Random Forest", RMSE = rf_rmse, R2 = rf_r2)))

#### XGBoost ####
cl <- makeCluster(detectCores() - 1)
registerDoParallel(cl)
control_xgb <- trainControl(method = "cv", number = 10, verboseIter = TRUE, allowParallel = TRUE)
grid_xgb <- expand.grid(# Define the hyperparameter grid for XGBoost
  nrounds = c(100), # No. of boosting rounds.
  max_depth = c(6), # Max tree depth.
  eta = c(0.05), # Learning rate.
  gamma = c(0.1), # Min.loss reduction to make a split.
  colsample_bytree = c(1), # Subsample ratio of columns when constructing each tree.
  min_child_weight = c(1), # Minimum sum of instance weight needed in a child.
  subsample = c(1)) # Subsample ratio of the training instance.
set.seed(123)
xgb_tuned <- train( active_power ~ ., data = training_data, method = "xgbTree", trControl = control_xgb, tuneGrid = grid_xgb, metric = "RMSE") # Train the XGBoost model.
stopCluster(cl) # Stop the parallel cluster.
registerDoSEQ()
cat("Best hyperparameters for XGBoost:\n")
print(xgb_tuned$bestTune) # Print the best hyperparameters.
xgb_predictions <- predict(xgb_tuned, newdata = test_data) # predictions and model evaluation
xgb_rmse <- RMSE(xgb_predictions, test_data$active_power)
xgb_r2 <- R2(xgb_predictions, test_data$active_power)
(results_df_random_split <- rbind(results_df_random_split, data.frame(Model = "XGBoost", RMSE = xgb_rmse, R2 = xgb_r2)))
