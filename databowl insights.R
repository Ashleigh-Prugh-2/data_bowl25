# Load necessary libraries
library(dplyr)
library(fastDummies)
library(xgboost)
library(iml)
library(caret)  # for cross-validation and evaluation
library(ggplot2)  # for plotting calibration plots
library(arrow)

# Read and combine all datasets
read_and_combine_datasets <- function(data_dir, weeks) {
  files <- paste0(data_dir, "/prepared_data_week_", weeks, ".parquet")
  combined_data <- bind_rows(lapply(files, read_parquet))
  return(combined_data)
}

# Split data into training and testing sets
split_data <- function(data, train_ratio = 0.8, seed = 42) {
  set.seed(seed)
  train_indices <- sample(1:nrow(data), size = train_ratio * nrow(data))
  train_data <- data[train_indices, ]
  test_data <- data[-train_indices, ]
  return(list(train = train_data, test = test_data))
}

# Train pre-motion model (XGBoost)
train_xgboost_model <- function(train_data) {
  train_data <- train_data %>% filter(frameId < motion_frame)  # Filter before motion
  
  x_features_pre_motion <- train_data %>% 
    select(relative_distance, is_motion, speed_diff, max_speed_diff, distance_from_ball, 
           x_diff, y_diff, down, yardsToGo, score_diff, gameClock_seconds, 
           mismatch, height_diff, weight_diff, starts_with("offensive_position_"), starts_with("primary_defensive_position_"), starts_with('pff_passCoverage_'))
  
  dtrain <- xgb.DMatrix(data = as.matrix(x_features_pre_motion), label = train_data$wasTargettedReceiver)
  
  params <- list(objective = 'binary:logistic', eval_metric = 'logloss', max_depth = 6, eta = 0.1, subsample = 0.8)
  model <- xgboost(params = params, data = dtrain, nrounds = 100, verbose = 0)
  
  return(model)
}

# Train post-motion model (XGBoost)
train_post_motion_xgboost <- function(train_data) {
  train_data <- train_data %>% filter(frameId > motion_frame & frameId < ball_snap_frame)  # Filter post-motion but before ball snap
  
  x_features_post_motion <- train_data %>% 
    select(relative_distance, is_motion, speed_diff, max_speed_diff, distance_from_ball, 
           x_diff, y_diff, down, yardsToGo, score_diff, gameClock_seconds, 
           mismatch, height_diff, weight_diff, starts_with("offensive_position_"), starts_with("primary_defensive_position_"), starts_with('pff_passCoverage_'))
  
  dtrain_post_motion <- xgb.DMatrix(data = as.matrix(x_features_post_motion), label = train_data$wasTargettedReceiver)
  
  params_post_motion <- list(objective = 'binary:logistic', eval_metric = 'logloss', max_depth = 6, eta = 0.1, subsample = 0.8)
  model_post_motion <- xgboost(params = params_post_motion, data = dtrain_post_motion, nrounds = 100, verbose = 0)
  
  return(model_post_motion)
}

# Train play success model (XGBoost for play success prediction given targeted receiver)
train_play_success_model <- function(train_data) {
  # Filter for rows where the receiver was targeted
  train_data <- train_data %>% 
    filter(frameId < ball_snap_frame) %>% 
    filter(wasTargettedReceiver == 1) %>%
    mutate(play_success = case_when(
      expectedPointsAdded > 0 ~ 1,  # consider positive epa as success
      TRUE ~ 0                      # Otherwise, consider it a failure
    ))
  
  # Select features for play success model
  x_features_success <- train_data %>%
    select(relative_distance, is_motion, speed_diff, max_speed_diff, distance_from_ball, 
           x_diff, y_diff, down, yardsToGo, score_diff, gameClock_seconds, 
           mismatch, height_diff, weight_diff, starts_with("offensive_position_"), starts_with("primary_defensive_position_"), starts_with('pff_passCoverage_'), 
           expectedPointsAdded, yardsGained)
  
  # Create DMatrix for training the play success model
  dtrain_success <- xgb.DMatrix(data = as.matrix(x_features_success), label = train_data$play_success)
  
  # Model parameters for XGBoost
  params_success <- list(objective = 'binary:logistic', eval_metric = 'logloss', max_depth = 6, eta = 0.1, subsample = 0.8)
  model_success <- xgboost(params = params_success, data = dtrain_success, nrounds = 100, verbose = 0)
  
  return(model_success)
}

# Function to predict pre-motion, post-motion, and play success probabilities
predict_play_outcome <- function(model_pre_motion, model_post_motion, model_success, test_data) {
  # Ensure consistent feature preparation
  test_data <- test_data %>%
    mutate(play_success = case_when(
      expectedPointsAdded > 0 ~ 1,
      TRUE ~ 0
    ))
  
  # Pre-motion prediction
  x_features_pre_motion <- test_data %>%
    select(relative_distance, is_motion, speed_diff, max_speed_diff, distance_from_ball,
           x_diff, y_diff, down, yardsToGo, score_diff, gameClock_seconds,
           mismatch, height_diff, weight_diff, starts_with("offensive_position_"), 
           starts_with("primary_defensive_position_"), starts_with('pff_passCoverage_'))
  dtest_pre_motion <- xgb.DMatrix(data = as.matrix(x_features_pre_motion))
  pre_motion_preds <- predict(model_pre_motion, dtest_pre_motion)
  
  # Post-motion prediction
  x_features_post_motion <- x_features_pre_motion
  dtest_post_motion <- xgb.DMatrix(data = as.matrix(x_features_post_motion))
  post_motion_preds <- predict(model_post_motion, dtest_post_motion)
  
  # Success prediction
  x_features_success <- test_data %>%
    select(relative_distance, is_motion, speed_diff, max_speed_diff, distance_from_ball,
           x_diff, y_diff, down, yardsToGo, score_diff, gameClock_seconds,
           mismatch, height_diff, weight_diff, starts_with("offensive_position_"), 
           starts_with("primary_defensive_position_"), starts_with('pff_passCoverage_'), expectedPointsAdded, yardsGained)
  
  dtest_success <- xgb.DMatrix(data = as.matrix(x_features_success))
  success_preds <- predict(model_success, dtest_success)
  
  return(list(pre_motion_preds = pre_motion_preds, post_motion_preds = post_motion_preds, success_preds = success_preds))
}

# Cross-validation and model evaluation using RMSE
calculate_rmse <- function(actual, predicted) {
  sqrt(mean((actual - predicted)^2))
}

# Main script
data_dir <- "/Users/user"
weeks <- 1:8

# Step 1: Read and combine datasets
combined_data <- read_and_combine_datasets(data_dir, weeks)

# Step 2: Prepare the combined data
prepared_data <- combined_data 

# Step 3: Split into training and test sets
splits <- split_data(prepared_data)
train_data <- splits$train
test_data <- splits$test

# Step 4: Train models
model_pre_motion <- train_xgboost_model(train_data)
model_post_motion <- train_post_motion_xgboost(train_data)
model_success <- train_play_success_model(train_data)

# Step 5: Evaluate models on test data
test_predictions <- predict_play_outcome(model_pre_motion, model_post_motion, model_success, test_data)


# Calculate RMSE for evaluation
rmse_pre_motion <- calculate_rmse(test_data$wasTargettedReceiver, test_predictions$pre_motion_preds)
rmse_post_motion <- calculate_rmse(test_data$wasTargettedReceiver, test_predictions$post_motion_preds)
rmse_play_success <- calculate_rmse(test_data$play_success, test_predictions$success_preds)

# Print RMSE results
rmse_results <- data.frame(
  Model = c("Pre-Motion", "Post-Motion", "Play Success"),
  RMSE = c(rmse_pre_motion, rmse_post_motion, rmse_play_success)
)
print(rmse_results)
