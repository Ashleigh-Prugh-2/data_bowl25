# Load necessary libraries
library(dplyr)
library(arrow)
library(ggplot2)
library(gganimate)
library(xgboost)
library(fastDummies)
library(data.table)

# Assuming the models are already trained (from previous code)

# 1. Filter for a Specific Play (e.g., gameId = 2021090900, playId = 75)
game_id <- 2022091200  # Example gameId
play_id <- 346          # Example playId

tracking <- fread("/Users/user/Desktop/data bowl 2025/tracking_week_1.csv") #, select = c("gameId", "playId", "nflId", "x", "y", "s", "dis", "o", "dir", "event", "position"))


# Filter the dataset to the specific play
play_data <- combined_data %>%  
  filter(gameId == game_id, playId == play_id)

tracking <- tracking %>%  
  select(gameId, playId, frameId, nflId, x, y, s, a, o, dir) %>%  
  filter(gameId == game_id, playId == play_id)

library(dplyr)

players <- read.csv('/Users/user/Desktop/data bowl 2025/players.csv')


# Join for offensive nflId
play_data_offense <- play_data %>% 
  filter(!is.na(off_nflId)) %>% 
  rename(nflId = off_nflId) 


play_data <- left_join(tracking, play_data_offense, by= c('gameId', 'playId', 'frameId', 'nflId'))

play_data <- left_join(play_data, players, by = c('nflId'))




# 2. Prepare the data for prediction
# Pre-motion predictions
x_features_pre_motion <- play_data %>%  
  select(relative_distance, is_motion, speed_diff, max_speed_diff, distance_from_ball,
         x_diff, y_diff, down, yardsToGo, score_diff, gameClock_seconds,
         mismatch, height_diff, weight_diff, starts_with("offensive_position_"),  
         starts_with("primary_defensive_position_"), starts_with('pff_passCoverage_'))
dtest_pre_motion <- xgb.DMatrix(data = as.matrix(x_features_pre_motion))
pre_motion_preds <- predict(model_pre_motion, dtest_pre_motion)

# Post-motion predictions
x_features_post_motion <- x_features_pre_motion
dtest_post_motion <- xgb.DMatrix(data = as.matrix(x_features_post_motion))
post_motion_preds <- predict(model_post_motion, dtest_post_motion)

# Success predictions
x_features_success <- play_data %>%
  select(relative_distance, is_motion, speed_diff, max_speed_diff, distance_from_ball,
         x_diff, y_diff, down, yardsToGo, score_diff, gameClock_seconds,
         mismatch, height_diff, weight_diff, starts_with("offensive_position_"),  
         starts_with("primary_defensive_position_"), starts_with('pff_passCoverage_'), expectedPointsAdded, yardsGained)
dtest_success <- xgb.DMatrix(data = as.matrix(x_features_success))
success_preds <- predict(model_success, dtest_success)

# Add the predictions (probabilities) to the play data
play_data <- play_data %>%
  mutate(pre_motion_prob = pre_motion_preds,
         post_motion_prob = post_motion_preds,
         play_success_prob = success_preds) 

play_data <- left_join(play_data, players, by = c('nflId'))

# 3. Create the Football Field Plot for Animation
create_field <- function() {
  ggplot() +
    geom_rect(aes(xmin = 0, xmax = 120, ymin = 0, ymax = 53.3), fill = "#006400") +
    geom_vline(xintercept = seq(10, 110, by = 10), color = "white", linetype = "dashed") +
    geom_hline(yintercept = c(0, 53.3), color = "white") +
    theme_minimal() +
    theme(
      panel.grid = element_blank(),
      axis.text = element_blank(),
      axis.title = element_blank(),
      axis.ticks = element_blank()
    )
}

animation <- create_field() + 
  # Players: Color them by pre-motion probability, but grey out those with NA in offensive_position_ columns
  geom_point(data = play_data %>% filter(!is.na(nflId)), aes(x = x, y = y, color = pre_motion_prob), size = 3) + 
  geom_point(data = play_data %>%
               mutate(
                 all_na_or_zero_in_positions = rowSums(
                   !is.na(select(., starts_with("offensive_position_"))) & 
                     select(., starts_with("offensive_position_")) != 0
                 ) == 0  # Check if all values are either NA or 0
               ) %>%
               filter(all_na_or_zero_in_positions),
             aes(x = x, y = y), color = "grey", size = 3) + 
  geom_text(data = play_data %>% 
              filter(!is.na(nflId), pre_motion_prob > 0.16),  # Add filter for pre_motion_prob > 0
            aes(x = x, y = y, label = sprintf("%.2f", pre_motion_prob)), 
            color = "white", size = 3, vjust = -1) +  # Display probability as text for players
  geom_text(data = play_data %>% 
              filter(!is.na(nflId), pre_motion_prob > 0.16),  # Add filter for pre_motion_prob > 0
            aes(x = x, y = y, label = displayName), 
            color = "black", size = 3, vjust = 1.5) +  # Add player names below the points
  # Ball: Rows with NA in nflId correspond to the ball
  geom_point(data = play_data %>% filter(is.na(nflId)), aes(x = x, y = y), color = "orange", size = 3, shape = 16) + 
  scale_color_gradient(low = "blue", high = "red") + 
  labs(title = paste("Play Animation | Game: ", game_id, " | Play: ", play_id, " | Frame: {frameId}"),
       subtitle = "Pre-Motion Target Probability", 
       color = "Target Probability") + 
  transition_states(frameId, transition_length = 1, state_length = 1) + 
  ease_aes('linear')


# Save the Animation as a GIF
animate(animation, width = 800, height = 400, fps = 10, renderer = gifski_renderer("play_animation_with_names.gif"))

# Optionally display the animation in the console
animate(animation, width = 800, height = 400, fps = 10)


