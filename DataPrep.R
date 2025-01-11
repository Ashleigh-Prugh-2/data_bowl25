# reading in data for week 1 

library(data.table)  # For memory-efficient data processing
library(dplyr)       # For data wrangling
library(xgboost)     # For XGBoost model
library(caret)       # For cross-validation and performance metrics
library(Matrix)      # For sparse matrix conversion
library(ggplot2)     # For visualization
library(tidyr)
library(arrow)

# --------------------------------
# 1. Load Data (Use Partial Data)
# --------------------------------
# Load required datasets
games <- read.csv('/Users/user/Desktop/data bowl 2025/games.csv')
players <- read.csv('/Users/user/Desktop/data bowl 2025/players.csv')
plays <- read.csv('/Users/user/Desktop/data bowl 2025/plays.csv')
player_play <-  read.csv('/Users/user/Desktop/data bowl 2025/player_play.csv')

# Load one tracking file (use more files as needed)
tracking <- fread("/Users/user/Desktop/data bowl 2025/tracking_week_8.csv") #, select = c("gameId", "playId", "nflId", "x", "y", "s", "dis", "o", "dir", "event", "position"))

players <- players %>%
  separate(height, into = c("feet", "inches"), sep = "-", convert = TRUE) %>%
  mutate(
    height = feet * 12 + inches  # Convert feet to inches and add remaining inches
  ) %>%
  select(-feet, -inches)

# Join all relevant datasets
full_data <- tracking %>%
  left_join(plays, by = c("gameId", "playId")) %>%
  left_join(games, by = "gameId") %>%
  left_join(players, by = c("nflId")) %>%
  left_join(player_play, by = c("gameId", "playId", "nflId"))

full_data <- full_data %>% 
  filter(!is.na(passLength))

# --------------------------------
# 2. Feature Engineering
# --------------------------------
library(dplyr)

full_data <- full_data %>%
  mutate(
    offensive_position = case_when(
      position %in% c("QB", "WR", "RB", "TE", "G", "C", "T", "FB") ~ position,
      TRUE ~ NA_character_
    ),
    defensive_position = case_when(
      position %in% c("SS", "CB", "OLB", "ILB", "FS", "NT", "DE", "DT", "MLB", "LB", "DB") ~ position,
      TRUE ~ NA_character_
    ),
    is_offense = ifelse(position %in% c("QB", "WR", "RB", "TE", "G", "C", "T", "FB"), 1, 0),
    is_defender = ifelse(position %in% c("SS", "CB", "OLB", "ILB", "FS", "NT", "DE", "DT", "MLB", "LB", "DB"), 1, 0), 
    is_motion = ifelse(event %in% c("man_in_motion", "shift"), 1, 0),
    distance_from_ball = sqrt((x - 0)^2 + (y - 0)^2),
    x_diff = x - lag(x, 1, default = 0),
    y_diff = y - lag(y, 1, default = 0), 
    speed_diff = s - lag(s, 1, default = 0), 
    alignment_distance = sqrt(x_diff^2 + y_diff^2)
  )



full_data <- full_data %>%
  mutate(
    score_diff = abs(preSnapHomeScore - preSnapVisitorScore)  # Absolute value of score difference
  )

full_data <- full_data %>%
  mutate(
    gameClock_seconds = as.numeric(sub(":.*", "", gameClock)) * 60 + as.numeric(sub(".*:", "", gameClock))
  )
speed_data <- full_data %>% 
  group_by(nflId) %>%
  summarize(max_speed = max(s, na.rm = TRUE)) %>% 
  ungroup()

full_data <- left_join(full_data, speed_data, by = c("nflId"))


full_data <- full_data %>%
  select(
    frameId,
    gameId, 
    playId,
    nflId,
    event, 
    height, 
    weight, 
    max_speed,
    position,
    expectedPointsAdded,
    pff_primaryDefensiveCoverageMatchupNflId, 
    pff_secondaryDefensiveCoverageMatchupNflId,
    wasTargettedReceiver,
    expectedPointsAdded, 
    yardsGained, 
    is_offense,            # Offensive indicator
    is_defender,           # Defender indicator
    is_motion,             # Motion indicator
    distance_from_ball,    # Distance from the ball
    x_diff,                # Change in x position
    y_diff,                # Change in y position
    offensive_position,    # Offensive position (WR, RB, TE)
    defensive_position,    # Defensive position (CB, LB, S)
    down,                  # Down (1st, 2nd, etc.)
    yardsToGo,             # Yardage to first down
    score_diff,            # Score difference (if available)
    gameClock_seconds,     # Time left in the game (if available)
    pff_passCoverage, 
    speed_diff
  )

offense_data <- full_data %>%
  filter(is_offense == 1)

# Primary defensive coverage
primary_coverage_data <- full_data %>%
  filter(!is.na(pff_primaryDefensiveCoverageMatchupNflId)) %>%
  select(
    nflId,
    gameId,
    frameId,
    playId, 
    pff_primaryDefensiveCoverageMatchupNflId, 
    pff_passCoverage,
    defensive_position, 
    position, 
    height, 
    weight, 
    max_speed
  ) %>%
  rename(
    primary_defensive_position = defensive_position
  )

# Secondary defensive coverage
secondary_coverage_data <- full_data %>%
  filter(!is.na(pff_secondaryDefensiveCoverageMatchupNflId)) %>%
  select(
    nflId,
    gameId, 
    frameId,
    playId, 
    pff_secondaryDefensiveCoverageMatchupNflId, 
    pff_passCoverage, 
    defensive_position, 
    position, 
    height, 
    weight, 
    max_speed
  ) %>%
  rename(
    secondary_defensive_position = defensive_position
  )

# Combine both primary and secondary coverage data
defense_data <- bind_rows(primary_coverage_data, secondary_coverage_data) %>%
  distinct()

offense_data <- offense_data %>%
  left_join(players %>% select(nflId, height, weight), by = c("nflId", "height", "weight")) 

offense_data <- offense_data %>%
  rename(off_height = height, off_weight = weight, off_max_speed = max_speed, off_nflId = nflId) %>% 
  select(-pff_primaryDefensiveCoverageMatchupNflId, -pff_secondaryDefensiveCoverageMatchupNflId)

# Join defensive player data with players dataset
defense_data <- defense_data %>%
  left_join(players %>% select(nflId, height, weight), by = c("nflId", "height", 'weight')) %>%
  rename(def_height = height, def_weight = weight, def_max_speed = max_speed, def_nflId = nflId)
# Join offensive and defensive data
mismatch_data <- offense_data %>%
  left_join(defense_data, by = c("gameId", "playId", "frameId", "pff_passCoverage")) %>%
  mutate(
    match = if_else(off_nflId == pff_primaryDefensiveCoverageMatchupNflId | off_nflId == pff_secondaryDefensiveCoverageMatchupNflId, TRUE, FALSE)
  ) %>%
  filter(match == TRUE) %>%
  select(-match)

mismatch_data <- mismatch_data %>%
  mutate(
    mismatch = case_when(
      position.x == "WR" & !position.y %in% c("CB", "S") ~ 1,
      position.x == "RB" & !position.y %in% c("LB", "S") ~ 1,
      position.x == "TE" & !position.y %in% c("LB", "S") ~ 1,
      TRUE ~ 0
    ))

mismatch_data <- mismatch_data %>% 
  filter(!is.na(position.x))

pre_motion <- mismatch_data %>%
  filter(event == "man_in_motion" | event == "shift") %>%
  select(gameId, playId, frameId) %>%
  group_by(gameId, playId) %>%
  summarise(motion_frame = min(frameId))  # Assuming the first "snap" event is the snap frame

pre_snap <- mismatch_data %>%
  filter(event == "ball_snap") %>%
  select(gameId, playId, frameId) %>%
  group_by(gameId, playId) %>%
  summarise(ball_snap_frame = min(frameId)) 


mismatch_data <- left_join(mismatch_data, pre_motion, by = c("gameId", "playId"))

mismatch_data <- left_join(mismatch_data, pre_snap, by = c("gameId", "playId"))

mismatch_data <- mismatch_data %>%
  mutate(
    height_diff = off_height - def_height,
    weight_diff = off_weight - def_weight, 
    max_speed_diff = off_max_speed - def_max_speed
  )


# Prepare data function
prepare_data <- function(mismatch_data) {
  mismatch_data <- mismatch_data %>% 
    filter(!is.na(wasTargettedReceiver)) %>%  # Remove rows without target info
    mutate(relative_distance = sqrt(x_diff^2 + y_diff^2))  # Euclidean distance between defender and offensive player
  
  # One-hot encode categorical variables after calculating relative_distance
  mismatch_data <- fastDummies::dummy_cols(mismatch_data, 
                                           select_columns = c('offensive_position', 'primary_defensive_position'),
                                           remove_selected_columns = TRUE)
  # One-hot encode categorical variables after calculating relative_distance
  mismatch_data <- fastDummies::dummy_cols(mismatch_data, 
                                           select_columns = c('pff_passCoverage'),
                                           remove_selected_columns = TRUE)
  mismatch_data <- mismatch_data %>%
    mutate(play_success = case_when(
      expectedPointsAdded > 0 ~ 1,
      TRUE ~ 0
    ))
  # Select relevant columns
  mismatch_data <- mismatch_data %>% 
    select(frameId, gameId, playId, off_nflId, def_nflId, 
           relative_distance, is_motion, speed_diff, max_speed_diff, distance_from_ball, 
           x_diff, y_diff, down, yardsToGo, score_diff, gameClock_seconds, 
           mismatch, wasTargettedReceiver, height_diff, weight_diff, 
           expectedPointsAdded, yardsGained, motion_frame, ball_snap_frame, play_success, 
           starts_with("offensive_position_"), starts_with("primary_defensive_position_"), starts_with('pff_passCoverage_'))  # Include the dummy columns
  
  return(mismatch_data)
}

mismatch_data <- prepare_data(mismatch_data) 



write_parquet(mismatch_data, 'prepared_data_week_8.parquet')



