##################################################
# ECON 418-518 Homework 3
# Ryder Rhoads
# The University of Arizona
# ryderrhoads@arizona.edu 
# 8 December 2024
###################################################


#####################
# Preliminaries
#####################

# Clear environment, console, and plot pane
rm(list = ls())
cat("\014")
graphics.off()

# Turn off scientific notation
options(scipen = 999)

# Load packages
pacman::p_load(data.table)

# Set seed
set.seed(418518)

#####################
# Problem 1
#####################

# Load necessary library
library(dplyr)

#################
# Question (i)
#################

# Load data
data <- read.csv("ECON_418-518_HW3_Data.csv")

# Drop specified columns (adjusting column names to match actual names)
cols_to_drop <- c("fnlwgt", "occupation", "relationship", "capital.gain", "capital.loss", "educational.num")
data <- data %>% select(-all_of(cols_to_drop))

#################
# Question (ii)
#################

##############
# Part (a)
##############

# Convert "income" column to binary indicator
data <- data %>%
  mutate(income = ifelse(income == ">50K", 1, 0))

##############
# Part (b)
##############

# Convert "race" column to binary indicator
data <- data %>%
  mutate(race = ifelse(race == "White", 1, 0))

##############
# Part (c)
##############

# Convert "gender" column to binary indicator
data <- data %>%
  mutate(gender = ifelse(gender == "Male", 1, 0))

##############
# Part (d)
##############

# Convert "workclass" column to binary indicator
data <- data %>%
  mutate(workclass = ifelse(workclass == "Private", 1, 0))

##############
# Part (e)
##############

# Convert "native-country" column to binary indicator
data <- data %>%
  mutate(`native.country` = ifelse(`native.country` == "United-States", 1, 0))

##############
# Part (f)
##############

# Convert "marital-status" column to binary indicator
data <- data %>%
  mutate(`marital.status` = ifelse(`marital.status` == "Married-civ-spouse", 1, 0))

##############
# Part (g)
##############

# Convert "education" column to binary indicator
education_levels <- c("Bachelors", "Masters", "Doctorate")
data <- data %>%
  mutate(education = ifelse(education %in% education_levels, 1, 0))

##############
# Part (h)
##############

# Create an "age sq" variable
data <- data %>%
  mutate(`age.sq` = age^2)

##############
# Part (i)
##############

standardize <- function(x) (x - mean(x, na.rm = TRUE)) / sd(x, na.rm = TRUE)
data <- data %>%
  mutate(across(c("age", "age.sq", "hours.per.week"), standardize))

#################
# Question (iii)
#################

##############
# Part (a)
##############

prop_income_gt_50k <- mean(data$income == 1, na.rm = TRUE)
cat("Proportion of income > $50k:", prop_income_gt_50k, "\n")

##############
# Part (b)
##############

prop_private_sector <- mean(data$workclass == 1, na.rm = TRUE)
cat("Proportion in private sector:", prop_private_sector, "\n")

##############
# Part (c)
##############

prop_married <- mean(data$`marital.status` == 1, na.rm = TRUE)
cat("Proportion married:", prop_married, "\n")

##############
# Part (d)
##############

prop_females <- mean(data$gender == 0, na.rm = TRUE)
cat("Proportion females:", prop_females, "\n")

##############
# Part (e)
##############
total_observations <- nrow(data)
total_nas <- sum(is.na(data))
cat("Total observations:", total_observations, "\n")
cat("Total NAs:", total_nas, "\n")


##############
# Part (f)
##############

# Convert "income" column to a factor
data$income <- as.factor(data$income)

#################
# Question (iv)
#################

##############
# Part (a)
##############

# Determine the index for the last observation in the training set
last_train_index <- floor(nrow(data) * 0.70)

##############
# Part (b)
##############

# Create training data: first row to last_train_index
train_data <- data[1:last_train_index, ]

##############
# Part (c)
##############

# Create testing data: from last_train_index + 1 to the end
test_data <- data[(last_train_index + 1):nrow(data), ]


#################
# Question (v)
#################

##############
# Part (b)
##############

library(caret)
lambda_grid <- 10^(seq(5, -2, length = 50))

lasso_model <- train(
  income ~ .,
  data = train_data,
  method = "glmnet",
  trControl = trainControl(method = "cv", number = 10),
  tuneGrid = expand.grid(alpha = 1, lambda = lambda_grid)
)

##############
# Part (c)
##############

best_lambda <- lasso_model$bestTune$lambda
accuracy <- max(lasso_model$results$Accuracy)
cat("Best λ for Lasso:", best_lambda, "\n")
cat("Classification accuracy:", accuracy, "\n")

##############
# Part (d)
##############

coef_matrix <- as.matrix(coef(lasso_model$finalModel, s = best_lambda))
non_zero_vars <- rownames(coef_matrix)[coef_matrix != 0]
selected_vars <- non_zero_vars[non_zero_vars != "(Intercept)"]
cat("Non-zero coefficient variables:", selected_vars, "\n")

##############
# Part (e)
##############

selected_vars <- c(selected_vars, "age.sq")
formula <- as.formula(paste("income ~", paste(selected_vars, collapse = " + ")))
train_data$income <- as.factor(train_data$income)

# Lasso Refit
lasso_refit <- train(
  formula,
  data = train_data,
  method = "glmnet",
  trControl = trainControl(method = "cv", number = 10, verboseIter = TRUE),
  tuneGrid = expand.grid(alpha = 1, lambda = lambda_grid)
)
# Best tuning parameters for Lasso
best_lambda_lasso <- lasso_refit$bestTune$lambda
lasso_accuracy <- max(lasso_refit$results$Accuracy)

cat("Best Lambda for Lasso:", best_lambda_lasso, "\n")
cat("Best Classification Accuracy for Lasso:", lasso_accuracy, "\n")

# Ridge Refit
ridge_refit <- train(
  formula,
  data = train_data,
  method = "glmnet",
  trControl = trainControl(method = "cv", number = 10, verboseIter = TRUE),
  tuneGrid = expand.grid(alpha = 0, lambda = lambda_grid)
)

# Best tuning parameters for Ridge
best_lambda_ridge <- ridge_refit$bestTune$lambda
ridge_accuracy <- max(ridge_refit$results$Accuracy)

cat("Best Lambda for Ridge:", best_lambda_ridge, "\n")
cat("Best Classification Accuracy for Ridge:", ridge_accuracy, "\n")

#################
# Question (vi)
#################

##############
# Part (b)
##############

library(randomForest)

rf_models <- list()
tree_counts <- c(100, 200, 300)

for (trees in tree_counts) {
  rf_models[[as.character(trees)]] <- randomForest(
    income ~ .,
    data = train_data,
    mtry = 5,  # Number of features to sample at each split
    ntree = trees  # Number of trees in the forest
  )
}



##############
# Part (c)
##############

rf_accuracies <- sapply(rf_models, function(model) {
  predictions <- predict(model, train_data)
  mean(predictions == train_data$income)  # Compute accuracy
})

# Identify the best model
best_rf_model <- rf_models[[which.max(rf_accuracies)]]
best_rf_accuracy <- max(rf_accuracies)

cat("Best RF accuracy on training data:", best_rf_accuracy, "\n")

##############
# Part (d)
##############

test_predictions <- predict(best_rf_model, test_data)
test_accuracy <- mean(test_predictions == test_data$income)

cat("Best RF accuracy on testing data:", test_accuracy, "\n")

# Confusion matrix for training data
train_confusion <- confusionMatrix(predict(best_rf_model, train_data), train_data$income)
print(train_confusion)

# Confusion matrix for testing data
test_confusion <- confusionMatrix(test_predictions, test_data$income)
print(test_confusion)

##############
# Question (vii)
##############

# Best Model Accuracy Evaluation
test_predictions <- predict(best_rf_model, test_data)  # Predict on testing data
test_accuracy <- mean(test_predictions == test_data$income)  # Classification accuracy

cat("Best Model Accuracy on Testing Data:", test_accuracy, "\n")

# Confusion Matrix
test_conf_matrix <- confusionMatrix(test_predictions, test_data$income)
print(test_conf_matrix)
zz