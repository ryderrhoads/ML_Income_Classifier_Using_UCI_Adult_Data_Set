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

# Clear environment, console, and plot pane to ensure a clean workspace
rm(list = ls())
cat("\014")
graphics.off()

# Turn off scientific notation for better readability of numbers
options(scipen = 999)

# Load required packages using pacman for convenience
pacman::p_load(data.table)

# Set a seed for reproducibility of random processes
set.seed(418518)

#####################
# Problem 1
#####################

# Load necessary library for data manipulation
library(dplyr)

#################
# Question (i)
#################

# Load the dataset for analysis
data <- read.csv("ECON_418-518_HW3_Data.csv")

# Drop specified columns to simplify the dataset and focus on relevant variables
cols_to_drop <- c("fnlwgt", "occupation", "relationship", "capital.gain", "capital.loss", "educational.num")
data <- data %>% select(-all_of(cols_to_drop))

#################
# Question (ii)
#################

##############
# Part (a)
##############

# Convert the "income" column to a binary indicator (1 for ">50K", 0 otherwise)
data <- data %>%
  mutate(income = ifelse(income == ">50K", 1, 0))

##############
# Part (b)
##############

# Convert the "race" column to a binary indicator (1 for "White", 0 otherwise)
data <- data %>%
  mutate(race = ifelse(race == "White", 1, 0))

##############
# Part (c)
##############

# Convert the "gender" column to a binary indicator (1 for "Male", 0 otherwise)
data <- data %>%
  mutate(gender = ifelse(gender == "Male", 1, 0))

##############
# Part (d)
##############

# Convert the "workclass" column to a binary indicator (1 for "Private", 0 otherwise)
data <- data %>%
  mutate(workclass = ifelse(workclass == "Private", 1, 0))

##############
# Part (e)
##############

# Convert the "native-country" column to a binary indicator (1 for "United-States", 0 otherwise)
data <- data %>%
  mutate(`native.country` = ifelse(`native.country` == "United-States", 1, 0))

##############
# Part (f)
##############

# Convert the "marital-status" column to a binary indicator (1 for "Married-civ-spouse", 0 otherwise)
data <- data %>%
  mutate(`marital.status` = ifelse(`marital.status` == "Married-civ-spouse", 1, 0))

##############
# Part (g)
##############

# Convert the "education" column to a binary indicator (1 for advanced degrees, 0 otherwise)
education_levels <- c("Bachelors", "Masters", "Doctorate")
data <- data %>%
  mutate(education = ifelse(education %in% education_levels, 1, 0))

##############
# Part (h)
##############

# Create a new "age squared" variable to capture non-linear effects of age
data <- data %>%
  mutate(`age.sq` = age^2)

##############
# Part (i)
##############

# Standardize specified numeric variables for easier comparison and modeling
standardize <- function(x) (x - mean(x, na.rm = TRUE)) / sd(x, na.rm = TRUE)
data <- data %>%
  mutate(across(c("age", "age.sq", "hours.per.week"), standardize))

#################
# Question (iii)
#################

##############
# Part (a)
##############

# Calculate and display the proportion of individuals with income > $50K
prop_income_gt_50k <- mean(data$income == 1, na.rm = TRUE)
cat("Proportion of income > $50k:", prop_income_gt_50k, "\n")

##############
# Part (b)
##############

# Calculate and display the proportion of individuals working in the private sector
prop_private_sector <- mean(data$workclass == 1, na.rm = TRUE)
cat("Proportion in private sector:", prop_private_sector, "\n")

##############
# Part (c)
##############

# Calculate and display the proportion of individuals who are married
prop_married <- mean(data$`marital.status` == 1, na.rm = TRUE)
cat("Proportion married:", prop_married, "\n")

##############
# Part (d)
##############

# Calculate and display the proportion of females
prop_females <- mean(data$gender == 0, na.rm = TRUE)
cat("Proportion females:", prop_females, "\n")

##############
# Part (e)
##############

# Display total observations and count of missing values in the dataset
total_observations <- nrow(data)
total_nas <- sum(is.na(data))
cat("Total observations:", total_observations, "\n")
cat("Total NAs:", total_nas, "\n")

##############
# Part (f)
##############

# Convert the "income" column to a factor for use in classification models
data$income <- as.factor(data$income)

#################
# Question (iv)
#################

##############
# Part (a)
##############

# Determine the index for splitting the dataset into training and testing sets (70% training)
last_train_index <- floor(nrow(data) * 0.70)

##############
# Part (b)
##############

# Create the training dataset using the first 70% of the observations
train_data <- data[1:last_train_index, ]

##############
# Part (c)
##############

# Create the testing dataset using the remaining 30% of the observations
test_data <- data[(last_train_index + 1):nrow(data), ]

#################
# Question (v)
#################

##############
# Part (b)
##############

# Use the caret package to perform Lasso regression with cross-validation
library(caret)

# Define a grid of lambda values to tune the Lasso model
lambda_grid <- 10^(seq(5, -2, length = 50))

# Train a Lasso regression model using 10-fold cross-validation
lasso_model <- train(
  income ~ .,  # Predict income using all other variables
  data = train_data,
  method = "glmnet",  # Use the glmnet method for Lasso
  trControl = trainControl(method = "cv", number = 10),  # Cross-validation settings
  tuneGrid = expand.grid(alpha = 1, lambda = lambda_grid)  # Specify Lasso (alpha = 1) with grid search over lambda
)

##############
# Part (c)
##############

# Extract the best lambda and maximum classification accuracy from the Lasso results
best_lambda <- lasso_model$bestTune$lambda
accuracy <- max(lasso_model$results$Accuracy)
cat("Best λ for Lasso:", best_lambda, "\n")
cat("Classification accuracy:", accuracy, "\n")

##############
# Part (d)
##############

# Identify non-zero coefficients in the Lasso model (features selected by the model)
coef_matrix <- as.matrix(coef(lasso_model$finalModel, s = best_lambda))
non_zero_vars <- rownames(coef_matrix)[coef_matrix != 0]
selected_vars <- non_zero_vars[non_zero_vars != "(Intercept)"]  # Exclude the intercept
cat("Non-zero coefficient variables:", selected_vars, "\n")

##############
# Part (e)
##############

# Add an additional variable ("age squared") to the list of selected variables
selected_vars <- c(selected_vars, "age.sq")

# Construct a formula for the refit model using only selected variables
formula <- as.formula(paste("income ~", paste(selected_vars, collapse = " + ")))

# Convert the income column to a factor in the training data (if not already done)
train_data$income <- as.factor(train_data$income)

# Refit the Lasso model using the selected variables and cross-validation
lasso_refit <- train(
  formula,  # Use the refined formula
  data = train_data,
  method = "glmnet",
  trControl = trainControl(method = "cv", number = 10, verboseIter = TRUE),
  tuneGrid = expand.grid(alpha = 1, lambda = lambda_grid)
)

# Extract the best lambda and accuracy from the refitted Lasso model
best_lambda_lasso <- lasso_refit$bestTune$lambda
lasso_accuracy <- max(lasso_refit$results$Accuracy)
cat("Best Lambda for Lasso:", best_lambda_lasso, "\n")
cat("Best Classification Accuracy for Lasso:", lasso_accuracy, "\n")

# Refit a Ridge regression model for comparison
ridge_refit <- train(
  formula,  # Use the same refined formula
  data = train_data,
  method = "glmnet",
  trControl = trainControl(method = "cv", number = 10, verboseIter = TRUE),
  tuneGrid = expand.grid(alpha = 0, lambda = lambda_grid)  # Specify Ridge (alpha = 0)
)

# Extract the best lambda and accuracy from the Ridge model
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

# Train Random Forest models with varying numbers of trees
library(randomForest)

rf_models <- list()  # Initialize a list to store models
tree_counts <- c(100, 200, 300)  # Define different tree counts to test

# Loop over tree counts and train Random Forest models
for (trees in tree_counts) {
  rf_models[[as.character(trees)]] <- randomForest(
    income ~ .,  # Predict income using all other variables
    data = train_data,
    mtry = 5,  # Number of features to sample at each split
    ntree = trees  # Number of trees in the forest
  )
}

##############
# Part (c)
##############

# Evaluate the accuracy of each Random Forest model on the training data
rf_accuracies <- sapply(rf_models, function(model) {
  predictions <- predict(model, train_data)  # Predict on training data
  mean(predictions == train_data$income)  # Calculate accuracy
})

# Identify the model with the highest training accuracy
best_rf_model <- rf_models[[which.max(rf_accuracies)]]
best_rf_accuracy <- max(rf_accuracies)
cat("Best RF accuracy on training data:", best_rf_accuracy, "\n")

##############
# Part (d)
##############

# Test the best Random Forest model on the testing data
test_predictions <- predict(best_rf_model, test_data)
test_accuracy <- mean(test_predictions == test_data$income)
cat("Best RF accuracy on testing data:", test_accuracy, "\n")

# Generate and display confusion matrices for training and testing data
train_confusion <- confusionMatrix(predict(best_rf_model, train_data), train_data$income)
print(train_confusion)

test_confusion <- confusionMatrix(test_predictions, test_data$income)
print(test_confusion)

#################
# Question (vii)
#################

# Reassess the accuracy of the best Random Forest model on the testing data
test_predictions <- predict(best_rf_model, test_data)
test_accuracy <- mean(test_predictions == test_data$income)  # Classification accuracy
cat("Best Model Accuracy on Testing Data:", test_accuracy, "\n")

# Generate the confusion matrix for the testing data
test_conf_matrix <- confusionMatrix(test_predictions, test_data$income)
print(test_conf_matrix)
