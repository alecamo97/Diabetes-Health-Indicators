#Libraries
library(ggplot2)
library(tidyr)
library(caret)
library(randomForest)
library(e1071)
library(class)
library(FactoMineR)
library(reshape2)

#install.packages("randomForest")
#install.packages("e1071")
#install.packages("class")
#install.packages("FactoMineR")
#install.packages("reshape2")

# --------------------------------------------------------------------------------------
# Loading data
database <- read.csv("~/Université Jean Monnet/Semester 2/Data Mining/Project/data.csv")
# Original dataset has 86.07% belonging to the negative class

# -------------------------------------------------------------------------------------------
# SUPERVISED MACHINE LEARNING - NO PREPROCESSING

set.seed(123)

predictors <- c("HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke", "HeartDiseaseorAttack",
                "PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost",
                "GenHlth", "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income")
outcome <- "Diabetes_binary"

# Train/test split (80% train, 20% test)
train_index <- sample(nrow(database), 0.8 * nrow(database))
train_data <- database[train_index, ]
test_data <- database[-train_index, ]

# Outcome variable to factor
train_data[[outcome]] <- as.factor(train_data[[outcome]])
test_data[[outcome]] <- as.factor(test_data[[outcome]])

# Function to calculate F1-score
calculate_f1_score <- function(true_labels, predicted_labels) {
  # Confusion matrix
  confusion_matrix <- table(true_labels, predicted_labels)
  
  # Precision, recall, and F1 score
  precision <- confusion_matrix[2, 2] / sum(confusion_matrix[, 2])
  recall <- confusion_matrix[2, 2] / sum(confusion_matrix[2, ])
  f1_score <- 2 * (precision * recall) / (precision + recall)
  
  return(f1_score)
}

# -----------------------------------------------------------------------------------
# 3 different classifiers will be created, Random Forest, Logistic Regression and KNN

# 1. Random Forest
rf_time <- system.time({
  rf_model <- randomForest(train_data[, predictors], train_data[[outcome]], ntree = 100, mtry = 4)
  rf_predictions <- predict(rf_model, test_data[, predictors])
})

rf_accuracy <- mean(rf_predictions == test_data[[outcome]])
rf_f1_score <- calculate_f1_score(test_data[[outcome]], rf_predictions)

print(paste("Random Forest Accuracy:", round(rf_accuracy, 4)))
print(paste("Random Forest F1 Score:", round(rf_f1_score, 4)))
print(paste("Random Forest Execution Time:", rf_time[["elapsed"]], "seconds"))

# 2. Logistic Regression
logistic_time <- system.time({
  logistic_model <- glm(as.formula(paste(outcome, "~ .")), data = train_data, family = "binomial")
  logistic_predictions <- predict(logistic_model, test_data, type = "response")
  logistic_predictions <- ifelse(logistic_predictions > 0.5, 1, 0)
})

logistic_accuracy <- mean(logistic_predictions == test_data[[outcome]])
logistic_f1_score <- calculate_f1_score(test_data[[outcome]], logistic_predictions)

print(paste("Logistic Regression Accuracy:", round(logistic_accuracy, 4)))
print(paste("Logistic Regression F1 Score:", round(logistic_f1_score, 4)))
print(paste("Logistic Regression Execution Time:", logistic_time[["elapsed"]], "seconds"))

# 3. k-Nearest Neighbors
knn_time <- system.time({
  knn_model <- knn(train_data[, predictors], test_data[, predictors], train_data[[outcome]], k = 5)
})

knn_accuracy <- mean(knn_model == test_data[[outcome]])
knn_f1_score <- calculate_f1_score(test_data[[outcome]], knn_model)

print(paste("k-Nearest Neighbors Accuracy:", round(knn_accuracy, 4)))
print(paste("k-Nearest Neighbors F1 Score:", round(knn_f1_score, 4)))
print(paste("k-Nearest Neighbors Execution Time:", knn_time[["elapsed"]], "seconds"))

#---------------------------------------------------------------------------------------------
# DATA ANALYSIS AND PROCESSING

# This part consists of calculating outliers, calculating correlation matrix, removing
# outliers using Z-score and balancing dataset

#--------------------------------------------------
#Box plot visualization to visually check outliers
# Outliers will be checked only in numerical features

# Divide features into binary and numerical
numerical_features <- c("BMI", "GenHlth", "MentHlth", "PhysHlth", "Age", "Education", "Income")

binary_features <- c("HighBP", "HighChol", "CholCheck", "Smoker", "Stroke", "HeartDiseaseorAttack",
                     "PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", 
                     "DiffWalk","Sex")

# Dataframe to store the numerical features
numerical_data <- database[, numerical_features]
melted_data <- gather(numerical_data, key = "variable", value = "value")

# Box plots for numerical features
ggplot(melted_data, aes(x = "", y = value)) +
  geom_boxplot() +
  facet_wrap(~ variable, scales = "free_y") +
  labs(x = NULL, y = "Value", title = "Box Plots of Numerical Features") +
  theme(axis.text.x = element_blank(), axis.ticks.x = element_blank())

#------------------------------------------------------------------------------
# Correlation matrix for numerical and binary variables

# 1. Numerical
# Extract the numerical features from the dataset
numerical_data <- database[numerical_features]

# Pearson correlation coefficient matrix
pearson_matrix <- cor(numerical_data)

# Convert the correlation matrix to a data frame
corr_df <- as.data.frame(pearson_matrix)

# Convert row names to a new column
corr_df$features <- rownames(corr_df)
rownames(corr_df) <- NULL

# Reshape the dataframe for plotting
corr_melted <- melt(corr_df, id.vars = "features")

# Heatmap plot
ggplot(data = corr_melted, aes(x = features, y = variable, fill = value, label = round(value, 2))) +
  geom_tile(alpha = 0.5) +  # Adjust transparency
  geom_text(color = "black") +  # Add correlation values
  scale_fill_gradient(low = "red", high = "green") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(title = "Pearson Correlation Coefficient Matrix Heatmap")

# 2. Binary
# Phi coefficient matrix for binary variables
phi_matrix <- cor(database[, binary_features], method = "pearson")

# Phi matrix to dataframe
phi_df <- as.data.frame(phi_matrix)
phi_df$Var1 <- rownames(phi_df)

# Melt data frame for ggplot
melted_phi <- melt(phi_df, id.vars = "Var1")

# Heatmap plot
ggplot(melted_phi, aes(x = Var1, y = variable, fill = value, label = round(value, 2))) +
  geom_tile(alpha = 0.5) +
  geom_text(size = 3) +
  scale_fill_gradient(low = "red", high = "green") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1),
        axis.text.y = element_text(size = 8),
        plot.title = element_text(hjust = 0.5)) +
  labs(title = "Correlation Heatmap of Binary Variables")

#---------------------------------------------------------------------------------------
#Remove outliers using Z-score

# Removing outliers with Z-score
remove_outliers <- function(numerical_data, numerical_features, threshold = 3) {
  # Calculate Z scores for each numerical feature
  z_scores <- apply(numerical_data[, numerical_features], 2, function(x) abs((x - mean(x)) / sd(x)))
  
  # Identify rows with any Z score greater than 3
  outlier_rows <- apply(z_scores, 1, function(row_z) any(row_z > threshold))
  
  # Remove outlier rows
  cleaned_data <- numerical_data[!outlier_rows, ]
  
  return(cleaned_data)
}

# Remove outliers from the numerical features in the database using Z-score method
cleaned_database <- remove_outliers(database, numerical_features)

# Print dimension before and after the cleaning
print(paste("Original database dimensions:", nrow(database), "rows,", ncol(database), "columns"))
print(paste("Cleaned database dimensions:", nrow(cleaned_database), "rows,", ncol(cleaned_database), "columns"))

#----------------------------------------------------------------------------------------------
# Balancing the dataset by downsampling tha majority class.

# Check class distribution
class_distribution <- table(cleaned_database$Diabetes_binary)
print(class_distribution)

# Number of positive and negative instances in the dataset
count_0 <- sum(cleaned_database$Diabetes_binary == 0)
count_1 <- sum(cleaned_database$Diabetes_binary == 1)

# Identify the minority and majority classes
minority_class <- which.min(class_distribution)
majority_class <- which.max(class_distribution)

# Number of observations in the minority class
minority_count <- min(class_distribution)

# Downsample the majority class to match the number of instances in the minority class
downsampled_majority_indices <- sample(which(cleaned_database$Diabetes_binary == 0), minority_count, replace = FALSE)

# Combine indices of minority and downsampled majority class
balanced_indices <- c(which(cleaned_database$Diabetes_binary == 1), downsampled_majority_indices)

# Create balanced dataset
balanced_database <- cleaned_database[balanced_indices, ]

# Shuffle the rows of the balanced dataset
balanced_database <- balanced_database[sample(nrow(balanced_database)), ]

# Check class distribution in the balanced dataset
balanced_class_distribution <- table(balanced_database$Diabetes_binary)
print(balanced_class_distribution)

# Print number of instances for each class
print(paste("Number of instances where diabetes_binary = 0:", minority_count))
print(paste("Number of instances where diabetes_binary = 1:", minority_count))

#---------------------------------------------------------------------------------------------
# With the new balanced dataset, the same 3 classifiers will be computed and the performance
# between balanced and imabalanced dataset will be compared


# Train/test balanced dataset

train_index <- sample(nrow(balanced_database), 0.8 * nrow(balanced_database))
train_data <- balanced_database[train_index, ]
test_data <- balanced_database[-train_index, ]

# Outcome variable to factor with two levels
train_data[[outcome]] <- as.factor(train_data[[outcome]])
test_data[[outcome]] <- as.factor(test_data[[outcome]])

# 1. Random Forest
rf_time_2 <- system.time({
  rf_model_2 <- randomForest(train_data[, predictors], train_data[[outcome]], ntree = 100, mtry = 4)
  rf_predictions_2 <- predict(rf_model_2, test_data[, predictors])
})

rf_accuracy_2 <- mean(rf_predictions_2 == test_data[[outcome]])
rf_f1_score_2 <- calculate_f1_score(test_data[[outcome]], rf_predictions_2)

print(paste("Random Forest Accuracy:", round(rf_accuracy_2, 4)))
print(paste("Random Forest F1 Score:", round(rf_f1_score_2, 4)))
print(paste("Random Forest Execution Time:", rf_time_2[["elapsed"]], "seconds"))

# 2. Logistic Regression
logistic_time_2 <- system.time({
  logistic_model_2 <- glm(as.formula(paste(outcome, "~ .")), data = train_data, family = "binomial")
  logistic_predictions_2 <- predict(logistic_model_2, test_data, type = "response")
  logistic_predictions_2 <- ifelse(logistic_predictions_2 > 0.5, 1, 0)
})

logistic_accuracy_2 <- mean(logistic_predictions_2 == test_data[[outcome]])
logistic_f1_score_2 <- calculate_f1_score(test_data[[outcome]], logistic_predictions_2)

print(paste("Logistic Regression Accuracy:", round(logistic_accuracy_2, 4)))
print(paste("Logistic Regression F1 Score:", round(logistic_f1_score_2, 4)))
print(paste("Logistic Regression Execution Time:", logistic_time_2[["elapsed"]], "seconds"))

# 3. k-Nearest Neighbors
knn_time_2 <- system.time({
  knn_model_2 <- knn(train_data[, predictors], test_data[, predictors], train_data[[outcome]], k = 5)
})

knn_accuracy_2 <- mean(knn_model_2 == test_data[[outcome]])
knn_f1_score_2 <- calculate_f1_score(test_data[[outcome]], knn_model_2)

print(paste("k-Nearest Neighbors Accuracy:", round(knn_accuracy_2, 4)))
print(paste("k-Nearest Neighbors F1 Score:", round(knn_f1_score_2, 4)))
print(paste("k-Nearest Neighbors Execution Time:", knn_time_2[["elapsed"]], "seconds"))