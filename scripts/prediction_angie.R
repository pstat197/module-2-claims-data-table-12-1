library(tidyverse)
library(glmnet)
library(irlba)

# load models + pca
load("results/final-models-group12-1_angie.RData")

# example: new TF–IDF + PCA matrix X_new (same columns as training X)
# x_new_pc <- predict(pca_fit, newdata = X_new)

# predict binary
# prob_bin_new <- predict(cv_bin, newx = x_new_pc, s = "lambda.min", type = "response")
# class_bin_new <- ifelse(prob_bin_new > 0.5, "claim", "non-claim")

# predict multiclass
# class_multi_new <- predict(cv_multi, newx = x_new_pc, s = "lambda.min", type = "class")
