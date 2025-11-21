library(tidymodels)
library(tidyverse)
load("./data/claims-raw.RData")
source('scripts/preprocessing.R')
claims_clean <- claims_raw %>%
  parse_data()
claims <- claims_clean %>% select(.id, bclass, mclass, text_clean)

# BINARY CLASSIFICATION
# training, validation, and testing split
set.seed(42)
partitions <- claims %>%
  initial_split(prop = 0.8, strata = bclass)

# Extract training data
claims_train <- training(partitions)
train_text <- claims_train %>%
  pull(text_clean)
train_labels <- claims_train %>%
  pull(bclass) %>%
  as.numeric() - 1

# Extract test data
claims_test <- testing(partitions)
test_text <- claims_test %>%
  pull(text_clean)
test_labels <- claims_test %>%
  pull(bclass) %>%
  as.numeric() - 1

claims_folds <- vfold_cv(claims_train, v = 5, strata = bclass)
# setting up recipe and model
claim_recipe <- recipe()

knn_spec <- nearest_neighbor(neighbors = tune()) %>% 
  set_mode("classification") %>% 
  set_engine("kknn")

knn_wf <- workflow() %>% 
  add_model(knn_spec) %>% 
  add_recipe(claims_recipe)

# tuning KNN
knn_grid <- grid_regular(neighbors(range = c(1, 10)),
                         levels = 10)
load("tune_knn.rda")
tune_knn <- tune_grid(knn_wf,
                     resamples = earthquakes_folds,
                     grid = knn_grid,
                     metrics = metric_set(accuracy))

autoplot(tune_knn)

best_knn <- select_best(tune_knn, metric = "accuracy")

knn_final_wf <- finalize_workflow(knn_wf, best_knn)

knn_final_fit <- fit(knn_final_wf, data = earthquakes_train)

# Predictions
knn_preds <- augment(knn_final_fit, earthquakes_test)
r3 <- roc_curve(knn_preds, tsunami, .pred_1, event_level = "second") %>% 
  autoplot()

c3 <- conf_mat(knn_preds, truth = tsunami, .pred_class) %>% 
  autoplot(type = "heatmap")

# export (KEEP THIS FORMAT IDENTICAL)
pred_df <- clean_df %>%
  bind_cols(bclass.pred = pred_classes) %>%
  select(.id, bclass.pred)

save(pred_df, file = 'results/example-preds.RData')
