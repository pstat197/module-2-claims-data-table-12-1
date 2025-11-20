
# Script to illustrate how to use the deployed SVM models
# Usage: Rscript scripts/deploy_svm.R

library(tidyverse)
library(e1071)
library(tidytext)
library(tokenizers)
library(textstem)
library(rvest)
library(qdapRegex)
library(stopwords)

# Load models
if(!file.exists("results/svm-binary.rds") || !file.exists("results/svm-multiclass.rds")) {
  stop("Models not found in results/ directory. Please run primaryTaskSVM.Rmd first.")
}

svm_binary <- readRDS("results/svm-binary.rds")
svm_mclass <- readRDS("results/svm-multiclass.rds")

cat("Models loaded successfully.\n")

# Load test data (or new data)
load("data/claims-test.RData")

# Preprocessing function (must match training)
parse_fn <- function(.html) {
  read_html(.html) %>%
    html_elements("p, h1, h2, h3, h4, h5, h6") %>%
    html_text2() %>%
    str_c(collapse = " ") %>%
    rm_url() %>%
    rm_email() %>%
    str_remove_all("'") %>%
    str_replace_all(paste(
      c(
        "\n",
        "[[:punct:]]",
        "nbsp",
        "[[:digit:]]",
        "[[:symbol:]]"
      ),
      collapse = "|"
    ), " ") %>%
    str_replace_all("([a-z])([A-Z])", "\\1 \\2") %>%
    tolower() %>%
    str_replace_all("\\s+", " ")
}

parse_data <- function(.df) {
  out <- .df %>%
    filter(str_detect(text_tmp, "<!")) %>%
    rowwise() %>%
    mutate(text_clean = parse_fn(text_tmp)) %>%
    unnest(text_clean)
  return(out)
}

# Note: For a real deployment, we would need to save the vocabulary and IDF values 
# or the scaling parameters (mean/sd) to apply to new data.
# In primaryTaskSVM.Rmd, we used the training data to scale the test data.
# To make this script truly standalone for *new* data, we would need those parameters saved.
# However, for this deliverable, illustrating use on the *test* data (which we have) is sufficient.
# We will assume the user wants to reproduce the predictions on claims-test.RData.

# Ideally, we should have saved the scaling parameters. 
# Let's update this script to just load the *predictions* if the goal is just to show results,
# OR, if we want to show *how* to predict, we need to replicate the pipeline.
# Since I didn't save the scaling parameters in Rmd, I can't perfectly replicate it here without loading training data again.
# I will modify the Rmd to save the scaling parameters, then update this script.
# But first, let's write this script to just print a message about this limitation or load training data to get parameters.
# Loading training data is safer to ensure correctness.

load("data/claims-raw.RData")

cat("Preprocessing data...\n")
claims_clean <- claims_raw %>% parse_data()
claims_test_clean <- claims_test %>% parse_data()

get_tokens <- function(df) {
  df %>%
    unnest_tokens(
      output = token,
      input = text_clean,
      token = "words",
      stopwords = str_remove_all(stop_words$word, "[[:punct:]]")
    ) %>%
    mutate(token.lem = lemmatize_words(token)) %>%
    filter(str_length(token.lem) > 2)
}

train_tokens <- get_tokens(claims_clean)
test_tokens <- get_tokens(claims_test_clean)

# TF-IDF
train_tfidf <- train_tokens %>%
  count(.id, token.lem, name = "n") %>%
  bind_tf_idf(token.lem, .id, n)

top_terms <- train_tokens %>%
  count(token.lem, sort = TRUE) %>%
  slice_head(n = 1000) %>%
  pull(token.lem)

train_features <- train_tfidf %>%
  filter(token.lem %in% top_terms) %>%
  pivot_wider(id_cols = .id, names_from = token.lem, values_from = tf_idf, values_fill = 0) %>%
  left_join(claims_clean %>% select(.id, bclass, mclass), by = ".id")

test_features <- test_tokens %>%
  filter(token.lem %in% top_terms) %>%
  count(.id, token.lem, name = "n") %>%
  bind_tf_idf(token.lem, .id, n) %>%
  pivot_wider(id_cols = .id, names_from = token.lem, values_from = tf_idf, values_fill = 0)

# Align columns
missing_cols <- setdiff(names(train_features), names(test_features))
missing_cols <- setdiff(missing_cols, c("bclass", "mclass"))
for(col in missing_cols) test_features[[col]] <- 0
test_features <- test_features %>%
  select(all_of(names(train_features %>% select(-bclass, -mclass))))

# Handle NAs
all_test_ids <- claims_test_clean$.id
test_features <- tibble(.id = all_test_ids) %>%
  left_join(test_features, by = ".id") %>%
  mutate(across(where(is.numeric), ~replace_na(., 0)))

# Scaling
train_mat <- train_features %>% select(-.id, -bclass, -mclass) %>% as.matrix()
test_mat <- test_features %>% select(-.id) %>% as.matrix()

train_scaled <- scale(train_mat)
train_center <- attr(train_scaled, "scaled:center")
train_scale <- attr(train_scaled, "scaled:scale")
test_scaled <- scale(test_mat, center = train_center, scale = train_scale)
test_scaled[, train_scale == 0] <- 0

test_df_svm <- as.data.frame(test_scaled)

cat("Generating predictions...\n")
binary_preds <- predict(svm_binary, test_df_svm)
mclass_preds <- predict(svm_mclass, test_df_svm)

results <- tibble(
  .id = test_features$.id,
  bclass_pred = binary_preds,
  mclass_pred = mclass_preds
)

print(head(results))
cat("Done.\n")
