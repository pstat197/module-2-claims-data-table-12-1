library(tidyverse)
library(rvest)
library(qdapRegex)

# paragraph parser
parse_fn_paragraphs <- function(.html){
  read_html(.html) %>%
    html_elements("p") %>% 
    html_text2() %>%
    str_c(collapse = " ") %>%
    rm_url() %>%
    rm_email() %>%
    str_remove_all("'") %>%
    str_replace_all(
      paste(c("\n", "[[:punct:]]", "nbsp", "[[:digit:]]", "[[:symbol:]]"),
            collapse = "|"),
      " "
    ) %>%
    tolower() %>%
    str_squish()
}

parse_data_paragraphs <- function(.df){
  .df %>%
    filter(str_detect(text_tmp, "<!")) %>%
    rowwise() %>%
    mutate(text_clean = parse_fn_paragraphs(text_tmp)) %>%
    unnest(text_clean)
}

# load raw data + preprocessing 
load("data/claims-raw.RData")

claims_clean <- claims_raw %>%
  parse_data_paragraphs()

dim(claims_clean)
head(claims_clean$text_clean)
table(claims_clean$bclass)

# training split
library(tidymodels)
set.seed(12345)

partitions <- claims_clean %>% 
  initial_split(prop= 0.8)

train_labels <- training(partitions) %>% pull(bclass) %>% as.numeric() - 1

test_text <- testing(partitions) %>% pull(text_clean)
test_labels <- testing(partitions) %>% pull(bclass) %>% as.numeric() - 1

library(irlba)
library(nnet)

# tf-idf matrix

parsed <- claims_clean %>% 
  select(.id, bclass, text_clean)

# tokenize and get TF-IDF using nlp file and preprocessing
source("scripts/preprocessing.R")
tokens <- nlp_fn(parsed)

# model matrix
model_df <- tokens %>%
  mutate(bclass = factor(bclass)) %>%
  select(-.id)

y <- model_df$bclass
X <- model_df %>% select(-bclass)

# drop zero-variance features
X <- X[, colSums(X) > 0]

# pca same as task 1 
set.seed(123)
pca_fit <- prcomp_irlba(X, n = 50, center = TRUE, scale. = TRUE)

pc_scores <- as_tibble(pca_fit$x) %>%
  mutate(
    .id = tokens$.id,   # KEEP ID FOR ALIGNMENT
    bclass = y
  )

# train/test split
train_ids <- training(partitions)$.id
test_ids  <- testing(partitions)$.id

train_pc <- pc_scores %>%
  filter(.id %in% train_ids) %>%
  select(-.id)

test_pc <- pc_scores %>%
  filter(.id %in% test_ids) %>%
  select(-.id)


# nnet neural network
nn_model <- nnet(
  bclass ~ ., 
  data = train_pc,
  size = 5,     # hidden units
  maxit = 200,  # iterations
  decay = 1e-3  # regularization
)

# predictions
nn_prob <- predict(nn_model, newdata = test_pc, type = "raw")
nn_pred <- ifelse(nn_prob > 0.5, levels(y)[2], levels(y)[1]) %>% 
  factor(levels = levels(y))

# accuracy
accuracy_nn <- mean(nn_pred == test_pc$bclass)
accuracy_nn

# 0.6366