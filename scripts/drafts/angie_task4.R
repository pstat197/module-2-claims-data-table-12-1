library(tidyverse)
library(tidymodels)
library(irlba)
library(glmnet)

load('data/claims-raw.RData')
source('scripts/preprocessing.R')

parse_fn_paragraphs <- function(.html){
  page <- tryCatch(read_html(.html), error = function(e) NA)
  if (is.na(page)) return("")
  text <- page %>% html_elements("p") %>% html_text2() %>% str_c(collapse=" ")
  text %>%
    rm_url() %>% rm_email() %>%
    str_replace_all("[[:punct:][:digit:][:symbol:]]", " ") %>%
    tolower() %>% str_squish()
}

parse_data_paragraphs <- function(.df){
  .df %>%
    filter(str_detect(text_tmp, "<!")) %>%
    rowwise() %>%
    mutate(text_clean = parse_fn_paragraphs(text_tmp)) %>%
    ungroup()
}

claims_clean <- parse_data_paragraphs(claims_raw)

# train/test split
set.seed(123)
partitions <- initial_split(claims_clean, prop = 0.8)
train_df <- training(partitions)
test_df  <- testing(partitions)

# tfidf tokenization
parsed <- claims_clean %>% select(.id, bclass, mclass, text_clean)
tokens <- nlp_fn(parsed)

glimpse(parsed)
glimpse(tokens)

tokens <- tokens %>% 
  left_join(parsed %>% select(.id, mclass), by='.id')

# pca from task 1
model_df <- tokens %>%
  mutate(bclass = factor(bclass),
         mclass = factor(mclass))

y_bin  <- model_df$bclass
y_multi <- model_df$mclass

X <- model_df %>% select(-.id, -bclass, -mclass)
X <- X[, colSums(X) > 0]   # drop zero-variance features

# PCA
set.seed(123)
pca_fit <- prcomp_irlba(X, n = 50, center = TRUE, scale. = TRUE)

pc_scores <- as_tibble(pca_fit$x) %>%
  mutate(
    .id = tokens$.id,
    bclass = y_bin,
    mclass = y_multi
  )

# align pca rows
train_ids <- train_df$.id
test_ids  <- test_df$.id

train_pc <- pc_scores %>% filter(.id %in% train_ids) %>% select(-.id)
test_pc <- pc_scores %>% filter(.id %in% test_ids) %>% select(-.id)

# fitting binary classifier
x_train_bin <- as.matrix(train_pc %>% select(-bclass, -mclass))
y_train_bin <- train_pc$bclass

test_ids_final <- pc_scores %>% 
  filter(.id %in% test_ids) %>% 
  pull(.id)

set.seed(123)
cv_bin <- cv.glmnet(x_train_bin, y_train_bin,
                    family = "binomial")

# predictions on test set
x_test_bin <- as.matrix(test_pc %>% select(-bclass, -mclass))
prob_bin <- predict(cv_bin, newx = x_test_bin, s = "lambda.min", type = "response")
pred_bin <- ifelse(prob_bin > 0.5, levels(y_train_bin)[2], levels(y_train_bin)[1])
pred_bin <- factor(pred_bin, levels = levels(y_train_bin))

acc_bin <- mean(pred_bin == test_pc$bclass)
acc_bin

# 0.5572

# fit multiclass classifier
x_train_multi <- as.matrix(train_pc %>% select(-bclass, -mclass))
y_train_multi <- train_pc$mclass

set.seed(123)
cv_multi <- cv.glmnet(
  x_train_multi, y_train_multi,
  family = "multinomial"
)

x_test_multi <- as.matrix(test_pc %>% select(-bclass, -mclass))
pred_multi <- predict(cv_multi, newx = x_test_multi, s = "lambda.min", type = "class")
pred_multi <- factor(pred_multi, levels = levels(y_train_multi))

acc_multi <- mean(pred_multi == test_pc$mclass)
acc_multi

# 0.5796

pred_df <- tibble(
  .id = test_ids_final,
  bclass.pred = pred_bin,
  mclass.pred = pred_multi
)

save(pred_df, file = "results/preds-group12-1_angie.RData") 
save(cv_bin, cv_multi, pca_fit,
     file = "results/final-models-group12-1_angie.RData")

