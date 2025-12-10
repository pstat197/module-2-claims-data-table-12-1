library(tidyverse)

# updating preprocessing functions
source("scripts/preprocessing.R")

# load raw HTML data
load("data/claims-raw.RData")

glimpse(claims_raw)

# parse HTML + clean text
parsed <- parse_data(claims_raw)
glimpse(parsed)

# tokenize + TF-IDF
claims_clean <- nlp_fn(parsed)
glimpse(claims_clean)

# pca preparation
model_df <- claims_clean %>%
  mutate(bclass = factor(bclass)) %>%
  select(-.id)

y <- model_df$bclass
X <- model_df %>% select(-bclass)

# pca
set.seed(123)
pca_fit <- prcomp(X, center = TRUE, scale. = TRUE)
K <- 50
pc_scores <- as_tibble(pca_fit$x[,1:K]) %>% 
  mutate(bclass = y)

# train/test split
set.seed(123)
n <- nrow(pc_scores)
train_idx <- sample(seq_len(n), size = floor(0.7*n))

train_pc <- pc_scores[train_idx,]
test_pc <- pc_scores[-train_idx,]

# fit logistic regression (pcr)
logit_pcr <- glm(
  bclass ~ .,
  data = train_pc,
  family = binomial
)

# test accuracy
test_prob <- predict(logit_pcr, newdata = test_pc, type = 'response')
pos_class <- levels(test_pc$bclass)[2]
neg_class <- levels(test_pc$bclass)[1]

test_pred <- ifelse(test_prob > 0.5, pos_class, neg_class) %>% 
  factor(levels = levels(test_pc$bclass))

accuracy_headers <- mean(test_pred == test_pc$bclass)
accuracy_headers

# accuracy: 0.5937