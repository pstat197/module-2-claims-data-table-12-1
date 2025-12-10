library(tidyverse)
library(tidytext)
library(textstem)
library(rvest)
library(qdapRegex)
library(stopwords)
library(tokenizers)
library(pROC)
library(irlba)

source("scripts/preprocessing.R")

parse_fn_paragraphs <- function(.html){
  
  # safely read HTML
  page <- tryCatch(
    read_html(.html),
    error = function(e) return(NA_character_)
  )
  
  # if read_html failed → return empty string
  if (is.na(page)) return("")
  
  # extract paragraph text
  txt <- page %>%
    html_elements("p") %>%
    html_text2() %>%
    str_c(collapse = " ")
  
  # clean text
  txt %>%
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
    filter(str_detect(text_tmp, "<!")) %>%   # keep only valid HTML
    rowwise() %>%
    mutate(text_clean = parse_fn_paragraphs(text_tmp)) %>%
    ungroup()
}

load("data/claims-raw.RData")   # gives claims_raw

parsed_par <- parse_data_paragraphs(claims_raw)

# reuse your nlp_fn from preprocessing.R for tokens + TF–IDF
claims_clean_par <- nlp_fn(parsed_par)

glimpse(claims_clean_par)

model_df_par <- claims_clean_par %>%
  mutate(bclass = factor(bclass)) %>%
  select(-.id)

y_par <- model_df_par$bclass
X_par <- model_df_par %>% select(-bclass)

# drop all-zero columns
X_par_nz <- X_par[, colSums(X_par) > 0]

# PCA via irlba: first 50 components
set.seed(123)
pca_par <- prcomp_irlba(X_par_nz, n = 50, center = TRUE, scale. = TRUE)

pc_scores_par <- as_tibble(pca_par$x) %>%
  mutate(bclass = y_par)

# train/test split
set.seed(123)
n_par <- nrow(pc_scores_par)
train_idx_par <- sample(seq_len(n_par), size = floor(0.7 * n_par))

train_pc_par <- pc_scores_par[train_idx_par, ]
test_pc_par  <- pc_scores_par[-train_idx_par, ]

# logistic regression on PCs
logit_pcr_par <- glm(
  bclass ~ .,
  data = train_pc_par,
  family = binomial
)

# predictions + accuracy
test_prob_par <- predict(logit_pcr_par, newdata = test_pc_par, type = "response")

pos_class_par <- levels(test_pc_par$bclass)[2]
neg_class_par <- levels(test_pc_par$bclass)[1]

test_pred_par <- ifelse(test_prob_par > 0.5, pos_class_par, neg_class_par) %>%
  factor(levels = levels(test_pc_par$bclass))

accuracy_paragraphs <- mean(test_pred_par == test_pc_par$bclass)
accuracy_paragraphs

# accuracy: 0.6706