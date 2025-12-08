library(tidyverse)
library(tidytext)
library(textstem)
library(stopwords)
library(qdapRegex)
library(irlba)

source('/Users/anjanisethi/Documents/GitHub/module-2-claims-data-table-12-1/scripts/preprocessing.R')

# loading raw HTML
load('/Users/anjanisethi/Documents/GitHub/module-2-claims-data-table-12-1/data/claims-raw.RData')

# using parser from task 1
parsed_par <- parse_data_paragraphs(claims_raw)

# word features from preprocessing script
word_df <- nlp_fn(parsed_par)

# bigram tokenization
bigrams <- parsed_par %>% 
  unnest_tokens(
    output = 'bigram',
    input = text_clean,
    token = 'ngrams',
    n = 2
  ) %>% 
  separate(bigram, into = c('w1', 'w2'), sep = ' ') %>% 
  filter(!str_detect(w1, '^[[:punct:]]+$')) %>% 
  filter(!str_detect(w2, '^[[:punct:]]+$')) %>% 
  mutate(bigram = paste(w1, w2, sep = '_')) %>% 
  count(.id, bclass, bigram, name = 'n')

# bigram tf-idf
bigram_tfidf <- bigrams %>% 
  bind_tf_idf(
    term = bigram,
    document = .id,
    n = n
  ) %>% 
  arrange(desc(tf-idf)) %>% 
  pivot_wider(
    id_cols = c(.id, bclass),
    names_from = bigram,
    values_from = tf_idf,
    values_fill = 0
  )

# align word and bigram features
word_df <- word_df %>% arrange(.id)
bigram_tfidf <- bigram_tfidf %>% arrange(.id)

# keep pages present in both
merged <- inner_join(word_df, bigram_tfidf, by = c('.id', 'bclass'), suffix =
                       c('_word', '_bigram'))

# prepare design matrices
y <- factor(merged$bclass)
X_words <- merged %>% select(ends_with('_word'))
X_bigrams <- merged %>% select(ends_with('_bigram'))

# dropping columns with zero variance
X_words <- X_words[, colSums(X_words) > 0]
X_bigrams <- X_bigrams[, colSums(X_bigrams) > 0]
dim(X_words)
dim(X_bigrams)


# word components can't exceed min(nrow, ncol) - 1
max_comp_words <- min(50, nrow(X_words) - 1, ncol(X_words) - 1)
ncomp_words <- min(50, ncol(X_words))

# pca word features
set.seed(123)
pca_words <- prcomp(X_words, center = TRUE, scale. = TRUE)

scores_words <- as.tibble(pca_words$x[, 1:ncomp_words])

max_comp_bigrams <- min(50, nrow(X_bigrams) - 1, ncol(X_bigrams) - 1)

# pca bigram features
if (ncol(X_bigrams) >= 2) {
  ncomp_bigrams <- min(50, ncol(X_bigrams))
  
  pca_bigrams <- prcomp(
    X_bigrams,
    center = TRUE,
    scale. = TRUE
  )
  
  scores_bigrams <- as_tibble(pca_bigrams$x[, 1:ncomp_bigrams])
} else {
  # no usable bigram structure → no bigram PCs
  scores_bigrams <- NULL
}

# Combine into one PC data frame
pc_df <- tibble(bclass = y) %>%
  bind_cols(scores_words) %>%
  { if (!is.null(scores_bigrams)) bind_cols(., scores_bigrams) else . }

set.seed(123)
n <- nrow(pc_df)
train_idx <- sample(seq_len(n), size = floor(0.7 * n))

train_pc <- pc_df[train_idx, ]
test_pc  <- pc_df[-train_idx, ]

logit_stack <- glm(
  bclass ~ .,
  data = train_pc,
  family = binomial
)

test_prob <- predict(logit_stack, newdata = test_pc, type = "response")

pos <- levels(test_pc$bclass)[2]
neg <- levels(test_pc$bclass)[1]

test_pred <- ifelse(test_prob > 0.5, pos, neg) %>%
  factor(levels = levels(test_pc$bclass))

accuracy_bigrams <- mean(test_pred == test_pc$bclass)
accuracy_bigrams