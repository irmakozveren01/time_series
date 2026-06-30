
# Women's Clothing E-Commerce Reviews Analysis

library(tm)
library(lsa)
library(topicmodels)
library(skmeans)
library(mclust)
library(wordcloud)
library(RColorBrewer)
library(cluster)

library(tidyverse)
library(tidytext)

set.seed(1234)

# Load the dataset

setwd("~/Desktop/Data Science Applications")
clothing_data <- read.csv("Womens Clothing E-Commerce Reviews.csv", stringsAsFactors = FALSE)

# Filter rows with missing Review Text
df_analysis<- clothing_data %>% 
  filter(!is.na(Review.Text) & Review.Text != "")
n_docs <- nrow(df_analysis)

# Cleaning and Corpus
corpus <- VCorpus(VectorSource(df_analysis$Review.Text))
corpus <- corpus %>%
  tm_map(content_transformer(tolower)) %>%
  tm_map(removePunctuation) %>%
  tm_map(removeNumbers) %>%
  tm_map(removeWords, stopwords("english")) %>%
  tm_map(stripWhitespace) %>% 
  tm_map(removeWords, stopwords("SMART"))


# Add context-specific stopwords
fashion_stopwords <- c("review","order","purchas", "store","retail", "online", "make","made", "work","look","want","time","day","person")
corpus <- tm_map(corpus, removeWords, fashion_stopwords)

# Stemming
corpus <- tm_map(corpus, stemDocument, language = "english")

fashion_stopwords_stemmed <- c(
  "review", "order", "purchas", "store", "retail", "onlin",
  "make", "made", "work", "look", "want", "time", "day", "person"
)

corpus <- tm_map(corpus, removeWords, fashion_stopwords_stemmed)

# Construct Document Term Matrix
# Rows = Documents n , Columns = Words p
dtm <- DocumentTermMatrix(corpus) 
dtm_matrix <- as.matrix(dtm)

# Word counts across columns
word_counts <- colSums(dtm_matrix)
word_freqs <- sort(word_counts, decreasing = TRUE)

head(word_freqs, 10)

# Cleaning rare words
rare_words <- names(word_counts[word_counts <= 1])
dtm_clean_mat <- dtm_matrix[, !(colnames(dtm_matrix) %in% rare_words)]

# Vocabulary Size p and Document Count n
p_vocab <- ncol(dtm_clean_mat)
n_docs  <- nrow(dtm_clean_mat)

cat("Vocabulary size p:", p_vocab, "\n")
cat("Number of Documents n:", n_docs, "\n")


# Word frequencies
word_freqs <- sort(colSums(dtm_clean_mat), decreasing = TRUE)
df_freq <- data.frame(word = names(word_freqs), freq = word_freqs)

# Word Cloud
wordcloud(words = df_freq$word, freq = df_freq$freq, min.freq = 3,
          max.words = 100, random.order = FALSE,
          colors = brewer.pal(8, "Dark2"))

# Tibble
reviews_df <- tibble(line = 1:nrow(df_analysis), text = df_analysis$Review.Text, Recommended = df_analysis$Recommended.IND)

data("stop_words")
bing_sentiments <- get_sentiments("bing")

reviews_words_clean <- reviews_df %>%
  unnest_tokens(word, text) %>%
  anti_join(stop_words, by = "word")

reviews_sentiment <- reviews_words_clean %>%
  inner_join(bing_sentiments, by = "word")

# Compute sentiment score Positive - Negative
review_scores <- reviews_sentiment %>%
  count(line, sentiment) %>%
  pivot_wider(names_from = sentiment, values_from = n, values_fill = 0) %>%
  mutate(sentiment_score = positive - negative)

final_sentiment <- reviews_df %>%
  left_join(review_scores, by = "line") %>%
  mutate(sentiment_score = ifelse(is.na(sentiment_score), 0, sentiment_score)) %>%
  mutate(sentiment_category = case_when(
    sentiment_score > 0 ~ "Positive",
    sentiment_score < 0 ~ "Negative",
    TRUE ~ "Neutral"
  ))

# Plot Distribution of Sentiment Scores
ggplot(final_sentiment, aes(x = sentiment_score)) +
  geom_histogram(binwidth = 1, fill = "green", color = "black") +
  theme_minimal() +
  labs(title = "Distribution of Clothing Review Sentiment Scores",
       x = "Sentiment Score (Positive - Negative)",
       y = "Number of Reviews")


# Plot Top 10 contributing sentimental words
reviews_words_clean %>%
  inner_join(bing_sentiments, by = "word") %>%
  count(word, sentiment, sort = TRUE) %>%
  group_by(sentiment) %>%
  slice_max(n, n = 10) %>%
  ungroup() %>%
  mutate(word = reorder(word, n)) %>%
  ggplot(aes(x = word, y = n, fill = sentiment)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~sentiment, scales = "free_y") +
  coord_flip() +
  labs(title = "Top Sentiment Contributing Words in Product Reviews",
       y = "Word Frequency", x = "Word") +
  theme_minimal()

# Plot Sentiment Categories grouped by Labels (Recommendation) 
ggplot(final_sentiment, aes(x = sentiment_category, fill = factor(Recommended))) +
  geom_bar(position = "dodge") +
  scale_fill_manual(values = c("0" = "blue", "1" = "red"), name = "Recommended", labels = c("No", "Yes")) +
  theme_minimal() +
  labs(title = "Review Sentiment Distribution by Recommendation Status",
       x = "Sentiment Category", y = "Count of Reviews")

#####################################
# Latent Semantic Analysis (LSA)

# Singular Value Decomposition on Document-Term Matrix
svd_out <- svd(dtm_clean_mat)

dim(dtm_clean_mat)

svd_out$d # component importance
length(svd_out$d) #number of components

# Plot the singular values
plot(svd_out$d, 
     main = "SVD Component Importance", 
     ylab = "Importance", 
     xlab = "Component Index")

# Plot the percent variability explained
plot(svd_out$d^2 / sum(svd_out$d^2) * 100, 
     ylab = "Percent variability explained",
     xlab = "SVD Components",
     main = "Explained Variance")

# Reconstruct the low-rank semantic matrix using the first 2 eigenvalues
td_appr <- svd_out$u[,1:2,drop=FALSE] %*%
  diag(svd_out$d[1:2]) %*%
  t(svd_out$v[,1:2,drop=FALSE])

# The sharp decline in singular values indicates that the first few components 
# capture most of the important structure in the review data. 
# Therefore, SVD is useful for reducing dimensionality and preparing the text data for clustering or further semantic analysis.

rownames(td_appr) <- rownames(dtm_clean_mat)
colnames(td_appr) <- colnames(dtm_clean_mat)

# Calculate suggested number of topics q to explain at least 70% of variability
q <- which((cumsum(svd_out$d^2) / sum(svd_out$d^2)) > 0.70)[1]  #268

# Extract the word weights for the q topics
A <- abs(t(svd_out$v[, 1:q]))
dim(A)
colnames(A) <- colnames(dtm_clean_mat)

# Get the indices of the highest-weighted words for the first 5 topics
index1 <- order(A[1,], decreasing = TRUE)
index2 <- order(A[2,], decreasing = TRUE)
index3 <- order(A[3,], decreasing = TRUE)
index4 <- order(A[4,], decreasing = TRUE)
index5 <- order(A[5,], decreasing = TRUE)

# top 5 contributing words for these latent topics
cat("\nTop 5 words for Latent Topic 1:", colnames(dtm_clean_mat)[index1[1:5]])
cat("\nTop 5 words for Latent Topic 2:", colnames(dtm_clean_mat)[index2[1:5]])
cat("\nTop 5 words for Latent Topic 3:", colnames(dtm_clean_mat)[index3[1:5]])
cat("\nTop 5 words for Latent Topic 4:", colnames(dtm_clean_mat)[index4[1:5]])
cat("\nTop 5 words for Latent Topic 5:", colnames(dtm_clean_mat)[index5[1:5]])

#####################################
# Mixture of Unigrams - MOU Estimation

library(deepMOU)

# We will fit the standard k=2 and k=3 to find the optimal framework. Since our dataset is huge (~22k rows) only k=2,3 are checked.
fit_mou2 <- mou_EM(dtm_clean_mat, k = 2, seed = 123)
fit_mou3 <- mou_EM(dtm_clean_mat, k = 3, seed = 123)

mou_k <- c(fit_mou2$k, fit_mou3$k)
mou_aic <- c(fit_mou2$AIC, fit_mou3$AIC)

cat("MOU AIC Scores (k=2 vs k=3):", mou_aic, "\n")
cat("Best MOU cluster choice based on AIC:", mou_k[which.min(mou_aic)], "clusters.")

# Final cluster assignments from MOU
best_mou <- if (fit_mou2$AIC < fit_mou3$AIC) fit_mou2 else fit_mou3
mou_clusters <- best_mou$clusters

# Heatmap for MOU Clusters

# Heatmap of the most representative words by each cluster
heatmap_words(
  x = dtm_clean_mat,
  clusters = mou_clusters
)

# Word frequency plot by cluster
words_freq_plot(
  dtm_clean_mat,
  clusters = mou_clusters)


#####################################
# Latent Dirichlet Allocation (LDA)

library(topicmodels)

dtm_clean <- dtm[, !(Terms(dtm) %in% rare_words)]

# Run LDA for k = 3 topics
# k = 3 was selected to obtain interpretable and stable latent semantic topics.
fit_lda <- LDA(dtm_clean, k = 3, method = "VEM", control = list(seed = 123))

# represents probability of each document belonging to each topic
theta_fit <- fit_lda@gamma

# top 10 words per topic
terms(fit_lda, 10)

# most likely topic per document
doc_topics <- topics(fit_lda)

# topic counts
table(doc_topics)

# plot topic distribution
barplot(
    table(doc_topics),
    main = "LDA Topic Distribution",
    xlab = "Topic",
    ylab = "Number of Reviews"
)

# LDA Top Words Barplot
lda_terms <- terms(fit_lda, 10)

topic1_df <- data.frame(
  word = lda_terms[,1],
  rank = 10:1
)

topic2_df <- data.frame(
  word = lda_terms[,2],
  rank = 10:1
)

topic3_df <- data.frame(
  word = lda_terms[,3],
  rank = 10:1
)

# Topic 1 barplot
ggplot(topic1_df, aes(x = reorder(word, rank), y = rank)) +
  geom_col(fill = "pink") +
  coord_flip() +
  theme_minimal() +
  labs(
    title = "Top Words - LDA Topic 1",
    x = "Words",
    y = "Importance Rank"
  )

# Topic 2 barplot
ggplot(topic2_df, aes(x = reorder(word, rank), y = rank)) +
  geom_col(fill = "orange") +
  coord_flip() +
  theme_minimal() +
  labs(
    title = "Top Words - LDA Topic 2",
    x = "Words",
    y = "Importance Rank"
  )

# Topic 3 barplot
ggplot(topic3_df, aes(x = reorder(word, rank), y = rank)) +
  geom_col(fill = "green") +
  coord_flip() +
  theme_minimal() +
  labs(
    title = "Top Words - LDA Topic 3",
    x = "Words",
    y = "Importance Rank"
  )


#####################################
# Evaluation Metrics with Department Name as Label
library(mclust)
cl_true <- df_analysis$Department.Name

# MOU clusters vs label
table(mou_clusters, cl_true)

# Adjusted Rand Index
ari_mou <- adjustedRandIndex(mou_clusters, cl_true)
cat("MOU Adjusted Rand Index:", round(ari_mou, 4), "\n")

# Accuracy-like measure
mou_table <- table(mou_clusters, cl_true)
mou_accuracy <- sum(apply(mou_table, 1, max)) / sum(mou_table)
cat("MOU Accuracy:", round(mou_accuracy, 4), "\n")


#####################################
# LDA Evaluation


# Convert LDA topic probabilities to hard topic assignment
lda_clusters <- apply(theta_fit, 1, which.max)

# LDA topics vs true recommendation label
table(lda_clusters, cl_true)

# Adjusted Rand Index
ari_lda <- adjustedRandIndex(lda_clusters, cl_true)
cat("LDA Adjusted Rand Index:", round(ari_lda, 4), "\n")

# Accuracy-like measure
lda_table <- table(lda_clusters, cl_true)
lda_accuracy <- sum(apply(lda_table, 1, max)) / sum(lda_table)
cat("LDA Accuracy:", round(lda_accuracy, 4), "\n")

# Although LDA successfully extracted balanced latent semantic topics from the 
# reviews, the discovered topic structure showed weak alignment with the 
# recommendation labels, resulting in an Adjusted Rand Index close to zero. 
# This suggests that recommendation behavior is influenced by factors beyond 
# latent semantic topics alone.


#####################################
# Model Comparison

# Create comparison table
model_comparison <- data.frame(
  Model = c("MOU", "LDA"),
  Clusters_or_Topics = c(length(unique(mou_clusters)), 3),
  ARI = c(round(ari_mou, 4), round(ari_lda, 4)),
  Accuracy = c(round(mou_accuracy, 4), round(lda_accuracy, 4))
)

print(model_comparison)

# Visualization of ARI comparison
ggplot(model_comparison, aes(x = Model, y = ARI, fill = Model)) +
  geom_col() +
  theme_minimal() +
  labs(
    title = "Adjusted Rand Index Comparison",
    x = "Model",
    y = "ARI"
  )

# Visualization of Accuracy comparison
ggplot(model_comparison, aes(x = Model, y = Accuracy, fill = Model)) +
  geom_col() +
  theme_minimal() +
  labs(
    title = "Accuracy Comparison",
    x = "Model",
    y = "Accuracy"
  )

# MOU achieved better alignment with the Department.Name labels than LDA.
# After additional domain-specific stopword removal, MOU reached the highest ARI
# and accuracy among the fitted models.
# This suggests that MOU captured product-department structure more effectively.
# LDA produced balanced and interpretable topic mixtures, but these topics showed
# weaker direct alignment with Department.Name labels when converted into hard clusters.
# Therefore, MOU was selected as the main clustering model, while LDA was used
# mainly for topic interpretation and LSA/SVD for dimensionality reduction.
