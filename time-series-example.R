# Timeseries classification
# https://keras3.posit.co/articles/examples/timeseries/timeseries_classification_from_scratch.html

library(keras3)
keras3::use_backend("jax")
# pip install -U jax needed before this worked

get_data <- function(path) {
  if(path |> startsWith("https://"))
    path <- get_file(origin = path)  # cache file locally

  data <- readr::read_tsv(
    path, col_names = FALSE,
    # Each row is: one integer (the label),
    # followed by 500 doubles (the timeseries)
    col_types = paste0("i", strrep("d", 500))
  )

  y <- as.matrix(data[[1]])
  x <- as.matrix(data[,-1])
  dimnames(x) <- dimnames(y) <- NULL

  list(x, y)
}

root_url <- "https://raw.githubusercontent.com/hfawaz/cd-diagram/master/FordA/"

# breaks here
c(x_train, y_train) %<-% get_data(paste0(root_url, "FordA_TRAIN.tsv"))
c(x_test, y_test) %<-% get_data(paste0(root_url, "FordA_TEST.tsv"))

# test
df <- get_data(paste0(root_url, "FordA_TRAIN.tsv"))

str(keras3:::named_list(
  x_train, y_train,
  x_test, y_test
))
