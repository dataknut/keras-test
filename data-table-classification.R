# https://keras3.posit.co/articles/examples/structured_data/structured_data_classification_with_feature_space.html

library(readr)
library(dplyr, warn.conflicts = FALSE)
library(keras3)
library(tensorflow, exclude = c("shape", "set_random_seed"))
library(tfdatasets, exclude = "shape")

conflicted::conflicts_prefer(
  keras3::shape(),
  keras3::set_random_seed(),
  dplyr::filter(),
  .quiet = TRUE
)

use_backend("tensorflow")

file_url <-
  "http://storage.googleapis.com/download.tensorflow.org/data/heart.csv"
df <- read_csv(file_url, col_types = cols(
  oldpeak = col_double(),
  thal = col_character(),
  .default = col_integer()
))

# the dataset has two malformed rows, filter them out
df <- df |> filter(!thal %in% c("1", "2"))

dplyr::glimpse(df)

# take a 20% subset for validation
val_idx <- nrow(df) %>% sample.int(., . * 0.2) # select 20% rows
val_df <- df[val_idx, ] # extract the 20% to a validation set
train_df <- df[-val_idx, ] # extract 80% to a training set

cat(sprintf(
  "Using %d samples for training and %d for validation",
  nrow(train_df), nrow(val_df)
))

# so many ways to skin a cat
message("Using ", nrow(train_df),
        " samples for training and ",
        nrow(val_df), " for validation."
)

# target is the outcome we're interested in - whether the patient has a heart disease (1) or not (0)
# this function transforms the data into the shape needed (tf_dataset)
# note use of |> pipe operator

dataframe_to_dataset <- function(df) {
  labels <- df |> dplyr::pull(target) |> as.integer() # the labels (i.e. target)
  inputs <- df |> dplyr::select(-target) |> as.list()

  ds <- tensor_slices_dataset(list(inputs, labels)) |> # create the slices from the inputs and labels
    dataset_shuffle(nrow(df)) # shuffle (why?)

  return(ds)
}

train_ds <- dataframe_to_dataset(train_df)
val_ds <- dataframe_to_dataset(val_df)

c(x, y) %<-% iter_next(as_iterator(train_ds))
cat("Input: "); str(x)
cat("Target: "); str(y)

# create batches - why?
train_ds <- train_ds |> dataset_batch(32)
val_ds <- val_ds |> dataset_batch(32)

# set up feature space
# tells tensorflow how to handle each of the labels (variables)

# breaks here

feature_space <- layer_feature_space(
  features = list(
    # Categorical features encoded as integers # BA: coded as integer values
    sex = "integer_categorical",
    cp = "integer_categorical",
    fbs = "integer_categorical",
    restecg = "integer_categorical",
    exang = "integer_categorical",
    ca = "integer_categorical",
    # Categorical feature encoded as string # BA: multiple strings
    thal = "string_categorical",
    # Numerical features to discretize # BA puts into bins
    age = "float_discretized",
    # Numerical features to normalize # BA: neural nets need normalised
    trestbps = "float_normalized",
    chol = "float_normalized",
    thalach = "float_normalized",
    oldpeak = "float_normalized",
    slope = "float_normalized"
  ),
  # We create additional features by hashing
  # value co-occurrences for the
  # following groups of categorical features.
  crosses = list(c("sex", "age"), c("thal", "ca")), # BA: sort of like interaction effects
  # The hashing space for these co-occurrences
  # will be 32-dimensional.
  crossing_dim = 32,
  # Our utility will one-hot encode all categorical
  # features and concat all features into a single
  # vector (one vector per sample).
  output_mode = "concat"
)

train_ds_with_no_labels <- train_ds |> dataset_map(\(x, y) x)
feature_space |> adapt(train_ds_with_no_labels)


