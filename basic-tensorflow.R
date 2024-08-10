# basic tensorflow
# https://tensorflow.rstudio.com/install/

# install.packages("remotes")
remotes::install_github("rstudio/tensorflow")

reticulate::install_python()

library(tensorflow)
install_tensorflow(envname = "r-tensorflow")

# woudn't work until installed tensorflow using pip install tensorflow
library(tensorflow)
tf$constant("Hello TensorFlow!")
