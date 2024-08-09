# setup

# install libraries
pkgs <- c("keras", "lime", "tidyquant", "rsample", "recipes", "yardstick", "corrr")
install.packages(pkgs)

# Load libraries
library(keras)

# Install Keras if you have not installed before
keras::install_keras()

# looks for Python & installs all sorts of stuff from tensorflow
# including tensorflow which is big...
# Downloading tensorflow-2.15.1-cp310-cp310-macosx_10_15_x86_64.whl (236.4 MB)
# etc
