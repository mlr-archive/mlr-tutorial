library(devtools)

p = c(
  "caret", 
  "clue", 
  "cmaes", 
  "devtools", 
  "digest", 
  "e1071",
  "fnn",
  "FSelector",
  "GGally", 
  "glmnet", 
  "Hmisc", 
  "irace",
  "kernlab",
  "knitr", 
  "mlbench",
  "nnet", 
  "pander", 
  "PMCMR",
  "randomForest",
  "rFerns", 
  "rgl", 
  "ROCR", 
  "roxygen2", 
  "stringr"
)

print(.libPaths())

ip = installed.packages()
ip2 = rownames(ip)
print(ip2)

install_github("mlr-org/mlr")

pmiss = setdiff(p, ip2)
print(pmiss)
if (length(pmiss) > 0)
  install.packages(pmiss)

update.packages(ask = FALSE)
