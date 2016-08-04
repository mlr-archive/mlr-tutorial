library(devtools)

p = c(
  "caret", 
  "clue", 
  "devtools", 
  "digest", 
  "e1071",
  "glmnet", 
  "kernlab",
  "knitr", 
  "mlbench",
  "nnet", 
  "pander", 
  "PMCMR",
  "randomForest",
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
