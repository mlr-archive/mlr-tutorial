p = c(
  "roxygen2", 
  "devtools", 
  "glmnet", 
  "ROCR", 
  "digest", 
  "pander", 
  "knitr", 
  "caret", 
  "rgl", 
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
