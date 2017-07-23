if (!require(devtools)) {
  install.packages("devtools")
  library(devtools)
}
install_github("mlr-org/mlr")

print("LIBPATHS:")
print(.libPaths())

ip = installed.packages()
ip2 = rownames(ip)
print("INSTALLED PACKAGES:")
print(ip2)

p = parse_deps(ip["mlr", "Suggests"])[, "name"]
p = c(p, "pander", "formatR")

pmiss = setdiff(p, ip2)
print("MISSING PACKAGES:")
print(pmiss)
if (length(pmiss) > 0)
  install.packages(pmiss[1:40])

update.packages(ask = FALSE)
