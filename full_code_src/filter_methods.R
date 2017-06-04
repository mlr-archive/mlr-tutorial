params <-
structure(list(full.code = FALSE), .Names = "full.code")

# urlContribPackages is defined in build
linkPkg = function(x) {
  ifelse(x == "", "", collapse(sprintf("[%1$s](%2$s%1$s/)", x, urlContribPackages), sep = "<br />"))
}
df = listFilterMethods(desc = TRUE, tasks = TRUE, features = TRUE, include.deprecated = TRUE)
df$package = sapply(df$package, linkPkg)

depr = df$deprecated
df$deprecated = NULL

logicals = vlapply(df, is.logical)
df[logicals] = lapply(df[logicals], function(x) ifelse(x, "X", ""))
names(df) = c("Method", "Package", "Description", "Classif", "Regr", "Surv", "Fac.", "Num.", "Ord.")
just = rep(c("left", "center"), c(3, ncol(df) - 3))
dfnd = df[!depr,]
rownames(dfnd) = seq_len(nrow(dfnd))
pandoc.table(dfnd, style = "rmarkdown", split.tables = Inf, split.cells = Inf, emphasize.rownames = FALSE, justify = just)
dfd = df[depr,]
rownames(dfd) = seq_len(nrow(dfd))
pandoc.table(dfd, style = "rmarkdown", split.tables = Inf, split.cells = Inf, justify = just)
