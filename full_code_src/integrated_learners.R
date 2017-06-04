params <-
structure(list(full.code = FALSE), .Names = "full.code")

library("mlr")
library("pander")
# baseR, urlBasePackages, urlContribPackages are defined in build
linkPkg = function(x) {
  x = strsplit(x, ",")[[1]]
  x = mlr:::cleanupPackageNames(x)    # remove exclamation marks
  if (urlBasePackages != urlContribPackages) {
    ind = x %in% baseR
    url = c(urlContribPackages, urlBasePackages)[ind + 1]
  } else {
    url = urlContribPackages
  }
  collapse(sprintf("[%1$s](%2$s%1$s/)", x, url), sep = "<br />")
}

getTab = function(type) {
  cn = function(x) if (is.null(x)) NA else gsub("\\n", " ", x)
  lrns = listLearners(type)
  lrns$note[is.na(lrns$note)] = ""
  cols = c("class", "type", "package", "short.name", "name", "numerics", "factors", "ordered", "missings", "weights", "note", "installed")
  props = setdiff(colnames(lrns), cols)

  colNames = c("Class / Short Name / Name", "Packages", "Num.", "Fac.", "Ord.", "NAs", "Weights", "Props", "Note")
  df = makeDataFrame(nrow = nrow(lrns), ncol = length(colNames),
    col.types = c("character", "character", "logical", "logical", "logical", "logical", "logical", "character", "character"))
  names(df) = colNames

  df[1] = apply(lrns[c("class", "short.name", "name")], 1, function(x)
    paste0("**", x["class"], "** <br /> *", cn(x["short.name"]), "* <br /><br />", cn(x["name"])))
  df$Packages = sapply(lrns$package, linkPkg)
  df[3:7] = lrns[c("numerics", "factors", "ordered", "missings", "weights")]
  df$Props = apply(lrns[props], 1, function(x) collapse(props[x], sep = "<br />"))
  df$Note = sapply(lrns$note, cn)
  logicals = vlapply(df, is.logical)
  df[logicals] = lapply(df[logicals], function(x) ifelse(x, "X", ""))
  df
}

makeTab = function(df) {
  pandoc.table(df, style = "rmarkdown", split.tables = Inf, split.cells = Inf,
    justify = c("left", "left", "center", "center", "center", "center", "center", "left", "left"))
}

types = c("classif", "multilabel", "regr", "surv", "cluster")
tables = lapply(types, getTab)
names(tables) = types
numbers = sapply(tables, nrow)
makeTab(tables[["classif"]])
makeTab(tables[["regr"]])
makeTab(tables[["surv"]])
makeTab(tables[["cluster"]])
makeTab(tables[["multilabel"]])
