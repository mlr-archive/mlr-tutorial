params <-
structure(list(full.code = FALSE), .Names = "full.code")

# urlMlrFunctions, ext are defined in build
linkFct = function(x, y) {
  collapse(sprintf("[%1$s](%3$s%2$s%4$s)", x, y, urlMlrFunctions, ext), sep = "<br />")
}

cn = function(x) if (is.null(x)) NA else gsub("\\n", " ", x)
urls = function(x) if (is.na(x)) NA else gsub("(http)(\\S+)(\\.)", "[\\1\\2](\\1\\2)\\3", x)
# regex is not ideal and can break

getTab = function(type) {
  m = list(featperc = featperc, timeboth = timeboth, timepredict = timepredict, timetrain = timetrain)

  if (type == "general") {
    meas = m
  } else {
    meas = listMeasures(type, create = TRUE)
    ord = order(names(meas))
    meas = meas[ord]
    keep = setdiff(names(meas), names(m))
    meas = meas[keep]
  }

  cols = c("ID / Name", "Minim.", "Best", "Worst", "Multi", "Pred.", "Truth", "Probs", "Model", "Task", "Feats", "Aggr.", "Note")
  df = makeDataFrame(nrow = length(meas), ncol = length(cols),
    col.types = c("character", "logical", "numeric", "numeric", "logical", "logical", "logical", "logical", "logical", "logical", "logical", "character", "character"))
  names(df) = cols

  for (i in seq_along(meas)) {
    mea = meas[[i]]
    df[i, 1] = paste0("**", linkFct(mea$id, "measures"), "** <br />", mea$name)
    df[i, 2] = mea$minimize
    df[i, 3] = mea$best
    df[i, 4] = mea$worst
    df[i, 5] = "classif.multi" %in% mea$properties
    df[i, 6] = "req.pred" %in% mea$properties
    df[i, 7] = "req.truth" %in% mea$properties
    df[i, 8] = "req.prob" %in% mea$properties
    df[i, 9] = "req.model" %in% mea$properties
    df[i, 10] = "req.task" %in% mea$properties
    df[i, 11] = "req.feats" %in% mea$properties
    df[i, 12] = linkFct(mea$aggr$id, "aggregations")
    df[i, 13] = urls(cn(mea$note))
  }

  just = c("left", "center", "right", "right", "center", "center", "center", "center", "center", "center", "center", "left", "left")

  if (type != "classif") {
    ind = cols != "Multi"
    df = df[ind]
    just = just[ind]
  }

  logicals = vlapply(df, is.logical)
  df[logicals] = lapply(df[logicals], function(x) ifelse(x, "X", ""))
  pandoc.table(df, style = "rmarkdown", split.tables = Inf, split.cells = Inf,
    justify = just)
}
getTab("classif")
getTab("regr")
getTab("surv")
getTab("cluster")
getTab("costsens")
getTab("multilabel")
getTab("general")
