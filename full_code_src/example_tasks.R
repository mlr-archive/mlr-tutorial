params <-
structure(list(full.code = FALSE), .Names = "full.code")

# urlMlrFunctions, ext are defined in build
linkTask = function(x) {
  collapse(sprintf("[%1$s](%2$s%1$s%3$s)", x, urlMlrFunctions, ext), sep = "<br />")
}
d = data(package = "mlr")
d = d$results
types = sapply(d[,"Item"], function(x) getTaskDesc(eval(parse(text = x)))$type)
names(types) = NULL
ord = order(types)
d = d[ord,]
types = types[ord]
types[duplicated(types)] = ""
df = data.frame(Type = types, Task = d[,"Item"], Description = d[,"Title"])
df$Task = sapply(df$Task, linkTask)
pandoc.table(df, style = "rmarkdown", split.tables = Inf, split.cells = Inf,
  justify = rep("left", 3))
