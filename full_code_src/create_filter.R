filters = as.list(mlr:::.FilterRegister)
filters$rank.correlation

str(filters$rank.correlation)

filters$rank.correlation$fun
makeFilter(
  name = "nonsense.filter",
  desc = "Calculates scores according to alphabetical order of features",
  pkg = "",
  supported.tasks = c("classif", "regr", "surv"),
  supported.features = c("numerics", "factors", "ordered"),
  fun = function(task, nselect, decreasing = TRUE, ...) {
    feats = getTaskFeatureNames(task)
    imp = order(feats, decreasing = decreasing)
    names(imp) = feats
    imp
  }
)
listFilterMethods()$id
d = generateFilterValuesData(iris.task, method = c("nonsense.filter", "anova.test"))
d

plotFilterValues(d)
iris.task.filtered = filterFeatures(iris.task, method = "nonsense.filter", abs = 2)
iris.task.filtered

getTaskFeatureNames(iris.task.filtered)
rm("nonsense.filter", envir = mlr:::.FilterRegister)
