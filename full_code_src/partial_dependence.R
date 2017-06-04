lrn.classif = makeLearner("classif.ksvm", predict.type = "prob")
fit.classif = train(lrn.classif, iris.task)
pd = generatePartialDependenceData(fit.classif, iris.task, "Petal.Width")
pd
pd.lst = generatePartialDependenceData(fit.classif, iris.task, c("Petal.Width", "Petal.Length"), FALSE)
head(pd.lst$data)

tail(pd.lst$data)
pd.int = generatePartialDependenceData(fit.classif, iris.task, c("Petal.Width", "Petal.Length"), TRUE)
pd.int
lrn.regr = makeLearner("regr.ksvm")
fit.regr = train(lrn.regr, bh.task)
pd.regr = generatePartialDependenceData(fit.regr, bh.task, "lstat", fun = median)
pd.regr
pd.ci = generatePartialDependenceData(fit.regr, bh.task, "lstat",
  fun = function(x) quantile(x, c(.25, .5, .75)))
pd.ci
pd.classif = generatePartialDependenceData(fit.classif, iris.task, "Petal.Length", fun = median)
pd.classif
fit.se = train(makeLearner("regr.randomForest", predict.type = "se"), bh.task)
pd.se = generatePartialDependenceData(fit.se, bh.task, c("lstat", "crim"))
head(pd.se$data)

tail(pd.se$data)
pd.ind.regr = generatePartialDependenceData(fit.regr, bh.task, "lstat", individual = TRUE)
pd.ind.regr
pd.ind.classif = generatePartialDependenceData(fit.classif, iris.task, "Petal.Length", individual = TRUE)
pd.ind.classif
iris = getTaskData(iris.task)
pd.ind.classif = generatePartialDependenceData(fit.classif, iris.task, "Petal.Length", individual = TRUE,
  center = list("Petal.Length" = min(iris$Petal.Length)))
pd.regr.der = generatePartialDependenceData(fit.regr, bh.task, "lstat", derivative = TRUE)
head(pd.regr.der$data)
pd.regr.der.ind = generatePartialDependenceData(fit.regr, bh.task, "lstat", derivative = TRUE,
  individual = TRUE)
head(pd.regr.der.ind$data)
pd.classif.der = generatePartialDependenceData(fit.classif, iris.task, "Petal.Width", derivative = TRUE)
head(pd.classif.der$data)
pd.classif.der.ind = generatePartialDependenceData(fit.classif, iris.task, "Petal.Width", derivative = TRUE,
  individual = TRUE)
head(pd.classif.der.ind$data)
lrn.regr = makeLearner("regr.ksvm")
fit.regr = train(lrn.regr, bh.task)

fa = generateFunctionalANOVAData(fit.regr, bh.task, "lstat", depth = 1, fun = median)
fa

pd.regr = generatePartialDependenceData(fit.regr, bh.task, "lstat", fun = median)
pd.regr
fa.bv = generateFunctionalANOVAData(fit.regr, bh.task, c("crim", "lstat", "age"),
  depth = 2)
fa.bv

names(table(fa.bv$data$effect)) ## interaction effects estimated
plotPartialDependence(pd.regr)
plotPartialDependence(pd.classif)
plotPartialDependence(pd.ci)
plotPartialDependence(pd.se)
plotPartialDependence(pd.lst)
plotPartialDependence(pd.int, facet = "Petal.Length")
## plotPartialDependenceGGVIS(pd.int, interact = "Petal.Length")
plotPartialDependence(pd.ind.regr)
plotPartialDependence(pd.ind.classif)
plotPartialDependence(pd.regr.der)
plotPartialDependence(pd.regr.der.ind)
plotPartialDependence(pd.classif.der.ind)
fa = generateFunctionalANOVAData(fit.regr, bh.task, c("crim", "lstat"), depth = 1)
plotPartialDependence(fa)
fa.bv = generateFunctionalANOVAData(fit.regr, bh.task, c("crim", "lstat"), depth = 2)
plotPartialDependence(fa.bv, "tile")
