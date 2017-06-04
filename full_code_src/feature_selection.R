fv = generateFilterValuesData(iris.task, method = "information.gain")
fv
fv2 = generateFilterValuesData(iris.task, method = c("information.gain", "chi.squared"))
fv2$data
plotFilterValues(fv2)
## plotFilterValuesGGVIS(fv2)
## Keep the 2 most important features
filtered.task = filterFeatures(iris.task, method = "information.gain", abs = 2)

## Keep the 25% most important features
filtered.task = filterFeatures(iris.task, fval = fv, perc = 0.25)

## Keep all features with importance greater than 0.5
filtered.task = filterFeatures(iris.task, fval = fv, threshold = 0.5)
filtered.task
lrn = makeFilterWrapper(learner = "classif.fnn", fw.method = "information.gain", fw.abs = 2)
rdesc = makeResampleDesc("CV", iters = 10)
r = resample(learner = lrn, task = iris.task, resampling = rdesc, show.info = FALSE, models = TRUE)
r$aggr
sfeats = sapply(r$models, getFilteredFeatures)
table(sfeats)
lrn = makeFilterWrapper(learner = "regr.lm", fw.method = "chi.squared")
ps = makeParamSet(makeDiscreteParam("fw.perc", values = seq(0.2, 0.5, 0.05)))
rdesc = makeResampleDesc("CV", iters = 3)
res = tuneParams(lrn, task = bh.task, resampling = rdesc, par.set = ps,
  control = makeTuneControlGrid())
res
as.data.frame(res$opt.path)
res$x
res$y
lrn = makeFilterWrapper(learner = "regr.lm", fw.method = "chi.squared", fw.perc = res$x$fw.perc)
mod = train(lrn, bh.task)
mod

getFilteredFeatures(mod)
lrn = makeFilterWrapper(learner = "classif.lda", fw.method = "chi.squared")
ps = makeParamSet(makeNumericParam("fw.threshold", lower = 0.1, upper = 0.9))
rdesc = makeResampleDesc("CV", iters = 10)
res = tuneParamsMultiCrit(lrn, task = sonar.task, resampling = rdesc, par.set = ps,
  measures = list(fpr, fnr), control = makeTuneMultiCritControlRandom(maxit = 50L),
  show.info = FALSE)
res
head(as.data.frame(res$opt.path))
plotTuneMultiCritResult(res)
## Specify the search strategy
ctrl = makeFeatSelControlRandom(maxit = 20L)
ctrl
## Resample description
rdesc = makeResampleDesc("Holdout")

## Select features
sfeats = selectFeatures(learner = "surv.coxph", task = wpbc.task, resampling = rdesc,
  control = ctrl, show.info = FALSE)
sfeats
sfeats$x
sfeats$y
## Specify the search strategy
ctrl = makeFeatSelControlSequential(method = "sfs", alpha = 0.02)

## Select features
rdesc = makeResampleDesc("CV", iters = 10)
sfeats = selectFeatures(learner = "regr.lm", task = bh.task, resampling = rdesc, control = ctrl,
  show.info = FALSE)
sfeats
analyzeFeatSelResult(sfeats)
rdesc = makeResampleDesc("CV", iters = 3)
lrn = makeFeatSelWrapper("surv.coxph", resampling = rdesc,
  control = makeFeatSelControlRandom(maxit = 10), show.info = FALSE)
mod = train(lrn, task = wpbc.task)
mod
sfeats = getFeatSelResult(mod)
sfeats
sfeats$x
out.rdesc = makeResampleDesc("CV", iters = 5)

r = resample(learner = lrn, task = wpbc.task, resampling = out.rdesc, models = TRUE,
  show.info = FALSE)
r$aggr
lapply(r$models, getFeatSelResult)
