## Tuning in inner resampling loop
ps = makeParamSet(
  makeDiscreteParam("C", values = 2^(-2:2)),
  makeDiscreteParam("sigma", values = 2^(-2:2))
)
ctrl = makeTuneControlGrid()
inner = makeResampleDesc("Subsample", iters = 2)
lrn = makeTuneWrapper("classif.ksvm", resampling = inner, par.set = ps, control = ctrl, show.info = FALSE)

## Outer resampling loop
outer = makeResampleDesc("CV", iters = 3)
r = resample(lrn, iris.task, resampling = outer, extract = getTuneResult, show.info = FALSE)

r
r$measures.test
r$extract

names(r$extract[[1]])
opt.paths = getNestedTuneResultsOptPathDf(r)
head(opt.paths, 10)
g = ggplot(opt.paths, aes(x = C, y = sigma, fill = mmce.test.mean))
g + geom_tile() + facet_wrap(~ iter)
getNestedTuneResultsX(r)
## Feature selection in inner resampling loop
inner = makeResampleDesc("CV", iters = 3)
lrn = makeFeatSelWrapper("regr.lm", resampling = inner,
  control = makeFeatSelControlSequential(method = "sfs"), show.info = FALSE)

## Outer resampling loop
outer = makeResampleDesc("Subsample", iters = 2)
r = resample(learner = lrn, task = bh.task, resampling = outer, extract = getFeatSelResult,
  show.info = FALSE)

r

r$measures.test
r$extract

## Selected features in the first outer resampling iteration
r$extract[[1]]$x

## Resampled performance of the selected feature subset on the first inner training set
r$extract[[1]]$y
opt.paths = lapply(r$extract, function(x) as.data.frame(x$opt.path))
head(opt.paths[[1]])
analyzeFeatSelResult(r$extract[[1]])
## Tuning of the percentage of selected filters in the inner loop
lrn = makeFilterWrapper(learner = "regr.lm", fw.method = "chi.squared")
ps = makeParamSet(makeDiscreteParam("fw.threshold", values = seq(0, 1, 0.2)))
ctrl = makeTuneControlGrid()
inner = makeResampleDesc("CV", iters = 3)
lrn = makeTuneWrapper(lrn, resampling = inner, par.set = ps, control = ctrl, show.info = FALSE)

## Outer resampling loop
outer = makeResampleDesc("CV", iters = 3)
r = resample(learner = lrn, task = bh.task, resampling = outer, models = TRUE, show.info = FALSE)
r
r$models
lapply(r$models, function(x) getFilteredFeatures(x$learner.model$next.model))
res = lapply(r$models, getTuneResult)
res

opt.paths = lapply(res, function(x) as.data.frame(x$opt.path))
opt.paths[[1]]
## List of learning tasks
tasks = list(iris.task, sonar.task)

## Tune svm in the inner resampling loop
ps = makeParamSet(
  makeDiscreteParam("C", 2^(-1:1)),
  makeDiscreteParam("sigma", 2^(-1:1)))
ctrl = makeTuneControlGrid()
inner = makeResampleDesc("Holdout")
lrn1 = makeTuneWrapper("classif.ksvm", resampling = inner, par.set = ps, control = ctrl,
  show.info = FALSE)

## Tune k-nearest neighbor in inner resampling loop
ps = makeParamSet(makeDiscreteParam("k", 3:5))
ctrl = makeTuneControlGrid()
inner = makeResampleDesc("Subsample", iters = 3)
lrn2 = makeTuneWrapper("classif.kknn", resampling = inner, par.set = ps, control = ctrl,
  show.info = FALSE)

## Learners
lrns = list(lrn1, lrn2)

## Outer resampling loop
outer = list(makeResampleDesc("Holdout"), makeResampleDesc("Bootstrap", iters = 2))
res = benchmark(lrns, tasks, outer, measures = list(acc, ber), show.info = FALSE)
res
getBMRPerformances(res, as.df = TRUE)
getBMRTuneResults(res)
getBMRTuneResults(res, as.df = TRUE)
tune.res = getBMRTuneResults(res, task.ids = "Sonar-example", learner.ids = "classif.ksvm.tuned",
  as.df = TRUE)
tune.res

getNestedTuneResultsOptPathDf(res$results[["Sonar-example"]][["classif.ksvm.tuned"]])
## Feature selection in inner resampling loop
ctrl = makeFeatSelControlSequential(method = "sfs")
inner = makeResampleDesc("Subsample", iters = 2)
lrn = makeFeatSelWrapper("regr.lm", resampling = inner, control = ctrl, show.info = FALSE)

## Learners
lrns = list("regr.rpart", lrn)

## Outer resampling loop
outer = makeResampleDesc("Subsample", iters = 2)
res = benchmark(tasks = bh.task, learners = lrns, resampling = outer, show.info = FALSE)

res
getBMRFeatSelResults(res)
getBMRFeatSelResults(res, drop = TRUE)
feats = getBMRFeatSelResults(res, learner.id = "regr.lm.featsel", drop = TRUE)

## Selected features in the first outer resampling iteration
feats[[1]]$x

## Resampled performance of the selected feature subset on the first inner training set
feats[[1]]$y
opt.paths = lapply(feats, function(x) as.data.frame(x$opt.path))
head(opt.paths[[1]])

analyzeFeatSelResult(feats[[1]])
## Feature filtering with tuning in the inner resampling loop
lrn = makeFilterWrapper(learner = "regr.lm", fw.method = "chi.squared")
ps = makeParamSet(makeDiscreteParam("fw.abs", values = seq_len(getTaskNFeats(bh.task))))
ctrl = makeTuneControlGrid()
inner = makeResampleDesc("CV", iter = 2)
lrn = makeTuneWrapper(lrn, resampling = inner, par.set = ps, control = ctrl,
  show.info = FALSE)

## Learners
lrns = list("regr.rpart", lrn)

## Outer resampling loop
outer = makeResampleDesc("Subsample", iter = 3)
res = benchmark(tasks = bh.task, learners = lrns, resampling = outer, show.info = FALSE)

res
## Performances on individual outer test data sets
getBMRPerformances(res, as.df = TRUE)
