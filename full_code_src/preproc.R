## preProcess(x, method = c("knnImpute", "pca"), pcaComp = 10)
## makePreprocWrapperCaret(learner, ppc.knnImpute = TRUE, ppc.pca = TRUE, ppc.pcaComp = 10)
sonar.task
lrn = makePreprocWrapperCaret("classif.qda", ppc.pca = TRUE, ppc.thresh = 0.9)
lrn
mod = train(lrn, sonar.task)
mod

getLearnerModel(mod)

getLearnerModel(mod, more.unwrap = TRUE)
rin = makeResampleInstance("CV", iters = 3, stratify = TRUE, task = sonar.task)
res = benchmark(list("classif.qda", lrn), sonar.task, rin, show.info = FALSE)
res
getParamSet(lrn)
ps = makeParamSet(
  makeIntegerParam("ppc.pcaComp", lower = 1, upper = getTaskNFeats(sonar.task)),
  makeDiscreteParam("predict.method", values = c("plug-in", "debiased"))
)
ctrl = makeTuneControlGrid(resolution = 10)
res = tuneParams(lrn, sonar.task, rin, par.set = ps, control = ctrl, show.info = FALSE)
res

as.data.frame(res$opt.path)[1:3]
trainfun = function(data, target, args = list(center, scale)) {
  ## Identify numerical features
  cns = colnames(data)
  nums = setdiff(cns[sapply(data, is.numeric)], target)
  ## Extract numerical features from the data set and call scale
  x = as.matrix(data[, nums, drop = FALSE])
  x = scale(x, center = args$center, scale = args$scale)
  ## Store the scaling parameters in control
  ## These are needed to preprocess the data before prediction
  control = args
  if (is.logical(control$center) && control$center)
    control$center = attr(x, "scaled:center")
  if (is.logical(control$scale) && control$scale)
    control$scale = attr(x, "scaled:scale")
  ## Recombine the data
  data = data[, setdiff(cns, nums), drop = FALSE]
  data = cbind(data, as.data.frame(x))
  return(list(data = data, control = control))
}
predictfun = function(data, target, args, control) {
  ## Identify numerical features
  cns = colnames(data)
  nums = cns[sapply(data, is.numeric)]
  ## Extract numerical features from the data set and call scale
  x = as.matrix(data[, nums, drop = FALSE])
  x = scale(x, center = control$center, scale = control$scale)
  ## Recombine the data
  data = data[, setdiff(cns, nums), drop = FALSE]
  data = cbind(data, as.data.frame(x))
  return(data)
}
lrn = makeLearner("regr.nnet", trace = FALSE, decay = 1e-02)
lrn = makePreprocWrapper(lrn, train = trainfun, predict = predictfun,
  par.vals = list(center = TRUE, scale = TRUE))
lrn
rdesc = makeResampleDesc("CV", iters = 3)

r = resample(lrn, bh.task, resampling = rdesc, show.info = FALSE)
r

lrn = makeLearner("regr.nnet", trace = FALSE, decay = 1e-02)
r = resample(lrn, bh.task, resampling = rdesc, show.info = FALSE)
r
lrn = makeLearner("regr.nnet", trace = FALSE)
lrn = makePreprocWrapper(lrn, train = trainfun, predict = predictfun,
  par.set = makeParamSet(
    makeLogicalLearnerParam("center"),
    makeLogicalLearnerParam("scale")
  ),
  par.vals = list(center = TRUE, scale = TRUE))

lrn

getParamSet(lrn)
rdesc = makeResampleDesc("Holdout")
ps = makeParamSet(
  makeDiscreteParam("decay", c(0, 0.05, 0.1)),
  makeLogicalParam("center"),
  makeLogicalParam("scale")
)
ctrl = makeTuneControlGrid()
res = tuneParams(lrn, bh.task, rdesc, par.set = ps, control = ctrl, show.info = FALSE)

res

as.data.frame(res$opt.path)
makePreprocWrapperScale = function(learner, center = TRUE, scale = TRUE) {
  trainfun = function(data, target, args = list(center, scale)) {
    cns = colnames(data)
    nums = setdiff(cns[sapply(data, is.numeric)], target)
    x = as.matrix(data[, nums, drop = FALSE])
    x = scale(x, center = args$center, scale = args$scale)
    control = args
    if (is.logical(control$center) && control$center)
      control$center = attr(x, "scaled:center")
    if (is.logical(control$scale) && control$scale)
      control$scale = attr(x, "scaled:scale")
    data = data[, setdiff(cns, nums), drop = FALSE]
    data = cbind(data, as.data.frame(x))
    return(list(data = data, control = control))
  }
  predictfun = function(data, target, args, control) {
    cns = colnames(data)
    nums = cns[sapply(data, is.numeric)]
    x = as.matrix(data[, nums, drop = FALSE])
    x = scale(x, center = control$center, scale = control$scale)
    data = data[, setdiff(cns, nums), drop = FALSE]
    data = cbind(data, as.data.frame(x))
    return(data)
  }
  makePreprocWrapper(
    learner,
    train = trainfun,
    predict = predictfun,
    par.set = makeParamSet(
      makeLogicalLearnerParam("center"),
      makeLogicalLearnerParam("scale")
    ),
    par.vals = list(center = center, scale = scale)
  )
}

lrn = makePreprocWrapperScale("classif.lda")
train(lrn, iris.task)
