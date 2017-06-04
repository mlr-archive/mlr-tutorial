class(makeLearner(cl = "classif.lda"))

class(makeLearner(cl = "regr.lm"))

class(makeLearner(cl = "surv.coxph"))

class(makeLearner(cl = "cluster.kmeans"))

class(makeLearner(cl = "multilabel.rFerns"))
makeRLearner.classif.lda = function() {
  makeRLearnerClassif(
    cl = "classif.lda",
    package = "MASS",
    par.set = makeParamSet(
      makeDiscreteLearnerParam(id = "method", default = "moment", values = c("moment", "mle", "mve", "t")),
      makeNumericLearnerParam(id = "nu", lower = 2, requires = quote(method == "t")),
      makeNumericLearnerParam(id = "tol", default = 1e-4, lower = 0),
      makeDiscreteLearnerParam(id = "predict.method", values = c("plug-in", "predictive", "debiased"),
        default = "plug-in", when = "predict"),
      makeLogicalLearnerParam(id = "CV", default = FALSE, tunable = FALSE)
    ),
    properties = c("twoclass", "multiclass", "numerics", "factors", "prob"),
    name = "Linear Discriminant Analysis",
    short.name = "lda",
    note = "Learner param 'predict.method' maps to 'method' in predict.lda."
  )
}
## function(.learner, .task, .subset, .weights = NULL, ...) { }
## trainLearner.classif.lda = function(.learner, .task, .subset, .weights = NULL,  ...) {
##   f = getTaskFormula(.task)
##   MASS::lda(f, data = getTaskData(.task, .subset), ...)
## }
## function(.learner, .model, .newdata, ...) { }
## predictLearner.classif.lda = function(.learner, .model, .newdata, predict.method = "plug-in", ...) {
##   p = predict(.model$learner.model, newdata = .newdata, method = predict.method, ...)
##   if(.learner$predict.type == "response")
##     return(p$class)
##   else
##     return(p$posterior)
## }
makeRLearner.regr.earth = function() {
  makeRLearnerRegr(
    cl = "regr.earth",
    package = "earth",
    par.set = makeParamSet(
      makeLogicalLearnerParam(id = "keepxy", default = FALSE, tunable = FALSE),
      makeNumericLearnerParam(id = "trace", default = 0, upper = 10, tunable = FALSE),
      makeIntegerLearnerParam(id = "degree", default = 1L, lower = 1L),
      makeNumericLearnerParam(id = "penalty"),
      makeIntegerLearnerParam(id = "nk", lower = 0L),
      makeNumericLearnerParam(id = "thres", default = 0.001),
      makeIntegerLearnerParam(id = "minspan", default = 0L),
      makeIntegerLearnerParam(id = "endspan", default = 0L),
      makeNumericLearnerParam(id = "newvar.penalty", default = 0),
      makeIntegerLearnerParam(id = "fast.k", default = 20L, lower = 0L),
      makeNumericLearnerParam(id = "fast.beta", default = 1),
      makeDiscreteLearnerParam(id = "pmethod", default = "backward",
        values = c("backward", "none", "exhaustive", "forward", "seqrep", "cv")),
      makeIntegerLearnerParam(id = "nprune")
    ),
    properties = c("numerics", "factors"),
    name = "Multivariate Adaptive Regression Splines",
    short.name = "earth",
    note = ""
  )
}
## trainLearner.regr.earth = function(.learner, .task, .subset, .weights = NULL,  ...) {
##   f = getTaskFormula(.task)
##   earth::earth(f, data = getTaskData(.task, .subset), ...)
## }
## predictLearner.regr.earth = function(.learner, .model, .newdata, ...) {
##   predict(.model$learner.model, newdata = .newdata)[, 1L]
## }
makeRLearner.surv.coxph = function() {
  makeRLearnerSurv(
    cl = "surv.coxph",
    package = "survival",
    par.set = makeParamSet(
      makeDiscreteLearnerParam(id = "ties", default = "efron", values = c("efron", "breslow", "exact")),
      makeLogicalLearnerParam(id = "singular.ok", default = TRUE),
      makeNumericLearnerParam(id = "eps", default = 1e-09, lower = 0),
      makeNumericLearnerParam(id = "toler.chol", default = .Machine$double.eps^0.75, lower = 0),
      makeIntegerLearnerParam(id = "iter.max", default = 20L, lower = 1L),
      makeNumericLearnerParam(id = "toler.inf", default = sqrt(.Machine$double.eps^0.75), lower = 0),
      makeIntegerLearnerParam(id = "outer.max", default = 10L, lower = 1L),
      makeLogicalLearnerParam(id = "model", default = FALSE, tunable = FALSE),
      makeLogicalLearnerParam(id = "x", default = FALSE, tunable = FALSE),
      makeLogicalLearnerParam(id = "y", default = TRUE, tunable = FALSE)
    ),
    properties = c("missings", "numerics", "factors", "weights", "prob", "rcens"),
    name = "Cox Proportional Hazard Model",
    short.name = "coxph",
    note = ""
  )
}
## trainLearner.surv.coxph = function(.learner, .task, .subset, .weights = NULL,  ...) {
##   f = getTaskFormula(.task)
##   data = getTaskData(.task, subset = .subset)
##   if (is.null(.weights)) {
##     mod = survival::coxph(formula = f, data = data, ...)
##   } else  {
##     mod = survival::coxph(formula = f, data = data, weights = .weights, ...)
##   }
##   if (.learner$predict.type == "prob")
##     mod = attachTrainingInfo(mod, list(surv.range = range(getTaskTargets(.task)[, 1L])))
##   mod
## }
## predictLearner.surv.coxph = function(.learner, .model, .newdata, ...) {
##   if(.learner$predict.type == "response") {
##     predict(.model$learner.model, newdata = .newdata, type = "lp", ...)
##   } else if (.learner$predict.type == "prob") {
##     surv.range = getTrainingInfo(.model$learner.model)$surv.range
##     times = seq(from = surv.range[1L], to = surv.range[2L], length.out = 1000)
##     t(summary(survival::survfit(.model$learner.model, newdata = .newdata, se.fit = FALSE, conf.int = FALSE), times = times)$surv)
##   } else {
##     stop("Unknown predict type")
##   }
## }
makeRLearner.cluster.FarthestFirst = function() {
  makeRLearnerCluster(
    cl = "cluster.FarthestFirst",
    package = "RWeka",
    par.set = makeParamSet(
      makeIntegerLearnerParam(id = "N", default = 2L, lower = 1L),
      makeIntegerLearnerParam(id = "S", default = 1L, lower = 1L),
      makeLogicalLearnerParam(id = "output-debug-info", default = FALSE, tunable = FALSE)
    ),
    properties = c("numerics"),
    name = "FarthestFirst Clustering Algorithm",
    short.name = "farthestfirst"
  )
}
## trainLearner.cluster.FarthestFirst = function(.learner, .task, .subset, .weights = NULL,  ...) {
##   ctrl = RWeka::Weka_control(...)
##   RWeka::FarthestFirst(getTaskData(.task, .subset), control = ctrl)
## }
## predictLearner.cluster.FarthestFirst = function(.learner, .model, .newdata, ...) {
##   # RWeka returns cluster indices (i.e. starting from 0, which some tools don't like
##   as.integer(predict(.model$learner.model, .newdata, ...)) + 1L
## }
makeRLearner.multilabel.rFerns = function() {
  makeRLearnerMultilabel(
    cl = "multilabel.rFerns",
    package = "rFerns",
    par.set = makeParamSet(
      makeIntegerLearnerParam(id = "depth", default = 5L),
      makeIntegerLearnerParam(id = "ferns", default = 1000L)
    ),
    properties = c("numerics", "factors", "ordered"),
    name = "Random ferns",
    short.name = "rFerns",
    note = ""
  )
}
## trainLearner.multilabel.rFerns = function(.learner, .task, .subset, .weights = NULL, ...) {
##   d = getTaskData(.task, .subset, target.extra = TRUE)
##   rFerns::rFerns(x = d$data, y = as.matrix(d$target), ...)
## }
## predictLearner.multilabel.rFerns = function(.learner, .model, .newdata, ...) {
##   as.matrix(predict(.model$learner.model, .newdata, ...))
## }
## getFeatureImportanceLearner.classif.rpart = function(.learner, .model, ...) {
##   mod = getLearnerModel(.model)
##   mod$variable.importance
## }
## getFeatureImportanceLearner.classif.randomForestSRC = function(.learner, .model, ...) {
##   mod = getLearnerModel(.model)
##   randomForestSRC::vimp(mod, ...)$importance[, "all"]
## }
## NA
## registerS3method("makeRLearner", "<awesome_new_learner_class>", makeRLearner.<awesome_new_learner_class>)
## registerS3method("trainLearner", "<awesome_new_learner_class>", trainLearner.<awesome_new_learner_class>)
## registerS3method("predictLearner", "<awesome_new_learner_class>", predictLearner.<awesome_new_learner_class>)
## registerS3method("getFeatureImportanceLearner", "<awesome_new_learner_class>",
##   getFeatureImportanceLearner.<awesome_new_learner_class>)
## test_that("classif_lda", {
##   requirePackagesOrSkip("MASS", default.method = "load")
## 
##   set.seed(getOption("mlr.debug.seed"))
##   m = MASS::lda(formula = multiclass.formula, data = multiclass.train)
##   set.seed(getOption("mlr.debug.seed"))
##   p = predict(m, newdata = multiclass.test)
## 
##   testSimple("classif.lda", multiclass.df, multiclass.target, multiclass.train.inds, p$class)
##   testProb("classif.lda", multiclass.df, multiclass.target, multiclass.train.inds, p$posterior)
## })
## test_that("regr_randomForest", {
##   requirePackagesOrSkip("randomForest", default.method = "load")
## 
##   parset.list = list(
##     list(),
##     list(ntree = 5, mtry = 2),
##     list(ntree = 5, mtry = 4),
##     list(proximity = TRUE, oob.prox = TRUE),
##     list(nPerm = 3)
##   )
## 
##   old.predicts.list = list()
## 
##   for (i in 1:length(parset.list)) {
##     parset = parset.list[[i]]
##     pars = list(formula = regr.formula, data = regr.train)
##     pars = c(pars, parset)
##     set.seed(getOption("mlr.debug.seed"))
##     m = do.call(randomForest::randomForest, pars)
##     set.seed(getOption("mlr.debug.seed"))
##     p = predict(m, newdata = regr.test, type = "response")
##     old.predicts.list[[i]] = p
##   }
## 
##   testSimpleParsets("regr.randomForest", regr.df, regr.target,
##     regr.train.inds, old.predicts.list, parset.list)
## })
## devtools::test("mlr", filter = "classif")
