data.imbal.train = rbind(
  data.frame(x = rnorm(100, mean = 1), class = "A"),
  data.frame(x = rnorm(5000, mean = 2), class = "B")
)
task = makeClassifTask(data = data.imbal.train, target = "class")
task.over = oversample(task, rate = 8)
task.under = undersample(task, rate = 1/8)

table(getTaskTargets(task))

table(getTaskTargets(task.over))

table(getTaskTargets(task.under))
lrn = makeLearner("classif.rpart", predict.type = "prob")
mod = train(lrn, task)
mod.over = train(lrn, task.over)
mod.under = train(lrn, task.under)
data.imbal.test = rbind(
  data.frame(x = rnorm(10, mean = 1), class = "A"),
  data.frame(x = rnorm(500, mean = 2), class = "B")
)

performance(predict(mod, newdata = data.imbal.test), measures = list(mmce, ber, auc))

performance(predict(mod.over, newdata = data.imbal.test), measures = list(mmce, ber, auc))

performance(predict(mod.under, newdata = data.imbal.test), measures = list(mmce, ber, auc))
lrn.over = makeOversampleWrapper(lrn, osw.rate = 8)
lrn.under = makeUndersampleWrapper(lrn, usw.rate = 1/8)
mod = train(lrn, task)
mod.over = train(lrn.over, task)
mod.under = train(lrn.under, task)

performance(predict(mod, newdata = data.imbal.test), measures = list(mmce, ber, auc))

performance(predict(mod.over, newdata = data.imbal.test), measures = list(mmce, ber, auc))

performance(predict(mod.under, newdata = data.imbal.test), measures = list(mmce, ber, auc))
task.smote = smote(task, rate = 8, nn = 5)
table(getTaskTargets(task))

table(getTaskTargets(task.smote))
lrn.smote = makeSMOTEWrapper(lrn, sw.rate = 8, sw.nn = 5)
mod.smote = train(lrn.smote, task)
performance(predict(mod.smote, newdata = data.imbal.test), measures = list(mmce, ber, auc))
lrn = makeLearner("classif.rpart", predict.type = "response")
obw.lrn = makeOverBaggingWrapper(lrn, obw.rate = 8, obw.iters = 3)
lrn = setPredictType(lrn, "prob")
rdesc = makeResampleDesc("CV", iters = 5)
r1 = resample(learner = lrn, task = task, resampling = rdesc, show.info = FALSE,
  measures = list(mmce, ber, auc))
r1$aggr

obw.lrn = setPredictType(obw.lrn, "prob")
r2 = resample(learner = obw.lrn, task = task, resampling = rdesc, show.info = FALSE,
  measures = list(mmce, ber, auc))
r2$aggr
lrn = makeLearner("classif.randomForest")
obw.lrn = makeOverBaggingWrapper(lrn, obw.rate = 8, obw.iters = 3)

lrn = setPredictType(lrn, "prob")
r1 = resample(learner = lrn, task = task, resampling = rdesc, show.info = FALSE,
  measures = list(mmce, ber, auc))
r1$aggr

obw.lrn = setPredictType(obw.lrn, "prob")
r2 = resample(learner = obw.lrn, task = task, resampling = rdesc, show.info = FALSE,
  measures = list(mmce, ber, auc))
r2$aggr
lrn = makeLearner("classif.logreg")
wcw.lrn = makeWeightedClassesWrapper(lrn, wcw.weight = 0.01)
lrn = makeLearner("classif.ksvm")
wcw.lrn = makeWeightedClassesWrapper(lrn, wcw.weight = 0.01)
