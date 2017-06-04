## Performance measures for classification with multiple classes
listMeasures("classif", properties = "classif.multi")
## Performance measure suitable for the iris classification task
listMeasures(iris.task)
## Get default measure for iris.task
getDefaultMeasure(iris.task)

## Get the default measure for linear regression
getDefaultMeasure(makeLearner("regr.lm"))
n = getTaskSize(bh.task)
lrn = makeLearner("regr.gbm", n.trees = 1000)
mod = train(lrn, task = bh.task, subset = seq(1, n, 2))
pred = predict(mod, task = bh.task, subset = seq(2, n, 2))

performance(pred)
performance(pred, measures = medse)
performance(pred, measures = list(mse, medse, mae))
performance(pred, measures = timetrain, model = mod)
lrn = makeLearner("cluster.kmeans", centers = 3)
mod = train(lrn, mtcars.task)
pred = predict(mod, task = mtcars.task)

## Calculate the Dunn index
performance(pred, measures = dunn, task = mtcars.task)
lrn = makeLearner("classif.rpart", predict.type = "prob")
mod = train(lrn, task = sonar.task)
pred = predict(mod, task = sonar.task)

performance(pred, measures = auc)
## Mean misclassification error
str(mmce)
lrn = makeLearner("classif.lda", predict.type = "prob")
n = getTaskSize(sonar.task)
mod = train(lrn, task = sonar.task, subset = seq(1, n, by = 2))
pred = predict(mod, task = sonar.task, subset = seq(2, n, by = 2))

## Performance for the default threshold 0.5
performance(pred, measures = list(fpr, fnr, mmce))
## Plot false negative and positive rates as well as the error rate versus the threshold
d = generateThreshVsPerfData(pred, measures = list(fpr, fnr, mmce))
plotThreshVsPerf(d)
## plotThreshVsPerfGGVIS(d)
r = calculateROCMeasures(pred)
r
