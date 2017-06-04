n = getTaskSize(bh.task)
train.set = seq(1, n, by = 2)
test.set = seq(2, n, by = 2)
lrn = makeLearner("regr.gbm", n.trees = 100)
mod = train(lrn, bh.task, subset = train.set)

task.pred = predict(mod, task = bh.task, subset = test.set)
task.pred
n = nrow(iris)
iris.train = iris[seq(1, n, by = 2), -5]
iris.test = iris[seq(2, n, by = 2), -5]
task = makeClusterTask(data = iris.train)
mod = train("cluster.kmeans", task)

newdata.pred = predict(mod, newdata = iris.test)
newdata.pred
## Result of predict with data passed via task argument
head(as.data.frame(task.pred))

## Result of predict with data passed via newdata argument
head(as.data.frame(newdata.pred))
head(getPredictionTruth(task.pred))

head(getPredictionResponse(task.pred))
listLearners("regr", check.packages = FALSE, properties = "se")[c("class", "name")]
## Create learner and specify predict.type
lrn.lm = makeLearner("regr.lm", predict.type = 'se')
mod.lm = train(lrn.lm, bh.task, subset = train.set)
task.pred.lm = predict(mod.lm, task = bh.task, subset = test.set)
task.pred.lm
head(getPredictionSE(task.pred.lm))
lrn = makeLearner("cluster.cmeans", predict.type = "prob")
mod = train(lrn, mtcars.task)

pred = predict(mod, task = mtcars.task)
head(getPredictionProbabilities(pred))
## Linear discriminant analysis on the iris data set
mod = train("classif.lda", task = iris.task)

pred = predict(mod, task = iris.task)
pred
lrn = makeLearner("classif.rpart", predict.type = "prob")
mod = train(lrn, iris.task)

pred = predict(mod, newdata = iris)
head(as.data.frame(pred))
head(getPredictionProbabilities(pred))
calculateConfusionMatrix(pred)
conf.matrix = calculateConfusionMatrix(pred, relative = TRUE)
conf.matrix
conf.matrix$relative.row
calculateConfusionMatrix(pred, relative = TRUE, sums = TRUE)
lrn = makeLearner("classif.rpart", predict.type = "prob")
mod = train(lrn, task = sonar.task)

## Label of the positive class
getTaskDesc(sonar.task)$positive

## Default threshold
pred1 = predict(mod, sonar.task)
pred1$threshold

## Set the threshold value for the positive class
pred2 = setThreshold(pred1, 0.9)
pred2$threshold

pred2

## We can also set the effect in the confusion matrix
calculateConfusionMatrix(pred1)

calculateConfusionMatrix(pred2)
head(getPredictionProbabilities(pred1))

## But we can change that, too
head(getPredictionProbabilities(pred1, cl = c("M", "R")))
lrn = makeLearner("classif.rpart", predict.type = "prob")
mod = train(lrn, iris.task)
pred = predict(mod, newdata = iris)
pred$threshold
table(as.data.frame(pred)$response)
pred = setThreshold(pred, c(setosa = 0.01, versicolor = 50, virginica = 1))
pred$threshold
table(as.data.frame(pred)$response)
lrn = makeLearner("classif.rpart", id = "CART")
plotLearnerPrediction(lrn, task = iris.task)
lrn = makeLearner("cluster.kmeans")
plotLearnerPrediction(lrn, task = mtcars.task, features = c("disp", "drat"), cv = 0)
plotLearnerPrediction("regr.lm", features = "lstat", task = bh.task)
plotLearnerPrediction("regr.lm", features = c("lstat", "rm"), task = bh.task)
