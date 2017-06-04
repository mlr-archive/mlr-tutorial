# Not strictly necessary, but otherwise we might get NAs later on
## if 'rpart' is not installed.
library("rpart")
## 3-fold cross-validation
rdesc = makeResampleDesc("CV", iters = 3)
rdesc
## Holdout estimation
rdesc = makeResampleDesc("Holdout")
rdesc
hout

cv3
## Specify the resampling strategy (3-fold cross-validation)
rdesc = makeResampleDesc("CV", iters = 3)

## Calculate the performance
r = resample("regr.lm", bh.task, rdesc)

r
## Peak into r
names(r)

r$aggr

r$measures.test
## Subsampling with 5 iterations and default split ratio 2/3
rdesc = makeResampleDesc("Subsample", iters = 5)

## Subsampling with 5 iterations and 4/5 training data
rdesc = makeResampleDesc("Subsample", iters = 5, split = 4/5)

## Classification tree with information splitting criterion
lrn = makeLearner("classif.rpart", parms = list(split = "information"))

## Calculate the performance measures
r = resample(lrn, sonar.task, rdesc, measures = list(mmce, fpr, fnr, timetrain))

r
## Add balanced error rate (ber) and time used to predict
addRRMeasure(r, list(ber, timepredict))
r = resample("classif.rpart", parms = list(split = "information"), sonar.task, rdesc,
  measures = list(mmce, fpr, fnr, timetrain), show.info = FALSE)

r
r$pred

pred = getRRPredictions(r)
pred
head(as.data.frame(pred))

head(getPredictionTruth(pred))

head(getPredictionResponse(pred))
## Make predictions on both training and test sets
rdesc = makeResampleDesc("Holdout", predict = "both")

r = resample("classif.lda", iris.task, rdesc, show.info = FALSE)
r

r$measures.train
predList = getRRPredictionList(r)
predList
## 3-fold cross-validation
rdesc = makeResampleDesc("CV", iters = 3)

r = resample("surv.coxph", lung.task, rdesc, show.info = FALSE, models = TRUE)
r$models
## 3-fold cross-validation
rdesc = makeResampleDesc("CV", iters = 3)

## Extract the compute cluster centers
r = resample("cluster.kmeans", mtcars.task, rdesc, show.info = FALSE,
  centers = 3, extract = function(x) getLearnerModel(x)$centers)
r$extract
## Extract the variable importance in a regression tree
r = resample("regr.rpart", bh.task, rdesc, show.info = FALSE, extract = getFeatureImportance)
r$extract
## 3-fold cross-validation
rdesc = makeResampleDesc("CV", iters = 3, stratify = TRUE)

r = resample("classif.lda", iris.task, rdesc, show.info = FALSE)
r
rdesc = makeResampleDesc("CV", iters = 3, stratify.cols = "chas")

r = resample("regr.rpart", bh.task, rdesc, show.info = FALSE)
r
## 5 blocks containing 30 observations each
task = makeClassifTask(data = iris, target = "Species", blocking = factor(rep(1:5, each = 30)))
task
rdesc = makeResampleDesc("CV", iters = 3)
rdesc

str(rdesc)

str(makeResampleDesc("Subsample", stratify.cols = "chas"))
## Create a resample instance based an a task
rin = makeResampleInstance(rdesc, iris.task)
rin

str(rin)

## Create a resample instance given the size of the data set
rin = makeResampleInstance(rdesc, size = nrow(iris))
str(rin)

## Access the indices of the training observations in iteration 3
rin$train.inds[[3]]
rdesc = makeResampleDesc("CV", iters = 3)
rin = makeResampleInstance(rdesc, task = iris.task)

## Calculate the performance of two learners based on the same resample instance
r.lda = resample("classif.lda", iris.task, rin, show.info = FALSE)
r.rpart = resample("classif.rpart", iris.task, rin, show.info = FALSE)
r.lda$aggr

r.rpart$aggr
rin = makeFixedHoldoutInstance(train.inds = 1:100, test.inds = 101:150, size = 150)
rin
## Mean misclassification error
mmce$aggr

mmce$aggr$fun

## Root mean square error
rmse$aggr

rmse$aggr$fun
mseTestMedian = setAggregation(mse, test.median)
mseTestMin = setAggregation(mse, test.min)
mseTestMax = setAggregation(mse, test.max)

mseTestMedian

rdesc = makeResampleDesc("CV", iters = 3)
r = resample("regr.lm", bh.task, rdesc, measures = list(mse, mseTestMedian, mseTestMin, mseTestMax))

r

r$aggr
mmceTrainMean = setAggregation(mmce, train.mean)
rdesc = makeResampleDesc("CV", iters = 3, predict = "both")
r = resample("classif.rpart", iris.task, rdesc, measures = list(mmce, mmceTrainMean))

r$measures.train

r$aggr
## Use bootstrap as resampling strategy and predict on both train and test sets
rdesc = makeResampleDesc("Bootstrap", predict = "both", iters = 10)

## Set aggregation schemes for b632 and b632+ bootstrap
mmceB632 = setAggregation(mmce, b632)
mmceB632plus = setAggregation(mmce, b632plus)

mmceB632

r = resample("classif.rpart", iris.task, rdesc, measures = list(mmce, mmceB632, mmceB632plus),
  show.info = FALSE)
head(r$measures.train)

## Compare misclassification rates for out-of-bag, b632, and b632+ bootstrap
r$aggr
crossval("classif.lda", iris.task, iters = 3, measures = list(mmce, ber))

bootstrapB632plus("regr.lm", bh.task, iters = 3, measures = list(mse, mae))
