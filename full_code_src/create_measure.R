str(mse)

mse$fun

measureMSE
listMeasureProperties()
str(test.mean)

test.mean$fun
## Define a function that calculates the misclassification rate
my.mmce.fun = function(task, model, pred, feats, extra.args) {
  tb = table(getPredictionResponse(pred), getPredictionTruth(pred))
  1 - sum(diag(tb)) / sum(tb)
}

## Generate the Measure object
my.mmce = makeMeasure(
  id = "my.mmce", name = "My Mean Misclassification Error",
  properties = c("classif", "classif.multi", "req.pred", "req.truth"),
  minimize = TRUE, best = 0, worst = 1,
  fun = my.mmce.fun
)

## Train a learner and make predictions
mod = train("classif.lda", iris.task)
pred = predict(mod, task = iris.task)

## Calculate the performance using the new measure
performance(pred, measures = my.mmce)

## Apparently the result coincides with the mlr implementation
performance(pred, measures = mmce)
## Create the cost matrix
costs = matrix(c(0, 2, 2, 3, 0, 2, 1, 1, 0), ncol = 3)
rownames(costs) = colnames(costs) = getTaskClassLevels(iris.task)

## Encapsulate the cost matrix in a Measure object
my.costs = makeCostMeasure(
  id = "my.costs", name = "My Costs",
  costs = costs,
  minimize = TRUE, best = 0, worst = 3
)

## Train a learner and make a prediction
mod = train("classif.lda", iris.task)
pred = predict(mod, newdata = iris)

## Calculate the average costs
performance(pred, measures = my.costs)
my.range.aggr = makeAggregation(id = "test.range", name = "Test Range",
  properties = "req.test",
  fun = function (task, perf.test, perf.train, measure, group, pred)
    diff(range(perf.test))
)
## mmce with default aggregation scheme test.mean
ms1 = mmce

## mmce with new aggregation scheme my.range.aggr
ms2 = setAggregation(ms1, my.range.aggr)

## Minimum and maximum of the mmce over test sets
ms1min = setAggregation(ms1, test.min)
ms1max = setAggregation(ms1, test.max)

## Feature selection
rdesc = makeResampleDesc("CV", iters = 3)
res = selectFeatures("classif.rpart", iris.task, rdesc, measures = list(ms1, ms2, ms1min, ms1max),
  control = makeFeatSelControlExhaustive(), show.info = FALSE)

## Optimization path, i.e., performances for the 16 possible feature subsets
perf.data = as.data.frame(res$opt.path)
head(perf.data[1:8])

pd = position_jitter(width = 0.005, height = 0)
p = ggplot(aes(x = mmce.test.range, y = mmce.test.mean, ymax = mmce.test.max, ymin = mmce.test.min,
  color = as.factor(Sepal.Width), pch = as.factor(Petal.Width)), data = perf.data) +
  geom_pointrange(position = pd) +
  coord_flip()
print(p)
