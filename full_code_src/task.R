data(BostonHousing, package = "mlbench")
regr.task = makeRegrTask(id = "bh", data = BostonHousing, target = "medv")
regr.task
data(BreastCancer, package = "mlbench")
df = BreastCancer
df$Id = NULL
classif.task = makeClassifTask(id = "BreastCancer", data = df, target = "Class")
classif.task
classif.task = makeClassifTask(id = "BreastCancer", data = df, target = "Class", positive = "malignant")
data(lung, package = "survival")
lung$status = (lung$status == 2) # convert to logical
surv.task = makeSurvTask(data = lung, target = c("time", "status"))
surv.task
yeast = getTaskData(yeast.task)

labels = colnames(yeast)[1:14]
yeast.task = makeMultilabelTask(id = "multi", data = yeast, target = labels)
yeast.task
data(mtcars, package = "datasets")
cluster.task = makeClusterTask(data = mtcars)
cluster.task
df = iris
cost = matrix(runif(150 * 3, 0, 2000), 150) * (1 - diag(3))[df$Species,]
df$Species = NULL

costsens.task = makeCostSensTask(data = df, cost = cost)
costsens.task
getTaskDesc(classif.task)
## Get the ID
getTaskId(classif.task)

## Get the type of task
getTaskType(classif.task)

## Get the names of the target columns
getTaskTargetNames(classif.task)

## Get the number of observations
getTaskSize(classif.task)

## Get the number of input variables
getTaskNFeats(classif.task)

## Get the class levels in classif.task
getTaskClassLevels(classif.task)
## Accessing the data set in classif.task
str(getTaskData(classif.task))

## Get the names of the input variables in cluster.task
getTaskFeatureNames(cluster.task)

## Get the values of the target variables in surv.task
head(getTaskTargets(surv.task))

## Get the cost matrix in costsens.task
head(getTaskCosts(costsens.task))
## Select observations and/or features
cluster.task = subsetTask(cluster.task, subset = 4:17)

## It may happen, especially after selecting observations, that features are constant.
## These should be removed.
removeConstantFeatures(cluster.task)

## Remove selected features
dropFeatures(surv.task, c("meal.cal", "wt.loss"))

## Standardize numerical features
task = normalizeFeatures(cluster.task, method = "range")
summary(getTaskData(task))
