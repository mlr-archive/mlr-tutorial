## Generate the task
task = makeClassifTask(data = iris, target = "Species")

## Generate the learner
lrn = makeLearner("classif.lda")

## Train the learner
mod = train(lrn, task)
mod
mod = train("classif.lda", task)
mod
mod = train("surv.coxph", lung.task)
mod
data(ruspini, package = "cluster")
plot(y ~ x, ruspini)
## Generate the task
ruspini.task = makeClusterTask(data = ruspini)

## Generate the learner
lrn = makeLearner("cluster.kmeans", centers = 4)

## Train the learner
mod = train(lrn, ruspini.task)
mod

## Peak into mod
names(mod)

mod$learner

mod$features

mod$time

## Extract the fitted model
getLearnerModel(mod)
## Get the number of observations
n = getTaskSize(bh.task)

## Use 1/3 of the observations for training
train.set = sample(n, size = n/3)

## Train the learner
mod = train("regr.lm", bh.task, subset = train.set)
mod
## Calculate the observation weights
target = getTaskTargets(bc.task)
tab = as.numeric(table(target))
w = 1/tab[target]

train("classif.rpart", task = bc.task, weights = w)
