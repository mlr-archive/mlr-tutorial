rdesc = makeResampleDesc("Holdout")
r = resample("classif.multinom", iris.task, rdesc)
lrn = makeLearner("classif.multinom", config = list(show.learner.output = FALSE))
r = resample(lrn, iris.task, rdesc, show.info = FALSE)
configureMlr(show.learner.output = FALSE, show.info = FALSE)
r = resample("classif.multinom", iris.task, rdesc)
getMlrOptions()
configureMlr()
getMlrOptions()
## Support Vector Machine with linear kernel and new parameter 'newParam'
lrn = makeLearner("classif.ksvm", kernel = "vanilladot", newParam = 3)

## Turn off parameter checking completely
configureMlr(on.par.without.desc = "quiet")
lrn = makeLearner("classif.ksvm", kernel = "vanilladot", newParam = 3)
train(lrn, iris.task)

## Option "quiet" also masks typos
lrn = makeLearner("classif.ksvm", kernl = "vanilladot")
train(lrn, iris.task)

## Alternatively turn off parameter checking, but still see warnings
configureMlr(on.par.without.desc = "warn")
lrn = makeLearner("classif.ksvm", kernl = "vanilladot", newParam = 3)

train(lrn, iris.task)
## This call gives an error caused by the low number of observations in class "virginica"
train("classif.qda", task = iris.task, subset = 1:104)

## Get a warning instead of an error
configureMlr(on.learner.error = "warn")
mod = train("classif.qda", task = iris.task, subset = 1:104)

mod

## mod is an object of class FailureModel
isFailureModel(mod)

## Retrieve the error message
getFailureModelMsg(mod)

## predict and performance return NA's
pred = predict(mod, iris.task)
pred

performance(pred)
