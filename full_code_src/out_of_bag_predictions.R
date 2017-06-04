listLearners(obj = NA, properties = "oobpreds")[c("class", "package")]
lrn = makeLearner("classif.ranger", predict.type = "prob", predict.threshold = 0.6)
mod = train(lrn, sonar.task)
oob = getOOBPreds(mod, sonar.task)
oob

performance(oob, measures = list(auc, mmce))
