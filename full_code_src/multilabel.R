yeast = getTaskData(yeast.task)
labels = colnames(yeast)[1:14]
yeast.task = makeMultilabelTask(id = "multi", data = yeast, target = labels)
yeast.task
lrn.rfsrc = makeLearner("multilabel.randomForestSRC")
lrn.rFerns = makeLearner("multilabel.rFerns")
lrn.rFerns
lrn.br = makeLearner("classif.rpart", predict.type = "prob")
lrn.br = makeMultilabelBinaryRelevanceWrapper(lrn.br)
lrn.br

lrn.br2 = makeMultilabelBinaryRelevanceWrapper("classif.rpart")
lrn.br2
mod = train(lrn.br, yeast.task)
mod = train(lrn.br, yeast.task, subset = 1:1500, weights = rep(1/1500, 1500))
mod

mod2 = train(lrn.rfsrc, yeast.task, subset = 1:100)
mod2
pred = predict(mod, task = yeast.task, subset = 1:10)
pred = predict(mod, newdata = yeast[1501:1600,])
names(as.data.frame(pred))

pred2 = predict(mod2, task = yeast.task)
names(as.data.frame(pred2))
performance(pred)

performance(pred2, measures = list(multilabel.subset01, multilabel.hamloss, multilabel.acc,
  multilabel.f1, timepredict))

listMeasures("multilabel")
rdesc = makeResampleDesc(method = "CV", stratify = FALSE, iters = 3)
r = resample(learner = lrn.br, task = yeast.task, resampling = rdesc, show.info = FALSE)
r

r = resample(learner = lrn.rFerns, task = yeast.task, resampling = rdesc, show.info = FALSE)
r
getMultilabelBinaryPerformances(pred, measures = list(acc, mmce, auc))

getMultilabelBinaryPerformances(r$pred, measures = list(acc, mmce))
