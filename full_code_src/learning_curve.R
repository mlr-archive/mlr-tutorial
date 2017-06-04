r = generateLearningCurveData(
  learners = c("classif.rpart", "classif.knn"),
  task = sonar.task,
  percs = seq(0.1, 1, by = 0.2),
  measures = list(tp, fp, tn, fn),
  resampling = makeResampleDesc(method = "CV", iters = 5),
  show.info = FALSE)
plotLearningCurve(r)
lrns = list(
  makeLearner(cl = "classif.ksvm", id = "ksvm1", sigma = 0.2, C = 2),
  makeLearner(cl = "classif.ksvm", id = "ksvm2", sigma = 0.1, C = 1),
  "classif.randomForest"
)
rin = makeResampleDesc(method = "CV", iters = 5)
lc = generateLearningCurveData(learners = lrns, task = sonar.task,
  percs = seq(0.1, 1, by = 0.1), measures = acc,
  resampling = rin, show.info = FALSE)
plotLearningCurve(lc)
rin2 = makeResampleDesc(method = "CV", iters = 5, predict = "both")
lc2 = generateLearningCurveData(learners = lrns, task = sonar.task,
  percs = seq(0.1, 1, by = 0.1),
  measures = list(acc, setAggregation(acc, train.mean)), resampling = rin2,
  show.info = FALSE)
plotLearningCurve(lc2, facet = "learner")
## plotLearningCurveGGVIS(lc2, interaction = "measure")
## plotLearningCurveGGVIS(lc2, interaction = "learner")
