## Two learners to be compared
lrns = list(makeLearner("classif.lda"), makeLearner("classif.rpart"))

## Choose the resampling strategy
rdesc = makeResampleDesc("Holdout")

## Conduct the benchmark experiment
bmr = benchmark(lrns, sonar.task, rdesc)

bmr
## Vector of strings
lrns = c("classif.lda", "classif.rpart")

## A mixed list of Learner objects and strings works, too
lrns = list(makeLearner("classif.lda", predict.type = "prob"), "classif.rpart")

bmr = benchmark(lrns, sonar.task, rdesc)

bmr
getBMRPerformances(bmr)

getBMRAggrPerformances(bmr)
getBMRPerformances(bmr, drop = TRUE)
getBMRPerformances(bmr, as.df = TRUE)

getBMRAggrPerformances(bmr, as.df = TRUE)
getBMRPredictions(bmr)

head(getBMRPredictions(bmr, as.df = TRUE))
head(getBMRPredictions(bmr, learner.ids = "classif.rpart", as.df = TRUE))
getBMRTaskIds(bmr)

getBMRLearnerIds(bmr)

getBMRMeasureIds(bmr)
getBMRModels(bmr)

getBMRModels(bmr, drop = TRUE)

getBMRModels(bmr, learner.ids = "classif.lda")
getBMRLearners(bmr)

getBMRMeasures(bmr)
## First benchmark result
bmr

## Benchmark experiment for the additional learners
lrns2 = list(makeLearner("classif.randomForest"), makeLearner("classif.qda"))
bmr2 = benchmark(lrns2, sonar.task, rdesc, show.info = FALSE)
bmr2

## Merge the results
mergeBenchmarkResults(list(bmr, bmr2))
rin = getBMRPredictions(bmr)[[1]][[1]]$instance
rin

## Benchmark experiment for the additional random forest
bmr3 = benchmark(lrns2, sonar.task, rin, show.info = FALSE)
bmr3

## Merge the results
mergeBenchmarkResults(list(bmr, bmr3))
set.seed(4444)
## Create a list of learners
lrns = list(
  makeLearner("classif.lda", id = "lda"),
  makeLearner("classif.rpart", id = "rpart"),
  makeLearner("classif.randomForest", id = "randomForest")
)

## Get additional Tasks from package mlbench
ring.task = convertMLBenchObjToTask("mlbench.ringnorm", n = 600)
wave.task = convertMLBenchObjToTask("mlbench.waveform", n = 600)

tasks = list(iris.task, sonar.task, pid.task, ring.task, wave.task)
rdesc = makeResampleDesc("CV", iters = 10)
meas = list(mmce, ber, timetrain)
bmr = benchmark(lrns, tasks, rdesc, meas, show.info = FALSE)
bmr
perf = getBMRPerformances(bmr, as.df = TRUE)
head(perf)
plotBMRBoxplots(bmr, measure = mmce)
plotBMRBoxplots(bmr, measure = ber, style = "violin", pretty.names = FALSE) +
  aes(color = learner.id) +
  theme(strip.text.x = element_text(size = 8))
mmce$name

mmce$id

getBMRLearnerIds(bmr)

getBMRLearnerShortNames(bmr)
plt = plotBMRBoxplots(bmr, measure = mmce)
head(plt$data)

levels(plt$data$task.id) = c("Iris", "Ringnorm", "Waveform", "Diabetes", "Sonar")
levels(plt$data$learner.id) = c("LDA", "CART", "RF")

plt + ylab("Error rate")
plotBMRSummary(bmr)
m = convertBMRToRankMatrix(bmr, mmce)
m
plotBMRRanksAsBarChart(bmr, pos = "tile")
plotBMRSummary(bmr, trafo = "rank", jitter = 0)
plotBMRRanksAsBarChart(bmr)
plotBMRRanksAsBarChart(bmr, pos = "dodge")
friedmanTestBMR(bmr)
friedmanPostHocTestBMR(bmr, p.value = 0.1)
## Nemenyi test
g = generateCritDifferencesData(bmr, p.value = 0.1, test = "nemenyi")
plotCritDifferences(g) + coord_cartesian(xlim = c(-1,5), ylim = c(0,2))
## Bonferroni-Dunn test
g = generateCritDifferencesData(bmr, p.value = 0.1, test = "bd", baseline = "randomForest")
plotCritDifferences(g) + coord_cartesian(xlim = c(-1,5), ylim = c(0,2))
perf = getBMRPerformances(bmr, as.df = TRUE)

## Density plots for two tasks
qplot(mmce, colour = learner.id, facets = . ~ task.id,
  data = perf[perf$task.id %in% c("iris-example", "Sonar-example"),], geom = "density") +
  theme(strip.text.x = element_text(size = 8))
## Compare mmce and timetrain
df = reshape2::melt(perf, id.vars = c("task.id", "learner.id", "iter"))
df = df[df$variable != "ber",]
head(df)

qplot(variable, value, data = df, colour = learner.id, geom = "boxplot",
  xlab = "measure", ylab = "performance") +
  facet_wrap(~ task.id, nrow = 2)
perf = getBMRPerformances(bmr, task.id = "Sonar-example", as.df = TRUE)
df = reshape2::melt(perf, id.vars = c("task.id", "learner.id", "iter"))
df = df[df$variable == "mmce",]
df = reshape2::dcast(df, task.id + iter ~ variable + learner.id)
head(df)

GGally::ggpairs(df, 3:5)
