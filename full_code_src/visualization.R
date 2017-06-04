lrn = makeLearner("classif.lda", predict.type = "prob")
n = getTaskSize(sonar.task)
mod = train(lrn, task = sonar.task, subset = seq(1, n, by = 2))
pred = predict(mod, task = sonar.task, subset = seq(2, n, by = 2))
d = generateThreshVsPerfData(pred, measures = list(fpr, fnr, mmce))

class(d)

head(d$data)
plotThreshVsPerf(d)
fpr$name

fpr$id
plt = plotThreshVsPerf(d, pretty.names = FALSE)

## Reshaped version of the underlying data d
head(plt$data)

levels(plt$data$measure)

## Rename and reorder factor levels
plt$data$measure = factor(plt$data$measure, levels = c("mmce", "fpr", "fnr"),
  labels = c("Error rate", "False positive rate", "False negative rate"))
plt = plt + xlab("Cutoff") + ylab("Performance")
plt
plt = plotThreshVsPerf(d, pretty.names = FALSE)

measure_names = c(
  fpr = "False positive rate",
  fnr = "False negative rate",
  mmce = "Error rate"
)
## Manipulate the measure names via the labeller function and
## arrange the panels in two columns and choose common axis limits for all panels
plt = plt + facet_wrap( ~ measure, labeller = labeller(measure = measure_names), ncol = 2)
plt = plt + xlab("Decision threshold") + ylab("Performance")
plt
ggplot(d$data, aes(threshold, fpr)) + geom_line()
lattice::xyplot(fpr + fnr + mmce ~ threshold, data = d$data, type = "l", ylab = "performance",
  outer = TRUE, scales = list(relation = "free"),
  strip = strip.custom(factor.levels = sapply(d$measures, function(x) x$name),
    par.strip.text = list(cex = 0.8)))
sonar = getTaskData(sonar.task)
pd = generatePartialDependenceData(mod, sonar, "V11")
plt = plotPartialDependence(pd)
head(plt$data)

plt
plot(Probability ~ Value, data = plt$data, type = "b", xlab = plt$data$Feature[1])
