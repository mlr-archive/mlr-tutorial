n = getTaskSize(sonar.task)
train.set = sample(n, size = round(2/3 * n))
test.set = setdiff(seq_len(n), train.set)

lrn1 = makeLearner("classif.lda", predict.type = "prob")
mod1 = train(lrn1, sonar.task, subset = train.set)
pred1 = predict(mod1, task = sonar.task, subset = test.set)
df = generateThreshVsPerfData(pred1, measures = list(fpr, tpr, mmce))
plotROCCurves(df)
performance(pred1, auc)
plotThreshVsPerf(df)
lrn2 = makeLearner("classif.ksvm", predict.type = "prob")
mod2 = train(lrn2, sonar.task, subset = train.set)
pred2 = predict(mod2, task = sonar.task, subset = test.set)
df = generateThreshVsPerfData(list(lda = pred1, ksvm = pred2), measures = list(fpr, tpr))
plotROCCurves(df)
performance(pred2, auc)
qplot(x = fpr, y = tpr, color = learner, data = df$data, geom = "path")
df = generateThreshVsPerfData(list(lda = pred1, ksvm = pred2), measures = list(ppv, tpr, tnr))

## Precision/recall graph
plotROCCurves(df, measures = list(tpr, ppv), diagonal = FALSE)

## Sensitivity/specificity plot
plotROCCurves(df, measures = list(tnr, tpr), diagonal = FALSE)
## Tune wrapper for ksvm
rdesc.inner = makeResampleDesc("Holdout")
ms = list(auc, mmce)
ps = makeParamSet(
  makeDiscreteParam("C", 2^(-1:1))
)
ctrl = makeTuneControlGrid()
lrn2 = makeTuneWrapper(lrn2, rdesc.inner, ms, ps, ctrl, show.info = FALSE)
## Benchmark experiment
lrns = list(lrn1, lrn2)
rdesc.outer = makeResampleDesc("CV", iters = 5)

bmr = benchmark(lrns, tasks = sonar.task, resampling = rdesc.outer, measures = ms, show.info = FALSE)
bmr
df = generateThreshVsPerfData(bmr, measures = list(fpr, tpr, mmce))
plotROCCurves(df)
df = generateThreshVsPerfData(bmr, measures = list(fpr, tpr, mmce), aggregate = FALSE)
plotROCCurves(df)
plotThreshVsPerf(df) +
  theme(strip.text.x = element_text(size = 7))
## Extract predictions
preds = getBMRPredictions(bmr, drop = TRUE)

## Change the class attribute
preds2 = lapply(preds, function(x) {class(x) = "Prediction"; return(x)})

## Draw ROC curves
df = generateThreshVsPerfData(preds2, measures = list(fpr, tpr, mmce))
plotROCCurves(df)
n = getTaskSize(sonar.task)
train.set = sample(n, size = round(2/3 * n))
test.set = setdiff(seq_len(n), train.set)

## Train and predict linear discriminant analysis
lrn1 = makeLearner("classif.lda", predict.type = "prob")
mod1 = train(lrn1, sonar.task, subset = train.set)
pred1 = predict(mod1, task = sonar.task, subset = test.set)
## Convert prediction
ROCRpred1 = asROCRPrediction(pred1)

## Calculate true and false positive rate
ROCRperf1 = ROCR::performance(ROCRpred1, "tpr", "fpr")

## Draw ROC curve
ROCR::plot(ROCRperf1)
## Draw ROC curve
ROCR::plot(ROCRperf1, colorize = TRUE, print.cutoffs.at = seq(0.1, 0.9, 0.1), lwd = 2)

## Draw convex hull of ROC curve
ch = ROCR::performance(ROCRpred1, "rch")
ROCR::plot(ch, add = TRUE, lty = 2)
## Extract predictions
preds = getBMRPredictions(bmr, drop = TRUE)

## Convert predictions
ROCRpreds = lapply(preds, asROCRPrediction)

## Calculate true and false positive rate
ROCRperfs = lapply(ROCRpreds, function(x) ROCR::performance(x, "tpr", "fpr"))
## lda average ROC curve
plot(ROCRperfs[[1]], col = "blue", avg = "vertical", spread.estimate = "stderror",
  show.spread.at = seq(0.1, 0.8, 0.1), plotCI.col = "blue", plotCI.lwd = 2, lwd = 2)
## lda individual ROC curves
plot(ROCRperfs[[1]], col = "blue", lty = 2, lwd = 0.25, add = TRUE)

## ksvm average ROC curve
plot(ROCRperfs[[2]], col = "red", avg = "vertical", spread.estimate = "stderror",
  show.spread.at = seq(0.1, 0.6, 0.1), plotCI.col = "red", plotCI.lwd = 2, lwd = 2, add = TRUE)
## ksvm individual ROC curves
plot(ROCRperfs[[2]], col = "red", lty = 2, lwd = 0.25, add = TRUE)

legend("bottomright", legend = getBMRLearnerIds(bmr), lty = 1, lwd = 2, col = c("blue", "red"))
## Extract and convert predictions
preds = getBMRPredictions(bmr, drop = TRUE)
ROCRpreds = lapply(preds, asROCRPrediction)

## Calculate precision and recall
ROCRperfs = lapply(ROCRpreds, function(x) ROCR::performance(x, "prec", "rec"))

## Draw performance plot
plot(ROCRperfs[[1]], col = "blue", avg = "threshold")
plot(ROCRperfs[[2]], col = "red", avg = "threshold", add = TRUE)
legend("bottomleft", legend = getBMRLearnerIds(bmr), lty = 1, col = c("blue", "red"))
## Extract and convert predictions
preds = getBMRPredictions(bmr, drop = TRUE)
ROCRpreds = lapply(preds, asROCRPrediction)

## Calculate accuracy
ROCRperfs = lapply(ROCRpreds, function(x) ROCR::performance(x, "acc"))

## Plot accuracy versus threshold
plot(ROCRperfs[[1]], avg = "vertical", spread.estimate = "boxplot", lwd = 2, col = "blue",
  show.spread.at = seq(0.1, 0.9, 0.1), ylim = c(0,1), xlab = "Threshold")
z = plotViperCharts(bmr, chart = "rocc", browse = FALSE)
