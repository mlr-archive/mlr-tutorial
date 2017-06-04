## Regression learners that can deal with missing values
listLearners("regr", properties = "missings")[c("class", "package")]
data(airquality)
summary(airquality)
airq = airquality
ind = sample(nrow(airq), 10)
airq$Wind[ind] = NA
airq$Wind = cut(airq$Wind, c(0,8,16,24))
summary(airq)
imp = impute(airq, classes = list(integer = imputeMean(), factor = imputeMode()),
  dummy.classes = "integer")
head(imp$data, 10)
imp$desc
airq = subset(airq, select = 1:4)
airq.train = airq[1:100,]
airq.test = airq[-c(1:100),]
imp = impute(airq.train, target = "Ozone", cols = list(Solar.R = imputeHist(),
  Wind = imputeLearner("classif.rpart")), dummy.cols = c("Solar.R", "Wind"))
summary(imp$data)

imp$desc
airq.test.imp = reimpute(airq.test, imp$desc)
head(airq.test.imp)
lrn = makeImputeWrapper("regr.lm", cols = list(Solar.R = imputeHist(),
  Wind = imputeLearner("classif.rpart")), dummy.cols = c("Solar.R", "Wind"))
lrn
airq = subset(airq, subset = !is.na(airq$Ozone))
task = makeRegrTask(data = airq, target = "Ozone")
rdesc = makeResampleDesc("CV", iters = 3)
r = resample(lrn, task, resampling = rdesc, show.info = FALSE, models = TRUE)
r$aggr
lapply(r$models, getLearnerModel, more.unwrap = TRUE)
