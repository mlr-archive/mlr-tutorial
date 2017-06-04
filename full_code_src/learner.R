## Classification tree, set it up for predicting probabilities
classif.lrn = makeLearner("classif.randomForest", predict.type = "prob", fix.factors.prediction = TRUE)

## Regression gradient boosting machine, specify hyperparameters via a list
regr.lrn = makeLearner("regr.gbm", par.vals = list(n.trees = 500, interaction.depth = 3))

## Cox proportional hazards model with custom name
surv.lrn = makeLearner("surv.coxph", id = "cph")

## K-means with 5 clusters
cluster.lrn = makeLearner("cluster.kmeans", centers = 5)

## Multilabel Random Ferns classification algorithm
multilabel.lrn = makeLearner("multilabel.rFerns")
classif.lrn

surv.lrn
## Get the configured hyperparameter settings that deviate from the defaults
cluster.lrn$par.vals

## Get the set of hyperparameters
classif.lrn$par.set

## Get the type of prediction
regr.lrn$predict.type
## Get current hyperparameter settings
getHyperPars(cluster.lrn)

## Get a description of all possible hyperparameter settings
getParamSet(classif.lrn)
getParamSet("classif.randomForest")
## Get object's id
getLearnerId(surv.lrn)

## Get the short name
getLearnerShortName(classif.lrn)

## Get the type of the learner
getLearnerType(multilabel.lrn)

## Get required packages
getLearnerPackages(cluster.lrn)
## Change the ID
surv.lrn = setLearnerId(surv.lrn, "CoxModel")
surv.lrn

## Change the prediction type, predict a factor with class labels instead of probabilities
classif.lrn = setPredictType(classif.lrn, "response")

## Change hyperparameter values
cluster.lrn = setHyperPars(cluster.lrn, centers = 4)

## Go back to default hyperparameter values
regr.lrn = removeHyperPars(regr.lrn, c("n.trees", "interaction.depth"))
## List everything in mlr
lrns = listLearners()
head(lrns[c("class", "package")])

## List classifiers that can output probabilities
lrns = listLearners("classif", properties = "prob")
head(lrns[c("class", "package")])

## List classifiers that can be applied to iris (i.e., multiclass) and output probabilities
lrns = listLearners(iris.task, properties = "prob")
head(lrns[c("class", "package")])

## The calls above return character vectors, but you can also create learner objects
head(listLearners("cluster", create = TRUE), 2)
