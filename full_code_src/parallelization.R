library("parallelMap")
parallelStartSocket(2)

rdesc = makeResampleDesc("CV", iters = 3)
r = resample("classif.lda", iris.task, rdesc)

parallelStop()
parallelGetRegisteredLevels()
