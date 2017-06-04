imputeLOCF = function() {
  makeImputeMethod(
    learn = function(data, target, col) {
      x = data[[col]]
      ind = is.na(x)
      dind = diff(ind)
      lastValue = which(dind == 1)  # position of the last observed value previous to NA
      lastNA = which(dind == -1)    # position of the last of potentially several consecutive NA's
      values = x[lastValue]         # last observed value previous to NA
      times = lastNA - lastValue    # number of consecutive NA's
      return(list(values = values, times = times))
    },
    impute = function(data, target, col, values, times) {
      x = data[[col]]
      replace(x, is.na(x), rep(values, times))
    }
  )
}
data(airquality)
imp = impute(airquality, cols = list(Ozone = imputeLOCF(), Solar.R = imputeLOCF()),
  dummy.cols = c("Ozone", "Solar.R"))
head(imp$data, 10)
