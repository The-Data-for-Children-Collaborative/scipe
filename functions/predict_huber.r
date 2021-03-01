library(hqreg)
library(Metrics)

# read args
args = commandArgs(trailingOnly=TRUE)
lambda = as.double(args[1])
path = args[2]

# load fitted model
fit = readRDS(paste(path,"model.rds",sep=""))

X <- as.matrix(read.csv(paste(path,"X_pred.csv",sep=""), sep = ",", header=FALSE))
y_pred <- predict(fit, X, lambda = lambda)

file.create(paste(path,"y_pred.csv",sep=""))
write.csv(y_pred,paste(path,"y_pred.csv",sep=""),row.names=FALSE)

quit(save="no")
