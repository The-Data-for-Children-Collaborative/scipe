library(hqreg)
library(Metrics)

# read args
args = commandArgs(trailingOnly=TRUE)
gamma = as.double(args[1])
path = args[2]

X_train <- as.matrix(read.csv(paste(path,"X_train.csv",sep=""), sep = ",", header=FALSE))
y_train <- read.csv(paste(path,"y_train.csv",sep=""), header=FALSE)[,1]

fit <- hqreg(X_train,y_train,gamma=gamma)

saveRDS(fit, paste(path,"model.rds",sep="")) # serialize model

quit(save="no")
