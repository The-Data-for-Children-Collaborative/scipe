library(hqreg)
X_train <- as.matrix(read.csv("./csv/X_train.csv", sep = ",", header=FALSE))
y_train <- read.csv("./csv/y_train.csv", header=FALSE)[,1]
X_val <- as.matrix(read.csv("./csv/X_val.csv", sep = ",", header=FALSE))

print(dim(X_val))

#print(length(y_train))
#print(typeof(y_train))
#print(dim(X_train))
#print(typeof(X_train))

fit <- hqreg(X_train,y_train)
y_pred <- predict(fit, X_val, lambda = 0.1) # specify for now

file.create("./csv/y_pred.csv")
write.csv(y_pred,"./csv/y_pred.csv",row.names=FALSE)

quit(save="no")
