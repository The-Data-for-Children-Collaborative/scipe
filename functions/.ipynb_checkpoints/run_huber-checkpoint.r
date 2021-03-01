library(hqreg)
library(Metrics)

X_train <- as.matrix(read.csv("./csv/X_train.csv", sep = ",", header=FALSE))
y_train <- read.csv("./csv/y_train.csv", header=FALSE)[,1]
X_val <- as.matrix(read.csv("./csv/X_val.csv", sep = ",", header=FALSE))
y_val <- as.matrix(read.csv("./csv/y_val.csv", sep = ",", header=FALSE))

#print(length(y_train))
#print(typeof(y_train))
#print(dim(X_train))
#print(typeof(X_train))



best_metric = 1000
best_gamma = 0.1
best_lambda = 0.1
best_ypred = list()

# for (gamma in seq(from=0.01,to=1,by=1/10)) {
#     fit <- hqreg(X_train,y_train,gamma=gamma)
#     for (lambda in seq(from=0.1,to=1,by=1/10)) {
#         y_pred <- predict(fit, X_val, lambda = lambda) # specify lambda for now
#         metric = mdae(y_val,y_pred)
#         if (metric < best_metric) {
#             best_metric = metric
#             best_ypred = y_pred
#             best_gamma = gamma
#             best_lambda = lambda
#         }
#     }
# }

fit <- hqreg(X_train,y_train,gamma=0.01)


# fit <- hqreg(X_train,y_train, gamma=0.1)
best_ypred <- predict(fit, X_val, lambda = 0.1) # specify lambda for now

print(best_lambda)
print(best_gamma)

file.create("./csv/y_pred.csv")
write.csv(best_ypred,"./csv/y_pred.csv",row.names=FALSE)

quit(save="no")
