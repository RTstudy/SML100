library(glmnet)

data <- read.table('C:/Users/YusukeSato/Documents/RTstudy/SML100/section5/crime.txt')
X <- data[,seq(3,7)]
y <- data[,1]
X_scaled <- scale(X)

Lmodel <- glmnet(x = X_scaled, y=y, family = 'gaussian', alpha = 1)
plot(Lmodel)

lambda_seq <- sapply(seq(100000),function(x){
  Lcv <- cv.glmnet(x = X_scaled, y=y, family = 'gaussian', alpha = 1)
  #plot(Lcv)
  return(Lcv$lambda.min)
})
hist(lambda_seq,breaks = 50)



estY <- predict(Lmodel, newx = X_scaled,
                s = Lcv$lambda.min,
                type = 'response')
mse <-sum((y-estY)^2) / length(y)
plot(y, estY)


Lmodel$beta
Lmodel$lambda
