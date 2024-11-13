library(Rcpp)
library(RcppProgress)
library(mvtnorm)
library(ggplot2)
library(coda)
library(mcclust)
library(sn)
library(readxl)

#DATA GENERATION

#SIM1----
#Data Generation: (x1,x2)~N, x3~Ber, x4~Ber ; Y ~ N(beta'x, 0.5).
#Model Assumption: Gaussian Mixture components with unknown means and variances.
Rcpp::sourceCpp('GibbsSampler_SplitandMerge.cpp')

set.seed(123)
group_1 <- cbind(rmvnorm(n=3775, mean=c(-3,3),sigma= diag(.5, 2)), 
                 sample(c(0,1), size = 3775, prob = c(0.9, 0.1), replace = T),
                 sample(c(0,1), size = 3775, prob = c(0.9, 0.1), replace = T))
group_1 <- cbind(apply(group_1, 1, function(x) rnorm(1, 1 + x %*% c(5, 2, 1, 0), .5)), group_1)

group_2<- cbind(rmvnorm(3775, c(0,0), diag(.5, 2)), 
                sample(c(0,1), size = 3775, prob = c(0.5, 0.5), replace = T),
                sample(c(0,1), size = 3775, prob = c(0.5, 0.5), replace = T))
group_2 <- cbind(apply(group_2, 1, function(x) rnorm(1, 4 + x %*% c(2, -2, 1, -1), .5)), group_2)

group_3 <- cbind(rmvnorm(2510, c(3,3), diag(.5, 2)), 
                 sample(c(0,1), size = 2510, prob = c(0.1, 0.9), replace = T),
                 sample(c(0,1), size = 2510, prob = c(0.1, 0.9), replace = T))
group_3 <- cbind(apply(group_3, 1, function(x) rnorm(1,  -1 + x %*% c(-5, -2, -1, 1), .5)), group_3)
data <- rbind(group_1, group_2, group_3)
new_column_names <- c("Y", "X1", "x2", "x3", "x4")
colnames(data) <- new_column_names
#scaling
data[,c(2,3)] <- scale(data[,c(2,3)])
clusters <- factor(rep(1:3, c(3775, 3775, 2510)))
#test set
sampled_test <- sample(nrow(data), 60)
data_test <- data[sampled_test,]
Y_test <- data_test[,1]
X_test <- data_test[,c(2,3,4,5)]
clusters_test <- clusters[sampled_test]
#training set
data_train <- data[-sampled_test,]
clusters_train <- clusters[-sampled_test]
Y_train <- data_train[,1]
X_train <- data_train[,c(2,3,4,5)]
sampled_train <- sample(nrow(data_train), 100)
data_sample <- data_train[sampled_train, ]
Y_sample <- data_sample[,1]
X_sample <- data_sample[,c(2,3,4,5)]
clusters_sample <- clusters_train[sampled_train]

histo_by_group <- ggplot(data.frame(Y_train, clusters_train), aes(x = Y_train, fill = clusters_train)) +
  geom_histogram(binwidth = 0.7, position = "identity", alpha = 0.7,color = "black")+
  labs(title = "Plot", x = "Y", y = "frequence") +
  scale_fill_manual(values = c("1" = "#E0FFFF","2"= "darkblue","3"= "darkgrey"), name = "groups")
print(histo_by_group)

col <-c("1" = "lightblue",  
        "2" = "darkblue",  
        "3" = "darkgray",  
        "4" = "lightgray",   
        "5" = "#E0FFFF",
        "6" = "#4682B4",
        "7" = "#87CEFA",
        "8" = "#87CEEB")

scatter_plot <- ggplot(as.data.frame(data_train), aes(x = X1, y = x2, color = clusters_train)) +
  geom_point(alpha = 0.7) +
  labs(title = "plot X1-Y", x = "X1", y = "Y") +
  scale_color_manual(values = c("1" = "#E0FFFF","2"= "darkblue","3"= "darkgrey"), name="groups")
print(scatter_plot)

##
#X1-Y
scatter_plot <- ggplot(as.data.frame(data_train), aes(x = X1, y = Y, color = clusters_train)) +
  geom_point(alpha = 0.7) +
  labs(title = "plot X1-Y", x = "X1", y = "Y") +
  scale_color_manual(values = c("1" = "#E0FFFF","2"= "darkblue","3"= "darkgrey"), name="groups")
print(scatter_plot)

#x2-Y1
scatter_plot <- ggplot(as.data.frame(data_train), aes(x = x2, y = Y, color = clusters_train)) +
  geom_point(alpha = 0.7) +
  labs(title = "plot X2-Y", x = "X2", y = "Y") +
  scale_color_manual(values = c("1" = "#E0FFFF","2"= "darkblue","3"= "darkgrey"), name="groups")
print(scatter_plot)

#x3-Y1
scatter_plot <- ggplot(as.data.frame(data_train), aes(x = x3, y = Y, color = clusters_train)) +
  geom_point(alpha = 0.6) +
  labs(title = "plot X3-Y", x = "X3", y = "Y") +
  scale_color_manual(values = c("1" = "#E0FFFF","2"= "darkblue","3"= "darkgrey"), name="groups")
print(scatter_plot)

#x4-Y1
scatter_plot <- ggplot(as.data.frame(data_train), aes(x = x4, y = Y, color = clusters_train)) +
  geom_point(alpha = 0.6) +
  labs(title = "plot X4-Y", x = "X4", y = "Y") +
  scale_color_manual(values = c("1" = "#E0FFFF","2"= "darkblue","3"= "darkgrey"), name="groups")
print(scatter_plot)



#G1/G2/G3
invcov_cont <- solve(cov(data_train[,c(2,3)]))
#PPMX
#continuous covariates
#sigma2 known (prior on mu) (ppmx_n)
c1 <- 1
c2 <- 2
#m0 <- colMeans(data_rob[,c(2,3)])
#S0 <- apply(data_rob[,c(2,3)], 2, var)
#sigma2 unknown (prior on mu and sigma2) (ppmx_t)
a0 <- 1
k0 <- 6
#m <- colMeans(data_rob[,c(2,3)])
#b0 <- apply(data_rob[,c(2,3)], 2, var)
#binary covariates
a <- 0.1
b <- 0.1
#categorical covariates
categs_1 <- NULL
alpha_1 <- rep(0.1, length(categs_1))


ngg_params <- list(sigma = 0.2, k = 0.3)
P0_params <- list(mu0 = mean(Y_train), lambda0 = 0.01, a0 = 1, b0 = 10)
niter <- 3000
nburn <- 2000
nGibbs <- 0
thinGibbs <- 1
thin <- 1
var_type <- c(1,1,0,0)

g_params <- list(fun = "g3", lambda = 0.1, alpha = 1, cov = invcov_cont)
g_params <- list("ppmx_n", c1, c2, a, b, categs_1, alpha_1)
g_params <- list("ppmx_t", a0, k0, a, b, categs_1, alpha_1)

system.time(result_0 <- run_mcmc(data=data_sample, type=var_type, g_params=g_params, 
                                 ngg_params=ngg_params, P0_params=P0_params, niter=niter, 
                                 nburn=nburn, thin=thin, nGibbs = nGibbs, thin_Gibbs = thinGibbs))

freq_chain<-t(result_0$freq_chain)[-1,]
matplot(freq_chain, type = "l", xlab = "Iteration", main = "Split & Merge (3000, 0, 1)",ylim = c(0, 1))

labels <- as.factor(result_0$best_clus_binder)
table(labels)
histo_by_group <- ggplot(as.data.frame(data_sample), aes(x = Y, fill = labels)) +
  geom_histogram(binwidth = 1, position = "identity", alpha = 0.7, color = "black") +
  labs(title = "Plot", x = "Y", y = "frequency") +
  scale_fill_discrete(name = "groups")
print(histo_by_group)

arandi(clusters_sample, result_0$best_clus_binder)



#PREDICTION
grid_y1 <- seq(min(Y_train), max(Y_train), by = 0.3)
grid_y2 <- c(0)
niter <- 2000
nburn <- 1000
thin <- 1
nGibbs <- 0
thinGibbs <- 1

g_params <- list(fun = "g0", lambda = 0.1, alpha = 1, cov = invcov_cont)
g_params <- list("ppmx_n", c1, c2, a, b, categs_1, alpha_1)
g_params <- list("ppmx_t", a0, k0, a, b, categs_1, alpha_1)

result_pred <- run_mcmc_pred(as.matrix(data_sample), var_type, g_params, ngg_params,
                             P0_params, Y_test, as.matrix(X_test[-c(1:26),]), grid_y1, niter, nburn, thin, nGibbs, thinGibbs,
                             TRUE, FALSE, FALSE, TRUE)

ggplot(data.frame(x = grid_y1, y = c(colMeans(result_pred$pred_grid)), 
                  ylow = apply(result_pred$pred_grid, 2, quantile, p = 0.05), 
                  yup = apply(result_pred$pred_grid, 2, quantile, p = 0.95))) +
  theme_bw() +
  geom_line(aes(x = x, y = y)) + 
  xlab("y") +
  ylab("density") +
  ggtitle("Predictive distribution with G0 ") + 
  geom_vline(aes(xintercept = Y_test[27]), lty = 2, col = 2) +
  geom_ribbon(aes(x = x, ymin = ylow, ymax = yup), alpha = 0.25) +
  ylim(c(0, 0.3))

#METRICS

niter <- 3000
nburn <- 2000
thin <- 1
nGibbs <- 0
thinGibbs <- 1
itermean <- 50

g_params <- list(fun = "g0", lambda = 1, alpha = 1, cov = invcov_cont)
metrics_0 <- compute_metrics(as.matrix(data_train), clusters_train, var_type, grid_y1,grid_y2,
                             g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs,
                             FALSE,FALSE,TRUE,TRUE)
g_params <- list(fun = "g1", lambda = 1, alpha = 1, cov = invcov_cont)
metrics_1 <- compute_metrics(as.matrix(data_train), clusters_train, var_type, grid_y1,grid_y2,
                             g_params, ngg_params, P0_params, niter, nburn, thin,itermean, nGibbs, thinGibbs,
                             FALSE,FALSE,TRUE,TRUE)
g_params <- list(fun = "g2", lambda = 1, alpha = 1, cov = invcov_cont)
metrics_2 <- compute_metrics(as.matrix(data_train), clusters_train, var_type, grid_y1,grid_y2,
                             g_params, ngg_params, P0_params, niter, nburn, thin,itermean, nGibbs, thinGibbs,
                             FALSE,FALSE,TRUE,TRUE)
g_params <- list(fun = "g3", lambda = 0.2, alpha = 1, cov = invcov_cont)
metrics_3 <- compute_metrics(as.matrix(data_train), clusters_train, var_type, grid_y1,grid_y2,
                             g_params, ngg_params, P0_params, niter, nburn, thin,itermean, nGibbs, thinGibbs,
                             FALSE,FALSE,TRUE,TRUE)
g_params <- list("ppmx_n", c1, c2, a, b, categs_1, alpha_1)
metrics_n <- compute_metrics(as.matrix(data_train), clusters_train, var_type, grid_y1,grid_y2,
                             g_params, ngg_params, P0_params, niter, nburn, thin,itermean, nGibbs, thinGibbs,
                             FALSE,FALSE,TRUE,TRUE)
g_params <- list("ppmx_t", a0, k0, a, b, categs_1, alpha_1)
metrics_t <- compute_metrics(as.matrix(data_train), clusters_train, var_type, grid_y1,grid_y2,
                             g_params, ngg_params, P0_params, niter, nburn, thin,itermean, nGibbs, thinGibbs,
                             FALSE,FALSE,TRUE,TRUE)

df1 <- data.frame(metrics_0)
df2 <- data.frame(metrics_1)
df3 <- data.frame(metrics_2)
df4 <- data.frame(metrics_3)
df5 <- data.frame(metrics_n)
df6 <- data.frame(metrics_t)
comb_Gibbs <- rbind(df1, df2, df3, df4, df5, df6)  
rownames(comb_Gibbs) <- c("g0", "g1", "g2", "g3", "ppmx_n", "ppmx_t")


niter <- 3000
nburn <- 2000
thinGibbs <- 1
nGibbs <- 0
thin <- 1
itermean <- 5
g_params <- list(fun = "g0", lambda = 1, alpha = 1, cov = invcov_cont)
time_g0 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g1", lambda = 1, alpha = 1, cov = invcov_cont)
time_g1 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g2", lambda = 1, alpha = 1, cov = invcov_cont)
time_g2 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g3", lambda = 0.2, alpha = 1, cov = invcov_cont)
time_g3 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list("ppmx_n", c1, c2, a, b, categs_1, alpha_1)
time_ppmx_n <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list("ppmx_t", a0, k0, a, b, categs_1, alpha_1)
time_ppmx_t <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
time_Gibbs <- data.frame(time = c(time_g0, time_g1, time_g2, time_g3,
                                  time_ppmx_n, time_ppmx_t))
rownames(time_Gibbs) <- c("g0", "g1", "g2", "g3", "ppmx_n", "ppmx_t")


#SIM2----
#Data Generation: x1~N(0,1), x2~Ber(0.8), x3~Ber(0.2), x4~Discr(1/3,1/3,1/3) ; 
#Y ~ N(b0j*x4 + b1j*x1, 0.5).
#Model Assumption: Gaussian Mixture components with unknown means and variances.
Rcpp::sourceCpp('split_with_g_ngg.cpp')

set.seed(123)
tot <- 10060
X1 <- rnorm(tot, mean = 1, sd = 1)
X2 <- runif(tot, min = 0, max = 10)
X3 <- rbinom(tot, size = 1, prob = 0.5)
X4 <- sample(1:3, tot, replace = TRUE)


# Form clusters based on interaction of x3 and x4
clusters <- interaction(X3, X4)
levels(clusters) <- c("1", "2", "3", "4", "5", "6")
table(clusters)
# Define unique slopes and intercepts for each cluster
unique_slopes <- c(-3, -1, 1, 1.7, 3, 4)
unique_intercepts <- c(-2, -1.5, -1, 0, 1, 2)

# Generate response y
Y <- numeric(tot)
for (i in 1:length(levels(clusters))) {
  idx <- which(clusters == levels(clusters)[i])
  Y[idx] <- unique_intercepts[i] + unique_slopes[i] * X1[idx] + rnorm(length(idx),0,.1)
}


data <- data.frame(Y, X1, X2, X3, X4)
#scaling
data[,c(2,3)] <- scale(data[,c(2,3)])
#test set
sampled_test <- sample(nrow(data), 60)
data_test <- data[sampled_test,]
Y_test <- data_test[,1]
X_test <- data_test[,c(2,3,4,5)]
clusters_test <- clusters[sampled_test]
#training set
data_train <- data[-sampled_test,]
clusters_train <- clusters[-sampled_test]
Y_train <- data_train[,1]
X_train <- data_train[,c(2,3,4,5)]
sampled_train <- sample(nrow(data_train), 100)
data_sample <- data_train[sampled_train, ]
Y_sample <- data_sample[,1]
X_sample <- data_sample[,c(2,3,4,5)]
clusters_sample <- clusters_train[sampled_train]


histo_by_group <- ggplot(data.frame(Y_train, clusters_train), aes(x = Y_train, fill = clusters_train)) +
  geom_histogram(binwidth = 0.4, position = "identity", alpha = 0.8,color = "black")+
  labs(title = "Plot", x = "Y", y = "frequence") +
  scale_fill_manual(values = c("1" = "darkgreen","5"= "darkblue","3"= "blue","4"="lightgreen", "2"="#E0FFFF","6"="lightyellow"), name = "groups")
print(histo_by_group)
table(clusters_train)
#X1-Y
scatter_plot <- ggplot(as.data.frame(data_train), aes(x = X1, y = Y, color = clusters_train)) +
  geom_point(alpha = 0.6) +
  labs(title = "plot X1-Y", x = "X1", y = "Y") +
  scale_color_manual(values = c("1" = "darkgreen","5"= "darkblue","3"= "blue","4"="lightgreen", "2"="#E0FFFF","6"="lightyellow"), name="groups")
print(scatter_plot)

#x2-Y1
scatter_plot <- ggplot(as.data.frame(data_train), aes(x = X2, y = Y, color = clusters_train)) +
  geom_point(alpha = 0.6) +
  labs(title = "plot X2-Y", x = "X2", y = "Y") +
  scale_color_manual(values = c("1" = "darkgreen","5"= "darkblue","3"= "blue","4"="lightgreen", "2"="#E0FFFF","6"="lightyellow"), name="groups")
print(scatter_plot)

#x3-Y1
scatter_plot <- ggplot(as.data.frame(data_train), aes(x = X3, y = Y, color = clusters_train)) +
  geom_point(alpha = 0.6) +
  labs(title = "plot X3-Y", x = "X3", y = "Y") +
  scale_color_manual(values = c("1" = "darkgreen","5"= "darkblue","3"= "blue","4"="lightgreen", "2"="#E0FFFF","6"="lightyellow"), name="groups")
print(scatter_plot)

#x4-Y1
scatter_plot <- ggplot(as.data.frame(data_train), aes(x = X4, y = Y, color = clusters_train)) +
  geom_point(alpha = 0.6) +
  labs(title = "plot X4-Y", x = "X4", y = "Y") +
  scale_color_manual(values = c("1" = "darkgreen","5"= "darkblue","3"= "blue","4"="lightgreen", "2"="#E0FFFF","6"="lightyellow"), name="groups")
print(scatter_plot)



#G1/G2/G3
invcov_cont <- solve(cov(data_train[,c(2,3)]))
#invcov_cont <- matrix(1/var(data_train[,2]))
espilon_star <- compute_lambda(as.matrix(data_train[,c(2,3,4,5)]), nrep = 10000,invcov_cont, var_type)
lambda_true <- .001 / mean(espilon_star[,1])

#PPMX
#continuous covariates
#sigma2 known (prior on mu) (ppmx_n)
c1 <- 2
c2 <- 1
#m0 <- mean(data_train[,2])
#S0 <- apply(data_train[,c(2,3)], 2, var)
#S0 <- var(data_train[,2])
#sigma2 unknown (prior on mu and sigma2) (ppmx_t)
a0 <- 2
k0 <- 1
#m <- colMeans(data_train[,c(2,3)])
#m <- mean(data_train[,2])
#b0 <- apply(data_train[,c(2,3)], 2, var)
#b0 <- var(data_train[,2])
#binary covariates
a <- 0.1
b <- 0.1
#categorical covariates
categs_1 <- unique(data[,5])
alpha_1 <- rep(0.1, length(categs_1))


ngg_params <- list(sigma = 0.2, k = 0.3)
P0_params <- list(mu0 = mean(Y_train), lambda0 = 0.01, a0 = 1, b0 = 4)
niter <- 5000
nburn <- 1000
thin <- 1
nGibbs <- 0
thinGibbs <- 10
var_type <- c(1,1,0,-1)

g_params <- list(fun = "g1", lambda = 1, alpha = 1, cov = invcov_cont)
g_params <- list("ppmx_n", c1, c2, a, b, categs_1, alpha_1)
g_params <- list("ppmx_t", a0, k0, a, b, categs_1, alpha_1)


system.time(result_0 <- run_mcmc(data=as.matrix(data_sample), type=var_type, g_params=g_params, 
                                 ngg_params=ngg_params, P0_params=P0_params, niter=niter, 
                                 nburn=nburn, thin=thin, nGibbs = nGibbs, thin_Gibbs = thinGibbs))

freq_chain<-t(result_0$freq_chain)[-1,]
matplot(freq_chain, type = "l", xlab = "", ylab = "", main = "Split-Merge(0, 10) - NN", ylim = c(0, 1), col = "black", axes = FALSE)
axis(1)
axis(2, at = seq(0.0, 1.0, by = 0.2), labels = format(seq(0.0, 1.0, by = 0.2), nsmall = 1), las = 1)
box()


labels <- as.factor(result_0$best_clus_binder)
table(labels)
histo_by_group <- ggplot(as.data.frame(data_sample), aes(x = Y, fill = labels)) +
  geom_histogram(binwidth = 1, position = "identity", alpha = 0.7, color = "black") +
  labs(title = "Plot", x = "Y", y = "frequency") +
  scale_fill_discrete(name = "groups")
print(histo_by_group)

arandi(clusters_sample, result_0$best_clus_binder)

#PREDICTION
grid_y1 <- seq(min(Y_train), max(Y_train), by = 0.1)
grid_y2 <- c(0)
niter <- 2000
nburn <- 1000
thin <- 1
nGibbs <- 0
thinGibbs <- 1

g_params <- list(fun = "g1", lambda = 1, alpha = 1, cov = invcov_cont)
g_params <- list("ppmx_n", c1, c2, a, b, categs_1, alpha_1)
g_params <- list("ppmx_t", a0, k0, a, b, categs_1, alpha_1)

result_pred <- run_mcmc_pred(as.matrix(data_sample), var_type, g_params, ngg_params,
                             P0_params, Y_test, as.matrix(X_test[-c(1:26),]), grid_y1, niter, nburn, thin, nGibbs, thinGibbs,
                             TRUE, FALSE, FALSE, TRUE)

ggplot(data.frame(x = grid_y1, y = c(colMeans(result_pred$pred_grid)), 
                  ylow = apply(result_pred$pred_grid, 2, quantile, p = 0.05), 
                  yup = apply(result_pred$pred_grid, 2, quantile, p = 0.95))) +
  theme_bw() +
  geom_line(aes(x = x, y = y)) + 
  xlab("y") +
  ylab("density") +
  ggtitle("Predictive distribution with N-N ") + 
  geom_vline(aes(xintercept = Y_test[27]), lty = 2, col = 2) +
  geom_ribbon(aes(x = x, ymin = ylow, ymax = yup), alpha = 0.25) +
  ylim(c(0, 0.3))

#METRICS
niter <- 3000
nburn <- 2000
thin <- 1
nGibbs <- 0
thinGibbs <- 1
itermean <- 30
g_params <- list(fun = "g0", lambda = 1, alpha = 1, cov = invcov_cont)
metrics_0 <- compute_metrics(as.matrix(data_train), clusters_train, var_type, grid_y1, grid_y2,
                             g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs,
                             FALSE,FALSE,TRUE,TRUE)
g_params <- list(fun = "g1", lambda = 1, alpha = 1, cov = invcov_cont)
metrics_1 <- compute_metrics(as.matrix(data_train), clusters_train, var_type, grid_y1, grid_y2,
                             g_params, ngg_params, P0_params, niter, nburn, thin,itermean, nGibbs, thinGibbs,
                             FALSE,FALSE,TRUE,TRUE)
g_params <- list(fun = "g2", lambda = 1, alpha = 1, cov = invcov_cont)
metrics_2 <- compute_metrics(as.matrix(data_train), clusters_train, var_type, grid_y1, grid_y2,
                             g_params, ngg_params, P0_params, niter, nburn, thin,itermean, nGibbs, thinGibbs,
                             FALSE,FALSE,TRUE,TRUE)
g_params <- list(fun = "g3", lambda = 0.1, alpha = 1, cov = invcov_cont)
metrics_3 <- compute_metrics(as.matrix(data_train), clusters_train, var_type, grid_y1, grid_y2,
                             g_params, ngg_params, P0_params, niter, nburn, thin,itermean, nGibbs, thinGibbs,
                             FALSE,FALSE,TRUE,TRUE)
g_params <- list("ppmx_n", c1, c2, a, b, categs_1, alpha_1)
metrics_n <- compute_metrics(as.matrix(data_train), clusters_train, var_type, grid_y1, grid_y2,
                             g_params, ngg_params, P0_params, niter, nburn, thin,itermean, nGibbs, thinGibbs,
                             FALSE,FALSE,TRUE,TRUE)
g_params <- list("ppmx_t", a0, k0, a, b, categs_1, alpha_1)
metrics_t <- compute_metrics(as.matrix(data_train), clusters_train, var_type, grid_y1, grid_y2,
                             g_params, ngg_params, P0_params, niter, nburn, thin,itermean, nGibbs, thinGibbs,
                             FALSE,FALSE,TRUE,TRUE)

df1 <- data.frame(metrics_0)
df2 <- data.frame(metrics_1)
df3 <- data.frame(metrics_2)
df4 <- data.frame(metrics_3)
df5 <- data.frame(metrics_n)
df6 <- data.frame(metrics_t)
comb_Gibbs <- rbind(df1, df2, df3, df4, df5, df6)  
rownames(comb_Gibbs) <- c("g0", "g1", "g2", "g3", "ppmx_n", "ppmx_t")

#SIM3----
# Data Generation: x1~Ber(0.5), x2~Discr(1/3,1/3,1/3), x3= 2*x1 + e1, e1~Unif(0,1), 
#                                                      x4= -2*x2 + e2, e2~Unif(0,1);
# 4 dependent covariates
# 4 clusters with the same n°obs not directly covariate dependent.
#Model Assumption: Gaussian Mixture components with unknown means and variances.
Rcpp::sourceCpp('split_with_g_ngg.cpp')

set.seed(123)
tot <- 10060
X1 <- rbinom(tot, size = 1, prob = 0.5)
X2 <- sample(1:3, tot, replace = TRUE)
X3 <- 2*X1 + runif(tot,0,1)
X4 <- -2*X2 + runif(tot,0,1)

num_clusters <- 4
means <- c(-5.0, -2, 2, 5.0)
std_dev <- 0.70

# Generate clusters
Y = c(rnorm(tot/num_clusters, mean = means[1], sd = std_dev),
        rnorm(tot/num_clusters, mean = means[2], sd = std_dev),
        rnorm(tot/num_clusters, mean = means[3], sd = std_dev),
        rnorm(tot/num_clusters, mean = means[4], sd = std_dev))

clusters = factor(rep(1:num_clusters, each = tot/num_clusters))


data <- data.frame(Y, X1, X2, X3, X4)
#scaling
data[,c(4,5)] <- scale(data[,c(4,5)])
#test set
sampled_test <- sample(nrow(data), 60)
data_test <- data[sampled_test,]
Y_test <- data_test[,1]
X_test <- data_test[,c(2,3,4,5)]
clusters_test <- clusters[sampled_test]
#training set
data_train <- data[-sampled_test,]
clusters_train <- clusters[-sampled_test]
Y_train <- data_train[,1]
X_train <- data_train[,c(2,3,4,5)]
sampled_train <- sample(nrow(data_train), 100)
data_sample <- data_train[sampled_train, ]
Y_sample <- data_sample[,1]
X_sample <- data_sample[,c(2,3,4,5)]
clusters_sample <- clusters_train[sampled_train]


histo_by_group <- ggplot(data.frame(Y_train, clusters_train), aes(x = Y_train, fill = clusters_train)) +
  geom_histogram(binwidth = 0.2, position = "identity", alpha = 0.7,color = "black")+
  labs(title = "Plot", x = "Y", y = "frequence") +
  scale_fill_manual(values = c("1" = "darkgray","2"= "darkblue","3"= "blue","4"="#87CEFA"), name = "groups")
print(histo_by_group)

#X1-Y
scatter_plot <- ggplot(as.data.frame(data_train), aes(x = X1, y = Y, color = clusters_train)) +
  geom_point(alpha = 0.6) +
  labs(title = "plot X1-Y", x = "X1", y = "Y") +
  scale_color_manual(values = c("1" = "darkgray","2"= "darkblue","3"= "blue","4"="#87CEFA"), name="groups")
print(scatter_plot)

#x2-Y1
scatter_plot <- ggplot(as.data.frame(data_train), aes(x = X2, y = Y, color = clusters_train)) +
  geom_point(alpha = 0.6) +
  labs(title = "plot X2-Y", x = "X2", y = "Y") +
  scale_color_manual(values = c("1" = "darkgray","2"= "darkblue","3"= "blue","4"="#87CEFA"), name="groups")
print(scatter_plot)

#x3-Y1
scatter_plot <- ggplot(as.data.frame(data_train), aes(x = X3, y = Y, color = clusters_train)) +
  geom_point(alpha = 0.6) +
  labs(title = "plot X3-Y", x = "X3", y = "Y") +
  scale_color_manual(values = c("1" = "darkgray","2"= "darkblue","3"= "blue","4"="#87CEFA"), name="groups")
print(scatter_plot)

#x4-Y1
scatter_plot <- ggplot(as.data.frame(data_train), aes(x = X4, y = Y, color = clusters_train)) +
  geom_point(alpha = 0.6) +
  labs(title = "plot X4-Y", x = "X4", y = "Y") +
  scale_color_manual(values = c("1" = "darkgray","2"= "darkblue","3"= "blue","4"="#87CEFA"), name="groups")
print(scatter_plot)



#G1/G2/G3
invcov_cont <- solve(cov(data_train[,c(4,5)]))
#invcov_cont <- matrix(1/var(data_train[,2]))
espilon_star <- compute_lambda(as.matrix(data_train[,c(2,3,4,5)]), nrep = 10000,invcov_cont, var_type)
lambda_true <- .08 / mean(espilon_star[,1])

#PPMX
#continuous covariates
#sigma2 known (prior on mu) (ppmx_n)
c1 <- 2
c2 <- 1
#m0 <- mean(data_rob[,2])
#S0 <- apply(data_rob[,c(2,3)], 2, var)
#S0 <- var(data_rob[,2])
#sigma2 unknown (prior on mu and sigma2) (ppmx_t)
a0 <- 1
k0 <- 4
#m <- colMeans(data_rob[,c(2,3)])
#m <- mean(data_rob[,2])
#b0 <- apply(data_rob[,c(2,3)], 2, var)
#b0 <- var(data_rob[,2])
#binary covariates
a <- 0.1
b <- 0.1
#categorical covariates
categs_1 <- unique(data[,3])
alpha_1 <- rep(0.1, length(categs_1))

g_params <- list(fun = "g1", lambda = 0.05, alpha = 1, cov = invcov_cont)
g_params <- list("ppmx_n", c1, c2, a, b, categs_1, alpha_1)
g_params <- list("ppmx_t", a0, k0, a, b, categs_1, alpha_1)

ngg_params <- list(sigma = 0.2, k = 0.3)
P0_params <- list(mu0 = mean(Y_train), lambda0 = 0.01, a0 = 2, b0 = 1)
niter <- 5000
nburn <- 1000
thin <- 1
nGibbs <- 0
thinGibbs <- 10
var_type <- c(0,-1,1,1)

system.time(result_0 <- run_mcmc(data=as.matrix(data_sample), type=var_type, g_params=g_params, 
                                 ngg_params=ngg_params, P0_params=P0_params, niter=niter, 
                                 nburn=nburn, thin=thin, nGibbs = nGibbs, thin_Gibbs = thinGibbs))

freq_chain<-t(result_0$freq_chain)[-1,]
matplot(freq_chain, type = "l", xlab = "", ylab = "", main = "Split-Merge(0, 10) - NN", ylim = c(0, 1), col = "black", axes = FALSE)
axis(1)
axis(2, at = seq(0.0, 1.0, by = 0.2), labels = format(seq(0.0, 1.0, by = 0.2), nsmall = 1), las = 1)
box()

labels <- as.factor(result_0$best_clus_binder)
table(labels)
histo_by_group <- ggplot(as.data.frame(data_sample), aes(x = Y, fill = labels)) +
  geom_histogram(binwidth = 1, position = "identity", alpha = 0.7, color = "black") +
  labs(title = "Plot", x = "Y", y = "frequency") +
  scale_fill_discrete(name = "groups")
print(histo_by_group)

arandi(clusters_sample, result_0$best_clus_binder)

##PREDICTION
grid_y1 <- seq(min(Y_train), max(Y_train), by = 0.1)
grid_y2 <- c(0)
niter <- 2000
nburn <- 1000
thin <- 1
nGibbs <- 0
thinGibbs <- 1

g_params <- list(fun = "g1", lambda = 0.05, alpha = 1, cov = invcov_cont)
g_params <- list("ppmx_n", c1, c2, a, b, categs_1, alpha_1)
g_params <- list("ppmx_t", a0, k0, a, b, categs_1, alpha_1)

result_pred <- run_mcmc_pred(as.matrix(data_sample), var_type, g_params, ngg_params,
                             P0_params, Y_test, as.matrix(X_test[-c(1:44),]), grid_y1, niter, nburn, thin, nGibbs, thinGibbs,
                             TRUE, FALSE, FALSE, TRUE)

ggplot(data.frame(x = grid_y1, y = c(colMeans(result_pred$pred_grid)), 
                  ylow = apply(result_pred$pred_grid, 2, quantile, p = 0.05), 
                  yup = apply(result_pred$pred_grid, 2, quantile, p = 0.95))) +
  theme_bw() +
  geom_line(aes(x = x, y = y)) + 
  xlab("y") +
  ylab("density") +
  ggtitle("Predictive distribution with G1 ") + 
  geom_vline(aes(xintercept = Y_test[45]), lty = 2, col = 2) +
  geom_ribbon(aes(x = x, ymin = ylow, ymax = yup), alpha = 0.25) +
  ylim(c(0, 0.6))


#SIM4
##study the impact that an increasingly larger p has on prediction and model fit,
##with p ∈ {10,20,50,100,150,200}.
#P = 2----
genera_dataset <- function(n, p, num_clusters) {
  # Initialize the matrix X
  X <- matrix(NA, n, p)
  
  # Generate covariates from specific distributions in a cyclic manner
  for (j in 1:p) {
    if (j %% 4 == 1) {
      # First covariate: Normal
      mean <- runif(1, -5, 5)  # Random mean
      sd <- runif(1, 0.5, 2)    # Random standard deviation
      X[, j] <- rnorm(n, mean = mean, sd = sd)
    } else if (j %% 4 == 2) {
      # Second covariate: Uniform
      min <- runif(1, -3, 0)  # Random lower limit
      max <- runif(1, 0, 3)    # Random upper limit
      X[, j] <- runif(n, min = min, max = max)
    } else if (j %% 4 == 3) {
      # Third covariate: t-Student
      df <- sample(3:10, 1)  # Random degrees of freedom
      offset <- runif(1, -5, 5)  # Random offset
      X[, j] <- rt(n, df = df) + offset
    } else if (j %% 4 == 0) {
      # Fourth covariate: Skew-Normal
      xi <- runif(1, 5, 15)  # Random parameter xi
      omega <- runif(1, 0.5, 2)  # Random parameter omega
      alpha <- runif(1, 5, 10)  # Random parameter alpha
      X[, j] <- rsn(n, xi = xi, omega = omega, alpha = alpha)
    }
  }
  
  return(X)
}
modifica_covariate <- function(X, num_clusters, num_cov_mod_per_cluster, contraction_factor = NULL, min_shift_diff = 5) {
  n <- nrow(X)
  p <- ncol(X)
  
  # Indici per i cluster
  cluster_indices <- list(
    cluster1 = 1:(n/num_clusters),
    cluster2 = (n/num_clusters + 1):(2 * n/num_clusters),
    cluster3 = (2 * n/num_clusters + 1):(3 * n/num_clusters),
    cluster4 = (3 * n/num_clusters + 1):n
  )
  
  # Inizializza lista per tenere traccia di quali shift sono stati usati per ogni covariata
  covariate_shifts <- list()
  
  # Modifica delle covariate per ciascun cluster
  for (cluster_id in 1:num_clusters) {
    cluster <- cluster_indices[[cluster_id]]  # Ottieni gli indici del cluster attuale
    
    # Seleziona casualmente le covariate da modificare per questo cluster
    covariates_to_modify <- sample(1:p, size = num_cov_mod_per_cluster, replace = FALSE)
    
    for (cov in covariates_to_modify) {
      # Se non è ancora stata modificata questa covariata in nessun cluster, inizializzala
      if (!is.null(covariate_shifts[[as.character(cov)]])) {
        # Genera uno shift che sia distante dagli altri shift già usati per questa covariata
        repeat {
          mean_shift <- sample(c(runif(1, -15, -3), runif(1, 3, 15)), 1)
          if (all(abs(mean_shift - covariate_shifts[[as.character(cov)]]) >= min_shift_diff)) {
            break
          }
        }
      } else {
        # Primo shift per questa covariata
        mean_shift <- sample(c(runif(1, -14, -5), runif(1, 5, 14)), 1)
        covariate_shifts[[as.character(cov)]] <- c()  # Inizializza la lista
      }
      
      # Salva lo shift utilizzato per questa covariata
      covariate_shifts[[as.character(cov)]] <- c(covariate_shifts[[as.character(cov)]], mean_shift)
      
      # Applica lo shift specifico per il cluster e covariata, con o senza contraction
      if (!is.null(contraction_factor)) {
        X[cluster, cov] <- (X[cluster, cov] + mean_shift) * contraction_factor
      } else {
        X[cluster, cov] <- X[cluster, cov] + mean_shift
      }
    }
  }
  
  return(X)
}

Rcpp::sourceCpp('split_with_g_ngg.cpp')

set.seed(123)
n <- 10060
num_clusters <- 4
cluster_means <- c(-3.5, -1.5, 1.5, 3.5)
Y <- c(rnorm(n/num_clusters, mean = cluster_means[1], sd = 0.8),
       rnorm(n/num_clusters, mean = cluster_means[2], sd = 0.8),
       rnorm(n/num_clusters, mean = cluster_means[3], sd = 0.8),
       rnorm(n/num_clusters, mean = cluster_means[4], sd = 0.8))

p <- 2
X_2 <- genera_dataset(n, p, num_clusters)
X_2_mod <- modifica_covariate(X_2, num_clusters, num_cov_mod_per_cluster = 1)

clusters = factor(rep(1:num_clusters, each = n/num_clusters))

data <- data.frame(Y,X_2_mod)
#scaling
data[,2:3] <- scale(data[,2:3])
#test set
sampled_test <- sample(nrow(data), 60)
data_test <- data[sampled_test,]
Y_test <- data_test[,1]
X_test <- data_test[,2:3]
clusters_test <- clusters[sampled_test]
#training set
data_train <- data[-sampled_test,]
clusters_train <- clusters[-sampled_test]
Y_train <- data_train[,1]
X_train <- data_train[,2:3]
sampled_train <- sample(nrow(data_train), 100)
data_sample <- data_train[sampled_train, ]
Y_sample <- data_sample[,1]
X_sample <- data_sample[,2:3]
clusters_sample <- clusters_train[sampled_train]


histo_by_group <- ggplot(data.frame(Y_train, clusters_train), aes(x = Y_train, fill = clusters_train)) +
  geom_histogram(binwidth = 0.2, position = "identity", alpha = 0.7,color = "black")+
  labs(title = "Plot", x = "Y", y = "frequence") +
  scale_fill_manual(values = c("1" = "darkgray","2"= "darkblue","3"= "blue","4"="#87CEFA"), name = "groups")
print(histo_by_group)
##
#X1-Y
scatter_plot <- ggplot(as.data.frame(data_train), aes(x = X1, y = Y, color = clusters_train)) +
  geom_point(alpha = 0.6) +
  labs(title = "plot X1-Y", x = "X1", y = "Y") +
  scale_color_manual(values = c("1" = "darkgray","2"= "darkblue","3"= "blue","4"="#87CEFA"), name="groups")
print(scatter_plot)

#x2-Y1
scatter_plot <- ggplot(as.data.frame(data_train), aes(x = X2, y = Y, color = clusters_train)) +
  geom_point(alpha = 0.6) +
  labs(title = "plot X2-Y", x = "X2", y = "Y") +
  scale_color_manual(values = c("1" = "darkgray","2"= "darkblue","3"= "blue","4"="#87CEFA"), name="groups")
print(scatter_plot)

#x3-Y1
scatter_plot <- ggplot(as.data.frame(data_train), aes(x = X3, y = Y, color = clusters_train)) +
  geom_point(alpha = 0.6) +
  labs(title = "plot X3-Y", x = "X3", y = "Y") +
  scale_color_manual(values = c("1" = "darkgray","2"= "darkblue","3"= "blue","4"="#87CEFA"), name="groups")
print(scatter_plot)

#x4-Y1
scatter_plot <- ggplot(as.data.frame(data_train), aes(x = X4, y = Y, color = clusters_train)) +
  geom_point(alpha = 0.6) +
  labs(title = "plot X4-Y", x = "X4", y = "Y") +
  scale_color_manual(values = c("1" = "darkgray","2"= "darkblue","3"= "blue","4"="#87CEFA"), name="groups")
print(scatter_plot)

#x5-Y1
scatter_plot <- ggplot(as.data.frame(data_train), aes(x = X5, y = Y, color = clusters_train)) +
  geom_point(alpha = 0.6) +
  labs(title = "plot X5-Y", x = "X5", y = "Y") +
  scale_color_manual(values = c("1" = "darkgray","2"= "darkblue","3"= "blue","4"="#87CEFA"), name="groups")
print(scatter_plot)

#x6-Y1
scatter_plot <- ggplot(as.data.frame(data_train), aes(x = X6, y = Y, color = clusters_train)) +
  geom_point(alpha = 0.6) +
  labs(title = "plot X6-Y", x = "X6", y = "Y") +
  scale_color_manual(values = c("1" = "#87CEFA","2"= "darkblue","3"= "blue","4"= "darkgray"), name="groups")
print(scatter_plot)

#x7-Y1
scatter_plot <- ggplot(as.data.frame(data_train), aes(x = X7, y = Y, color = clusters_train)) +
  geom_point(alpha = 0.6) +
  labs(title = "plot X7-Y", x = "X7", y = "Y") +
  scale_color_manual(values = c("1" = "#87CEFA","2"= "darkblue","3"= "blue","4"= "darkgray"), name="groups")
print(scatter_plot)

#x8-Y1
scatter_plot <- ggplot(as.data.frame(data_train), aes(x = X8, y = Y, color = clusters_train)) +
  geom_point(alpha = 0.6) +
  labs(title = "plot X8-Y", x = "X8", y = "Y") +
  scale_color_manual(values = c("1" = "#87CEFA","2"= "darkblue","3"= "blue","4"= "darkgray"), name="groups")
print(scatter_plot)

#x9-Y1
scatter_plot <- ggplot(as.data.frame(data_train), aes(x = X9, y = Y, color = clusters_train)) +
  geom_point(alpha = 0.6) +
  labs(title = "plot X9-Y", x = "X9", y = "Y") +
  scale_color_manual(values = c("1" = "#87CEFA","2"= "darkblue","3"= "blue","4"= "darkgray"), name="groups")
print(scatter_plot)

#x10-Y1
scatter_plot <- ggplot(as.data.frame(data_train), aes(x = X10, y = Y, color = clusters_train)) +
  geom_point(alpha = 0.6) +
  labs(title = "plot X10-Y", x = "X10", y = "Y") +
  scale_color_manual(values = c("1" = "#87CEFA","2"= "darkblue","3"= "blue","4"= "darkgray"), name="groups")
print(scatter_plot)

#G1/G2/G3
invcov_cont <- solve(cov(data_train[,2:3]))
#invcov_cont <- matrix(1/var(data_train[,2]))
espilon_star <- compute_lambda(as.matrix(data_train[,2:11]), nrep = 5000,invcov_cont, var_type)
lambda_true <- .01 / mean(espilon_star[,1])

#PPMX
#continuous covariates
#sigma2 known (prior on mu) (ppmx_n)
c1 <- 1
c2 <- 2
#m0 <- mean(data_rob[,2])
#S0 <- apply(data_rob[,c(2,3)], 2, var)
#S0 <- var(data_rob[,2])
#sigma2 unknown (prior on mu and sigma2) (ppmx_t)
a0 <- 1
k0 <- 4
#m <- colMeans(data_rob[,c(2,3)])
#m <- mean(data_rob[,2])
#b0 <- apply(data_rob[,c(2,3)], 2, var)
#b0 <- var(data_rob[,2])
#binary covariates
a <- 0.1
b <- 0.1
#categorical covariates
categs_1 <- NULL
alpha_1 <- rep(0.1, length(categs_1))

g_params <- list(fun = "g1", lambda = 1, alpha = 1, cov = invcov_cont)
g_params <- list("ppmx_n", c1, c2, a, b, categs_1, alpha_1)
g_params <- list("ppmx_t", a0, k0, a, b, categs_1, alpha_1)

ngg_params <- list(sigma = 0.2, k = 0.3)
P0_params <- list(mu0 = mean(Y_train), lambda0 = 0.01, a0 = 1, b0 = 2)
niter <- 5000
nburn <- 1000
thin <- 1
nGibbs <- 0
thinGibbs <- 10
var_type <- c(1,1)


system.time(result_0 <- run_mcmc(data=as.matrix(data_sample), type=var_type, g_params=g_params, 
                                 ngg_params=ngg_params, P0_params=P0_params, niter=niter, 
                                 nburn=nburn, thin=thin, nGibbs = nGibbs, thin_Gibbs = thinGibbs))

freq_chain<-t(result_0$freq_chain)[-1,]
matplot(freq_chain, type = "l", xlab = "", ylab = "", main = "Split-Merge(0, 10) - NNIG", ylim = c(0, 1), col = "black", axes = FALSE)
axis(1)
axis(2, at = seq(0.0, 1.0, by = 0.2), labels = format(seq(0.0, 1.0, by = 0.2), nsmall = 1), las = 1)
box()

labels <- as.factor(result_0$best_clus_binder)
table(labels)
histo_by_group <- ggplot(as.data.frame(data_sample), aes(x = Y, fill = labels)) +
  geom_histogram(binwidth = 0.6, position = "identity", alpha = 0.7, color = "black") +
  labs(title = "Clustering with G0", x = "Y", y = "frequency") +
  scale_fill_manual(values = c("1" = "#87CEFA","0"= "darkblue","2"= "blue","3"= "darkgray"), name = "groups")
print(histo_by_group)

arandi(clusters_sample, result_0$best_clus_binder)


#PREDICTION
grid_y1 <- seq(min(Y_train), max(Y_train), by = 0.1)
grid_y2 <- c(0)
niter <- 2000
nburn <- 1000
thin <- 1
nGibbs <- 0
thinGibbs <- 1

g_params <- list(fun = "g1", lambda = 1, alpha = 1, cov = invcov_cont)
g_params <- list("ppmx_n", c1, c2, a, b, categs_1, alpha_1)
g_params <- list("ppmx_t", a0, k0, a, b, categs_1, alpha_1)

result_pred <- run_mcmc_pred(as.matrix(data_sample), var_type, g_params, ngg_params,
                             P0_params, Y_test, as.matrix(X_test[-c(1:5),]), grid_y1, niter, nburn, thin, nGibbs, thinGibbs,
                             TRUE, FALSE, FALSE, TRUE)

ggplot(data.frame(x = grid_y1, y = c(colMeans(result_pred$pred_grid)), 
                  ylow = apply(result_pred$pred_grid, 2, quantile, p = 0.05), 
                  yup = apply(result_pred$pred_grid, 2, quantile, p = 0.95))) +
  theme_bw() +
  geom_line(aes(x = x, y = y)) + 
  xlab("y") +
  ylab("density") +
  ggtitle("Predictive distribution with N-NIG ") + 
  geom_vline(aes(xintercept = Y_test[6]), lty = 2, col = 2) +
  geom_ribbon(aes(x = x, ymin = ylow, ymax = yup), alpha = 0.25) +
  ylim(c(0, 0.6))


#P = 5----
Rcpp::sourceCpp('split_with_g_ngg.cpp')

set.seed(123)
n <- 10060
num_clusters <- 4
cluster_means <- c(-3.5, -1.5, 1.5, 3.5)
Y <- c(rnorm(n/num_clusters, mean = cluster_means[1], sd = 0.8),
       rnorm(n/num_clusters, mean = cluster_means[2], sd = 0.8),
       rnorm(n/num_clusters, mean = cluster_means[3], sd = 0.8),
       rnorm(n/num_clusters, mean = cluster_means[4], sd = 0.8))

p <- 5
X_5 <- genera_dataset(n, p, num_clusters)
X_5_mod <- modifica_covariate(X_5, num_clusters, num_cov_mod_per_cluster = 2)

clusters = factor(rep(1:num_clusters, each = n/num_clusters))

data <- data.frame(Y,X_5_mod)
#scaling
data[,2:6] <- scale(data[,2:6])
#test set
sampled_test <- sample(nrow(data), 60)
data_test <- data[sampled_test,]
Y_test <- data_test[,1]
X_test <- data_test[,2:6]
clusters_test <- clusters[sampled_test]
#training set
data_train <- data[-sampled_test,]
clusters_train <- clusters[-sampled_test]
Y_train <- data_train[,1]
X_train <- data_train[,2:6]
sampled_train <- sample(nrow(data_train), 100)
data_sample <- data_train[sampled_train, ]
Y_sample <- data_sample[,1]
X_sample <- data_sample[,2:6]
clusters_sample <- clusters_train[sampled_train]


histo_by_group <- ggplot(data.frame(Y_train, clusters_train), aes(x = Y_train, fill = clusters_train)) +
  geom_histogram(binwidth = 0.2, position = "identity", alpha = 0.7,color = "black")+
  labs(title = "Plot", x = "Y", y = "frequence") +
  scale_fill_manual(values = c("1" = "darkgray","2"= "darkblue","3"= "blue","4"="#87CEFA"), name = "groups")
print(histo_by_group)
##
#X1-Y
scatter_plot <- ggplot(as.data.frame(data_train), aes(x = X1, y = Y, color = clusters_train)) +
  geom_point(alpha = 0.6) +
  labs(title = "plot X1-Y", x = "X1", y = "Y") +
  scale_color_manual(values = c("1" = "darkgray","2"= "darkblue","3"= "blue","4"="#87CEFA"), name="groups")
print(scatter_plot)

#x2-Y1
scatter_plot <- ggplot(as.data.frame(data_train), aes(x = X2, y = Y, color = clusters_train)) +
  geom_point(alpha = 0.6) +
  labs(title = "plot X2-Y", x = "X2", y = "Y") +
  scale_color_manual(values = c("1" = "darkgray","2"= "darkblue","3"= "blue","4"="#87CEFA"), name="groups")
print(scatter_plot)

#x3-Y1
scatter_plot <- ggplot(as.data.frame(data_train), aes(x = X3, y = Y, color = clusters_train)) +
  geom_point(alpha = 0.6) +
  labs(title = "plot X3-Y", x = "X3", y = "Y") +
  scale_color_manual(values = c("1" = "darkgray","2"= "darkblue","3"= "blue","4"="#87CEFA"), name="groups")
print(scatter_plot)

#x4-Y1
scatter_plot <- ggplot(as.data.frame(data_train), aes(x = X4, y = Y, color = clusters_train)) +
  geom_point(alpha = 0.6) +
  labs(title = "plot X4-Y", x = "X4", y = "Y") +
  scale_color_manual(values = c("1" = "darkgray","2"= "darkblue","3"= "blue","4"="#87CEFA"), name="groups")
print(scatter_plot)

#x5-Y1
scatter_plot <- ggplot(as.data.frame(data_train), aes(x = X5, y = Y, color = clusters_train)) +
  geom_point(alpha = 0.6) +
  labs(title = "plot X5-Y", x = "X5", y = "Y") +
  scale_color_manual(values = c("1" = "darkgray","2"= "darkblue","3"= "blue","4"="#87CEFA"), name="groups")
print(scatter_plot)

#x6-Y1
scatter_plot <- ggplot(as.data.frame(data_train), aes(x = X6, y = Y, color = clusters_train)) +
  geom_point(alpha = 0.6) +
  labs(title = "plot X6-Y", x = "X6", y = "Y") +
  scale_color_manual(values = c("1" = "#87CEFA","2"= "darkblue","3"= "blue","4"= "darkgray"), name="groups")
print(scatter_plot)

#x7-Y1
scatter_plot <- ggplot(as.data.frame(data_train), aes(x = X7, y = Y, color = clusters_train)) +
  geom_point(alpha = 0.6) +
  labs(title = "plot X7-Y", x = "X7", y = "Y") +
  scale_color_manual(values = c("1" = "#87CEFA","2"= "darkblue","3"= "blue","4"= "darkgray"), name="groups")
print(scatter_plot)

#x8-Y1
scatter_plot <- ggplot(as.data.frame(data_train), aes(x = X8, y = Y, color = clusters_train)) +
  geom_point(alpha = 0.6) +
  labs(title = "plot X8-Y", x = "X8", y = "Y") +
  scale_color_manual(values = c("1" = "#87CEFA","2"= "darkblue","3"= "blue","4"= "darkgray"), name="groups")
print(scatter_plot)

#x9-Y1
scatter_plot <- ggplot(as.data.frame(data_train), aes(x = X9, y = Y, color = clusters_train)) +
  geom_point(alpha = 0.6) +
  labs(title = "plot X9-Y", x = "X9", y = "Y") +
  scale_color_manual(values = c("1" = "#87CEFA","2"= "darkblue","3"= "blue","4"= "darkgray"), name="groups")
print(scatter_plot)

#x10-Y1
scatter_plot <- ggplot(as.data.frame(data_train), aes(x = X10, y = Y, color = clusters_train)) +
  geom_point(alpha = 0.6) +
  labs(title = "plot X10-Y", x = "X10", y = "Y") +
  scale_color_manual(values = c("1" = "#87CEFA","2"= "darkblue","3"= "blue","4"= "darkgray"), name="groups")
print(scatter_plot)

#G1/G2/G3
invcov_cont <- solve(cov(data_train[,2:6]))
#invcov_cont <- matrix(1/var(data_train[,2]))
espilon_star <- compute_lambda(as.matrix(data_train[,2:11]), nrep = 5000,invcov_cont, var_type)
lambda_true <- .01 / mean(espilon_star[,1])

#PPMX
#continuous covariates
#sigma2 known (prior on mu) (ppmx_n)
c1 <- 1
c2 <- 2
#m0 <- mean(data_rob[,2])
#S0 <- apply(data_rob[,c(2,3)], 2, var)
#S0 <- var(data_rob[,2])
#sigma2 unknown (prior on mu and sigma2) (ppmx_t)
a0 <- 1
k0 <- 4
#m <- colMeans(data_rob[,c(2,3)])
#m <- mean(data_rob[,2])
#b0 <- apply(data_rob[,c(2,3)], 2, var)
#b0 <- var(data_rob[,2])
#binary covariates
a <- 0.1
b <- 0.1
#categorical covariates
categs_1 <- NULL
alpha_1 <- rep(0.1, length(categs_1))

g_params <- list(fun = "g1", lambda = 1, alpha = 1, cov = invcov_cont)
g_params <- list("ppmx_n", c1, c2, a, b, categs_1, alpha_1)
g_params <- list("ppmx_t", a0, k0, a, b, categs_1, alpha_1)

ngg_params <- list(sigma = 0.2, k = 0.3)
P0_params <- list(mu0 = mean(Y_train), lambda0 = 0.01, a0 = 1, b0 = 2)
niter <- 5000
nburn <- 1000
thin <- 1
nGibbs <- 0
thinGibbs <- 10
var_type <- c(1,1,1,1,1)


system.time(result_0 <- run_mcmc(data=as.matrix(data_sample), type=var_type, g_params=g_params, 
                                 ngg_params=ngg_params, P0_params=P0_params, niter=niter, 
                                 nburn=nburn, thin=thin, nGibbs = nGibbs, thin_Gibbs = thinGibbs))

freq_chain<-t(result_0$freq_chain)[-1,]
matplot(freq_chain, type = "l", xlab = "", ylab = "", main = "Split-Merge(0, 10) - NNIG", ylim = c(0, 1), col = "black", axes = FALSE)
axis(1)
axis(2, at = seq(0.0, 1.0, by = 0.2), labels = format(seq(0.0, 1.0, by = 0.2), nsmall = 1), las = 1)
box()
labels <- as.factor(result_0$best_clus_binder)
table(labels)
histo_by_group <- ggplot(as.data.frame(data_sample), aes(x = Y, fill = labels)) +
  geom_histogram(binwidth = 0.6, position = "identity", alpha = 0.7, color = "black") +
  labs(title = "Clustering with G0", x = "Y", y = "frequency") +
  scale_fill_manual(values = c("1" = "#87CEFA","0"= "darkblue","2"= "blue","3"= "darkgray"), name = "groups")
print(histo_by_group)

arandi(clusters_sample, result_0$best_clus_binder)


#PREDICTION
grid_y1 <- seq(min(Y_train), max(Y_train), by = 0.1)
grid_y2 <- c(0)
niter <- 2000
nburn <- 1000
thin <- 1
nGibbs <- 0
thinGibbs <- 1

g_params <- list(fun = "g0", lambda = 0.05, alpha = 1, cov = invcov_cont)
g_params <- list("ppmx_n", c1, c2, a, b, categs_1, alpha_1)
g_params <- list("ppmx_t", a0, k0, a, b, categs_1, alpha_1)

result_pred <- run_mcmc_pred(as.matrix(data_sample), var_type, g_params, ngg_params,
                             P0_params, Y_test, as.matrix(X_test[-c(1:54),]), grid_y1, niter, nburn, thin, nGibbs, thinGibbs,
                             TRUE, FALSE, FALSE, TRUE)

ggplot(data.frame(x = grid_y1, y = c(colMeans(result_pred$pred_grid)), 
                  ylow = apply(result_pred$pred_grid, 2, quantile, p = 0.05), 
                  yup = apply(result_pred$pred_grid, 2, quantile, p = 0.95))) +
  theme_bw() +
  geom_line(aes(x = x, y = y)) + 
  xlab("y") +
  ylab("density") +
  ggtitle("N-NIG") + 
  geom_vline(aes(xintercept = Y_test[55]), lty = 2, col = 2) +
  geom_ribbon(aes(x = x, ymin = ylow, ymax = yup), alpha = 0.25) +
  ylim(c(0, 0.6))+
  theme(plot.title = element_text(hjust = 0.5))

#P = 10----
Rcpp::sourceCpp('split_with_g_ngg.cpp')

set.seed(123)
n <- 10060
num_clusters <- 4
cluster_means <- c(-3.5, -1.5, 1.5, 3.5)
Y <- c(rnorm(n/num_clusters, mean = cluster_means[1], sd = 0.8),
       rnorm(n/num_clusters, mean = cluster_means[2], sd = 0.8),
       rnorm(n/num_clusters, mean = cluster_means[3], sd = 0.8),
       rnorm(n/num_clusters, mean = cluster_means[4], sd = 0.8))

p <- 10
X_10 <- genera_dataset(n, p, num_clusters)
X_10_mod <- modifica_covariate(X_10, num_clusters, num_cov_mod_per_cluster = 10)

clusters = factor(rep(1:num_clusters, each = n/num_clusters))

data <- data.frame(Y,X_10_mod)
#scaling
data[,2:11] <- scale(data[,2:11])
#test set
sampled_test <- sample(nrow(data), 60)
data_test <- data[sampled_test,]
Y_test <- data_test[,1]
X_test <- data_test[,2:11]
clusters_test <- clusters[sampled_test]
#training set
data_train <- data[-sampled_test,]
clusters_train <- clusters[-sampled_test]
Y_train <- data_train[,1]
X_train <- data_train[,2:11]
sampled_train <- sample(nrow(data_train), 100)
data_sample <- data_train[sampled_train, ]
Y_sample <- data_sample[,1]
X_sample <- data_sample[,2:11]
clusters_sample <- clusters_train[sampled_train]


histo_by_group <- ggplot(data.frame(Y_train, clusters_train), aes(x = Y_train, fill = clusters_train)) +
  geom_histogram(binwidth = 0.2, position = "identity", alpha = 0.7,color = "black")+
  labs(title = "Plot", x = "Y", y = "frequence") +
  scale_fill_manual(values = c("1" = "darkgray","2"= "darkblue","3"= "blue","4"="#87CEFA"), name = "groups")
print(histo_by_group)

#X1-Y
scatter_plot <- ggplot(as.data.frame(data_train), aes(x = X1, y = Y, color = clusters_train)) +
  geom_point(alpha = 0.6) +
  labs(title = "plot X1-Y", x = "X1", y = "Y") +
  scale_color_manual(values = c("1" = "darkgray","2"= "darkblue","3"= "blue","4"="#87CEFA"), name="groups")
print(scatter_plot)

#x2-Y1
scatter_plot <- ggplot(as.data.frame(data_train), aes(x = X2, y = Y, color = clusters_train)) +
  geom_point(alpha = 0.6) +
  labs(title = "plot X2-Y", x = "X2", y = "Y") +
  scale_color_manual(values = c("1" = "darkgray","2"= "darkblue","3"= "blue","4"="#87CEFA"), name="groups")
print(scatter_plot)

#x3-Y1
scatter_plot <- ggplot(as.data.frame(data_train), aes(x = X3, y = Y, color = clusters_train)) +
  geom_point(alpha = 0.6) +
  labs(title = "plot X3-Y", x = "X3", y = "Y") +
  scale_color_manual(values = c("1" = "darkgray","2"= "darkblue","3"= "blue","4"="#87CEFA"), name="groups")
print(scatter_plot)

#x4-Y1
scatter_plot <- ggplot(as.data.frame(data_train), aes(x = X4, y = Y, color = clusters_train)) +
  geom_point(alpha = 0.6) +
  labs(title = "plot X4-Y", x = "X4", y = "Y") +
  scale_color_manual(values = c("1" = "darkgray","2"= "darkblue","3"= "blue","4"="#87CEFA"), name="groups")
print(scatter_plot)

#x5-Y1
scatter_plot <- ggplot(as.data.frame(data_train), aes(x = X5, y = Y, color = clusters_train)) +
  geom_point(alpha = 0.6) +
  labs(title = "plot X5-Y", x = "X5", y = "Y") +
  scale_color_manual(values = c("1" = "darkgray","2"= "darkblue","3"= "blue","4"="#87CEFA"), name="groups")
print(scatter_plot)

#x6-Y1
scatter_plot <- ggplot(as.data.frame(data_train), aes(x = X6, y = Y, color = clusters_train)) +
  geom_point(alpha = 0.6) +
  labs(title = "plot X6-Y", x = "X6", y = "Y") +
  scale_color_manual(values = c("1" = "#87CEFA","2"= "darkblue","3"= "blue","4"= "darkgray"), name="groups")
print(scatter_plot)

#x7-Y1
scatter_plot <- ggplot(as.data.frame(data_train), aes(x = X7, y = Y, color = clusters_train)) +
  geom_point(alpha = 0.6) +
  labs(title = "plot X7-Y", x = "X7", y = "Y") +
  scale_color_manual(values = c("1" = "#87CEFA","2"= "darkblue","3"= "blue","4"= "darkgray"), name="groups")
print(scatter_plot)

#x8-Y1
scatter_plot <- ggplot(as.data.frame(data_train), aes(x = X8, y = Y, color = clusters_train)) +
  geom_point(alpha = 0.6) +
  labs(title = "plot X8-Y", x = "X8", y = "Y") +
  scale_color_manual(values = c("1" = "#87CEFA","2"= "darkblue","3"= "blue","4"= "darkgray"), name="groups")
print(scatter_plot)

#x9-Y1
scatter_plot <- ggplot(as.data.frame(data_train), aes(x = X9, y = Y, color = clusters_train)) +
  geom_point(alpha = 0.6) +
  labs(title = "plot X9-Y", x = "X9", y = "Y") +
  scale_color_manual(values = c("1" = "#87CEFA","2"= "darkblue","3"= "blue","4"= "darkgray"), name="groups")
print(scatter_plot)

#x10-Y1
scatter_plot <- ggplot(as.data.frame(data_train), aes(x = X10, y = Y, color = clusters_train)) +
  geom_point(alpha = 0.6) +
  labs(title = "plot X10-Y", x = "X10", y = "Y") +
  scale_color_manual(values = c("1" = "#87CEFA","2"= "darkblue","3"= "blue","4"= "darkgray"), name="groups")
print(scatter_plot)



#G1/G2/G3
invcov_cont <- solve(cov(data_train[,2:11]))
#invcov_cont <- matrix(1/var(data_train[,2]))
espilon_star <- compute_lambda(as.matrix(data_train[,2:11]), nrep = 5000,invcov_cont, var_type)
lambda_true <- .01 / mean(espilon_star[,1])

#PPMX
#continuous covariates
#sigma2 known (prior on mu) (ppmx_n)
c1 <- 1
c2 <- 2
#m0 <- mean(data_rob[,2])
#S0 <- apply(data_rob[,c(2,3)], 2, var)
#S0 <- var(data_rob[,2])
#sigma2 unknown (prior on mu and sigma2) (ppmx_t)
a0 <- 1
k0 <- 4
#m <- colMeans(data_rob[,c(2,3)])
#m <- mean(data_rob[,2])
#b0 <- apply(data_rob[,c(2,3)], 2, var)
#b0 <- var(data_rob[,2])
#binary covariates
a <- 0.1
b <- 0.1
#categorical covariates
categs_1 <- NULL
alpha_1 <- rep(0.1, length(categs_1))

g_params <- list(fun = "g0", lambda = 0.7, alpha = 1, cov = invcov_cont)
g_params <- list("ppmx_n", c1, c2, a, b, categs_1, alpha_1)
g_params <- list("ppmx_t", a0, k0, a, b, categs_1, alpha_1)

ngg_params <- list(sigma = 0.2, k = 0.3)
P0_params <- list(mu0 = mean(Y_train), lambda0 = 0.01, a0 = 1, b0 = 2)
niter <- 5000
nburn <- 1000
thin <- 1
nGibbs <- 0
thinGibbs <- 10
var_type <- rep(1,10)

system.time(result_0 <- run_mcmc(data=as.matrix(data_sample), type=var_type, g_params=g_params, 
                                 ngg_params=ngg_params, P0_params=P0_params, niter=niter, 
                                 nburn=nburn, thin=thin, nGibbs = nGibbs, thin_Gibbs = thinGibbs))


freq_chain<-t(result_0$freq_chain)[-1,]
matplot(freq_chain, type = "l", xlab = "", ylab = "", main = "Split-Merge(0, 10) - NNIG", ylim = c(0, 1), col = "black", axes = FALSE)
axis(1)
axis(2, at = seq(0.0, 1.0, by = 0.2), labels = format(seq(0.0, 1.0, by = 0.2), nsmall = 1), las = 1)
box()

labels <- as.factor(result_0$best_clus_binder)
table(labels)
histo_by_group <- ggplot(as.data.frame(data_sample), aes(x = Y, fill = labels)) +
  geom_histogram(binwidth = 1, position = "identity", alpha = 0.7, color = "black") +
  labs(title = "Plot", x = "Y", y = "frequency") +
  scale_fill_discrete(name = "groups")
print(histo_by_group)

arandi(clusters_sample, result_0$best_clus_binder)



#PREDICTION
grid_y1 <- seq(min(Y_train), max(Y_train), by = 0.1)
grid_y2 <- c(0)
niter <- 2000
nburn <- 1000
thin <- 1
nGibbs <- 0
thinGibbs <- 1

g_params <- list(fun = "g1", lambda = 0.7, alpha = 1, cov = invcov_cont)
g_params <- list("ppmx_n", c1, c2, a, b, categs_1, alpha_1)
g_params <- list("ppmx_t", a0, k0, a, b, categs_1, alpha_1)

result_pred <- run_mcmc_pred(as.matrix(data_sample), var_type, g_params, ngg_params,
                             P0_params, Y_test, as.matrix(X_test[-c(1:21),]), grid_y1, niter, nburn, thin, nGibbs, thinGibbs,
                             TRUE, FALSE, FALSE, TRUE)

ggplot(data.frame(x = grid_y1, y = c(colMeans(result_pred$pred_grid)), 
                  ylow = apply(result_pred$pred_grid, 2, quantile, p = 0.05), 
                  yup = apply(result_pred$pred_grid, 2, quantile, p = 0.95))) +
  theme_bw() +
  geom_line(aes(x = x, y = y)) + 
  xlab("y") +
  ylab("density") +
  ggtitle("Predictive distribution with N-NIG ") + 
  geom_vline(aes(xintercept = Y_test[22]), lty = 2, col = 2) +
  geom_ribbon(aes(x = x, ymin = ylow, ymax = yup), alpha = 0.25) +
  ylim(c(0, 0.6))


#P = 20----
Rcpp::sourceCpp('split_with_g_ngg.cpp')

set.seed(123)
n <- 10060
num_clusters <- 4
cluster_means <- c(-3, -1.5, 1.5, 3)
Y <- c(rnorm(n/num_clusters, mean = cluster_means[1], sd = 0.6),
       rnorm(n/num_clusters, mean = cluster_means[2], sd = 0.6),
       rnorm(n/num_clusters, mean = cluster_means[3], sd = 0.6),
       rnorm(n/num_clusters, mean = cluster_means[4], sd = 0.6))

p <- 20
X_20 <- genera_dataset(n, p, num_clusters)
X_20_mod <- modifica_covariate(X_20, num_clusters, num_cov_mod_per_cluster = 8)

clusters = factor(rep(1:num_clusters, each = n/num_clusters))

data <- data.frame(Y,X_20_mod)
#scaling
data[,2:21] <- scale(data[,2:21])
#test set
sampled_test <- sample(nrow(data), 60)
data_test <- data[sampled_test,]
Y_test <- data_test[,1]
X_test <- data_test[,2:21]
clusters_test <- clusters[sampled_test]
#training set
data_train <- data[-sampled_test,]
clusters_train <- clusters[-sampled_test]
Y_train <- data_train[,1]
X_train <- data_train[,2:21]
sampled_train <- sample(nrow(data_train), 100)
data_sample <- data_train[sampled_train, ]
Y_sample <- data_sample[,1]
X_sample <- data_sample[,2:21]
clusters_sample <- clusters_train[sampled_train]


histo_by_group <- ggplot(data.frame(Y_train, clusters_train), aes(x = Y_train, fill = clusters_train)) +
  geom_histogram(binwidth = 0.2, position = "identity", alpha = 0.7,color = "black")+
  labs(title = "Plot", x = "Y", y = "frequence") +
  scale_fill_manual(values = c("1" = "darkgray","2"= "darkblue","3"= "blue","4"="#87CEFA"), name = "groups")
print(histo_by_group)


##
#X1-Y
scatter_plot <- ggplot(as.data.frame(data_train), aes(x = X1, y = Y, color = clusters_train)) +
  geom_point(alpha = 0.6) +
  labs(title = "plot X1-Y", x = "X1", y = "Y") +
  scale_color_manual(values = c("1" = "darkgray","2"= "darkblue","3"= "blue","4"="#87CEFA"), name="groups")
print(scatter_plot)

#x2-Y1
scatter_plot <- ggplot(as.data.frame(data_train), aes(x = X2, y = Y, color = clusters_train)) +
  geom_point(alpha = 0.6) +
  labs(title = "plot X2-Y", x = "X2", y = "Y") +
  scale_color_manual(values = c("1" = "darkgray","2"= "darkblue","3"= "blue","4"="#87CEFA"), name="groups")
print(scatter_plot)

#x3-Y1
scatter_plot <- ggplot(as.data.frame(data_train), aes(x = X3, y = Y, color = clusters_train)) +
  geom_point(alpha = 0.6) +
  labs(title = "plot X3-Y", x = "X3", y = "Y") +
  scale_color_manual(values = c("1" = "darkgray","2"= "darkblue","3"= "blue","4"="#87CEFA"), name="groups")
print(scatter_plot)

#x4-Y1
scatter_plot <- ggplot(as.data.frame(data_train), aes(x = X4, y = Y, color = clusters_train)) +
  geom_point(alpha = 0.6) +
  labs(title = "plot X4-Y", x = "X4", y = "Y") +
  scale_color_manual(values = c("1" = "darkgray","2"= "darkblue","3"= "blue","4"="#87CEFA"), name="groups")
print(scatter_plot)

#x5-Y1
scatter_plot <- ggplot(as.data.frame(data_train), aes(x = X5, y = Y, color = clusters_train)) +
  geom_point(alpha = 0.6) +
  labs(title = "plot X5-Y", x = "X5", y = "Y") +
  scale_color_manual(values = c("1" = "darkgray","2"= "darkblue","3"= "blue","4"="#87CEFA"), name="groups")
print(scatter_plot)

#x6-Y1
scatter_plot <- ggplot(as.data.frame(data_train), aes(x = X6, y = Y, color = clusters_train)) +
  geom_point(alpha = 0.6) +
  labs(title = "plot X6-Y", x = "X6", y = "Y") +
  scale_color_manual(values = c("1" = "#87CEFA","2"= "darkblue","3"= "blue","4"= "darkgray"), name="groups")
print(scatter_plot)

#x7-Y1
scatter_plot <- ggplot(as.data.frame(data_train), aes(x = X7, y = Y, color = clusters_train)) +
  geom_point(alpha = 0.6) +
  labs(title = "plot X7-Y", x = "X7", y = "Y") +
  scale_color_manual(values = c("1" = "#87CEFA","2"= "darkblue","3"= "blue","4"= "darkgray"), name="groups")
print(scatter_plot)

#x8-Y1
scatter_plot <- ggplot(as.data.frame(data_train), aes(x = X8, y = Y, color = clusters_train)) +
  geom_point(alpha = 0.6) +
  labs(title = "plot X8-Y", x = "X8", y = "Y") +
  scale_color_manual(values = c("1" = "#87CEFA","2"= "darkblue","3"= "blue","4"= "darkgray"), name="groups")
print(scatter_plot)

#x9-Y1
scatter_plot <- ggplot(as.data.frame(data_train), aes(x = X9, y = Y, color = clusters_train)) +
  geom_point(alpha = 0.6) +
  labs(title = "plot X9-Y", x = "X9", y = "Y") +
  scale_color_manual(values = c("1" = "#87CEFA","2"= "darkblue","3"= "blue","4"= "darkgray"), name="groups")
print(scatter_plot)

#x10-Y1
scatter_plot <- ggplot(as.data.frame(data_train), aes(x = X10, y = Y, color = clusters_train)) +
  geom_point(alpha = 0.6) +
  labs(title = "plot X10-Y", x = "X10", y = "Y") +
  scale_color_manual(values = c("1" = "#87CEFA","2"= "darkblue","3"= "blue","4"= "darkgray"), name="groups")
print(scatter_plot)



#G1/G2/G3
invcov_cont <- solve(cov(data_train[,2:21]))
#invcov_cont <- matrix(1/var(data_train[,2]))
espilon_star <- compute_lambda(as.matrix(data_train[,2:11]), nrep = 5000,invcov_cont, var_type)
lambda_true <- .01 / mean(espilon_star[,1])

#PPMX
#continuous covariates
#sigma2 known (prior on mu) (ppmx_n)
c1 <- 1
c2 <- 2
#m0 <- mean(data_rob[,2])
#S0 <- apply(data_rob[,c(2,3)], 2, var)
#S0 <- var(data_rob[,2])
#sigma2 unknown (prior on mu and sigma2) (ppmx_t)
a0 <- 1
k0 <- 4
#m <- colMeans(data_rob[,c(2,3)])
#m <- mean(data_rob[,2])
#b0 <- apply(data_rob[,c(2,3)], 2, var)
#b0 <- var(data_rob[,2])
#binary covariates
a <- 0.1
b <- 0.1
#categorical covariates
categs_1 <- NULL
alpha_1 <- rep(0.1, length(categs_1))

g_params <- list(fun = "g1", lambda = 0.5, alpha = 1, cov = invcov_cont)
g_params <- list("ppmx_n", c1, c2, a, b, categs_1, alpha_1)
g_params <- list("ppmx_t", a0, k0, a, b, categs_1, alpha_1)

ngg_params <- list(sigma = 0.2, k = 0.3)
P0_params <- list(mu0 = mean(Y_train), lambda0 = 0.01, a0 = 1, b0 = 2)
niter <- 5000
nburn <- 1000
thin <- 1
nGibbs <- 0
thinGibbs <- 10
var_type <- rep(1,20)

system.time(result_0 <- run_mcmc(data=as.matrix(data_sample), type=var_type, g_params=g_params, 
                                 ngg_params=ngg_params, P0_params=P0_params, niter=niter, 
                                 nburn=nburn, thin=thin, nGibbs = nGibbs, thin_Gibbs = thinGibbs))

freq_chain<-t(result_0$freq_chain)[-1,]
matplot(freq_chain, type = "l", xlab = "", ylab = "", main = "Split-Merge(0, 10) - NNIG", ylim = c(0, 1), col = "black", axes = FALSE)
axis(1)
axis(2, at = seq(0.0, 1.0, by = 0.2), labels = format(seq(0.0, 1.0, by = 0.2), nsmall = 1), las = 1)
box()

labels <- as.factor(result_0$best_clus_binder)
table(labels)
histo_by_group <- ggplot(as.data.frame(data_sample), aes(x = Y, fill = labels)) +
  geom_histogram(binwidth = 0.5, position = "identity", alpha = 0.7, color = "black") +
  labs(title = "Plot", x = "Y", y = "frequency") +
  scale_fill_discrete(name = "groups")
print(histo_by_group)
arandi(clusters_sample, result_0$best_clus_binder)


#PREDICTION
grid_y1 <- seq(min(Y_train), max(Y_train), by = 0.10)
grid_y2 <- c(0)
niter <- 2000
nburn <- 1000
thin <- 1
nGibbs <- 0
thinGibbs <- 1

g_params <- list(fun = "g1", lambda = 0.5, alpha = 1, cov = invcov_cont)
g_params <- list("ppmx_n", c1, c2, a, b, categs_1, alpha_1)
g_params <- list("ppmx_t", a0, k0, a, b, categs_1, alpha_1)

result_pred <- run_mcmc_pred(as.matrix(data_sample), var_type, g_params, ngg_params,
                             P0_params, Y_test, as.matrix(X_test[-c(1:9),]), grid_y1, niter, nburn, thin, nGibbs, thinGibbs,
                             TRUE, FALSE, FALSE, TRUE)

ggplot(data.frame(x = grid_y1, y = c(colMeans(result_pred$pred_grid)), 
                  ylow = apply(result_pred$pred_grid, 2, quantile, p = 0.05), 
                  yup = apply(result_pred$pred_grid, 2, quantile, p = 0.95))) +
  theme_bw() +
  geom_line(aes(x = x, y = y)) + 
  xlab("y") +
  ylab("density") +
  ggtitle("Predictive distribution with N-NIG ") + 
  geom_vline(aes(xintercept = Y_test[10]), lty = 2, col = 2) +
  geom_ribbon(aes(x = x, ymin = ylow, ymax = yup), alpha = 0.25) +
  ylim(c(0, 0.7))

#mixture of normals with a regression on covariates----
#Data Generation: (x1,x2)~N, x3~Ber, x4~Ber ; Y ~ N(beta'x, 0.5).
#Model Assumption: Gaussian Mixture components with unknown means and variances,
#                  Y ~ N(beta'x, 0.5)
Rcpp::sourceCpp('split_with_g_ngg.cpp')

set.seed(123)
group_1 <- cbind(rmvnorm(n=3775, mean=c(-3,3),sigma= diag(.5, 2)), 
                 sample(c(0,1), size = 3775, prob = c(0.9, 0.1), replace = T),
                 sample(c(0,1), size = 3775, prob = c(0.9, 0.1), replace = T))
group_1 <- cbind(apply(group_1, 1, function(x) rnorm(1, 1 + x %*% c(5, 2, 1, 0), .5)), group_1)

group_2<- cbind(rmvnorm(3775, c(0,0), diag(.5, 2)), 
                sample(c(0,1), size = 3775, prob = c(0.5, 0.5), replace = T),
                sample(c(0,1), size = 3775, prob = c(0.5, 0.5), replace = T))
group_2 <- cbind(apply(group_2, 1, function(x) rnorm(1, 4 + x %*% c(2, -2, 1, -1), .5)), group_2)

group_3 <- cbind(rmvnorm(2510, c(3,3), diag(.5, 2)), 
                 sample(c(0,1), size = 2510, prob = c(0.1, 0.9), replace = T),
                 sample(c(0,1), size = 2510, prob = c(0.1, 0.9), replace = T))
group_3 <- cbind(apply(group_3, 1, function(x) rnorm(1,  -1 + x %*% c(-5, -2, -1, 1), .5)), group_3)

data <- rbind(group_1, group_2, group_3)
new_column_names <- c("Y", "X1", "x2", "x3", "x4")
colnames(data) <- new_column_names

#scaling
data[,c(2,3)] <- scale(data[,c(2,3)])
clusters <- factor(rep(1:3, c(3775, 3775, 2510)))
#test set
sampled_test <- sample(nrow(data), 60)
data_test <- data[sampled_test,]
Y_test <- data_test[,1]
X_test <- data_test[,c(2,3,4,5)]
clusters_test <- clusters[sampled_test]
#training set
data_train <- data[-sampled_test,]
clusters_train <- clusters[-sampled_test]
Y_train <- data_train[,1]
X_train <- data_train[,c(2,3,4,5)]
sampled_train <- sample(nrow(data_train), 100)
data_sample <- data_train[sampled_train, ]
Y_sample <- data_sample[,1]
X_sample <- data_sample[,c(2,3,4,5)]
clusters_sample <- clusters_train[sampled_train]

histo_by_group <- ggplot(data.frame(Y_train, clusters_train), aes(x = Y_train, fill = clusters_train)) +
  geom_histogram(binwidth = 1, position = "identity", alpha = 0.7,color = "black")+
  labs(title = "Plot", x = "Y", y = "frequence") +
  scale_fill_manual(values = c("1" = "blue","2" = "darkblue","3" = "#87CEFA"), name = "groups")
print(histo_by_group)

##
#x1-Y
scatter_plot <- ggplot(as.data.frame(data_train), aes(x = X1, y = Y, color = clusters_train)) +
  geom_point(alpha = 0.6) +
  labs(title = "plot X1-Y", x = "X1", y = "Y") +
  scale_color_manual(values = c("1" = "blue","2" = "darkblue","3" = "#87CEFA"), name="groups")
print(scatter_plot)

#x2-Y1
scatter_plot <- ggplot(as.data.frame(data_train), aes(x = x2, y = Y, color = clusters_train)) +
  geom_point(alpha = 0.6) +
  labs(title = "plot X2-Y", x = "X2", y = "Y") +
  scale_color_manual(values = c("1" = "blue","2" = "darkblue","3" = "#87CEFA"), name="groups")
print(scatter_plot)

#x3-Y1
scatter_plot <- ggplot(as.data.frame(data_train), aes(x = x3, y = Y, color = clusters_train)) +
  geom_point(alpha = 0.6) +
  labs(title = "plot X3-Y", x = "X3", y = "Y") +
  scale_color_manual(values = c("1" = "blue","2" = "darkblue","3" = "#87CEFA"), name="groups")
print(scatter_plot)

#x4-Y1
scatter_plot <- ggplot(as.data.frame(data_train), aes(x = x4, y = Y, color = clusters_train)) +
  geom_point(alpha = 0.6) +
  labs(title = "plot X4-Y", x = "X4", y = "Y") +
  scale_color_manual(values = c("1" = "blue","2" = "darkblue","3" = "#87CEFA"), name="groups")
print(scatter_plot)


#G1/G2/G3
invcov_cont <- solve(cov(data_train[,c(2,3)]))
#PPMX
#continuous covariates
#sigma2 known (prior on mu) (ppmx_n)
c1 <- 1
c2 <- 2
#m0 <- colMeans(data_rob[,c(2,3)])
#S0 <- apply(data_rob[,c(2,3)], 2, var)
#sigma2 unknown (prior on mu and sigma2) (ppmx_t)
a0 <- 1
k0 <- 2 #4
#m <- colMeans(data_rob[,c(2,3)])
#b0 <- apply(data_rob[,c(2,3)], 2, var)
#binary covariates
a <- 0.1
b <- 0.1
#categorical covariates
categs_1 <- NULL
alpha_1 <- rep(0.1, length(categs_1))

g_params <- list(fun = "g3", lambda = 0.5, alpha = 1, cov = invcov_cont)
g_params <- list("ppmx_n", c1, c2, a, b, categs_1, alpha_1)
g_params <- list("ppmx_t", a0, k0, a, b, categs_1, alpha_1)


beta0 <- numeric(5)
B0 <- diag(5)/0.01
P0_params <- list(beta0 = beta0, B0 = B0, a0 = 2.0, b0 = 1.0)
ngg_params <- list(sigma = 0.2, k = 0.3)
niter <- 5000
nburn <- 1000
thin <- 1
nGibbs <- 0
thinGibbs <-1
var_type <- c(1,1,0,0)

system.time(result_0 <- run_mcmc(data=as.matrix(data_sample), type=var_type, g_params=g_params, 
                                 ngg_params=ngg_params, P0_params=P0_params, niter=niter, 
                                 nburn=nburn, thin=thin, nGibbs = nGibbs, thin_Gibbs = thinGibbs))

freq_chain<-t(result_0$freq_chain)[-1,]
matplot(freq_chain, type = "l", xlab = "", ylab = "", main = "Split-Merge(0, 10) - NN", ylim = c(0, 1), col = "black", axes = FALSE)
axis(1)
axis(2, at = seq(0.0, 1.0, by = 0.2), labels = format(seq(0.0, 1.0, by = 0.2), nsmall = 1), las = 1)
box()

labels <- as.factor(result_0$best_clus_binder)
table(labels)
histo_by_group <- ggplot(as.data.frame(data_sample), aes(x = Y, fill = labels)) +
  geom_histogram(binwidth = 1, position = "identity", alpha = 0.7, color = "black") +
  labs(title = "Plot", x = "Y", y = "frequency") +
  scale_fill_discrete(name = "groups")
print(histo_by_group)


#PREDICTION
grid_y1 <- seq(min(Y_train), max(Y_train), by = 0.3)
grid_y2 <- c(0)
niter <- 2000
nburn <- 1000
thin <- 1
nGibbs <- 0
thinGibbs <-1

g_params <- list(fun = "g3", lambda = 0.5, alpha = 1, cov = invcov_cont)
g_params <- list("ppmx_n", c1, c2, a, b, categs_1, alpha_1)
g_params <- list("ppmx_t", a0, k0, a, b, categs_1, alpha_1)

result_pred <- run_mcmc_pred(as.matrix(data_sample), var_type, g_params, ngg_params,
                             P0_params, Y_test, as.matrix(X_test[-c(1:20),]), grid_y1, niter, nburn, thin, nGibbs, thinGibbs,
                             TRUE, FALSE, FALSE, TRUE)

ggplot(data.frame(x = grid_y1, y = c(colMeans(result_pred$pred_grid)), 
                  ylow = apply(result_pred$pred_grid, 2, quantile, p = 0.05), 
                  yup = apply(result_pred$pred_grid, 2, quantile, p = 0.95))) +
  theme_bw() +
  geom_line(aes(x = x, y = y)) + 
  xlab("y") +
  ylab("density") +
  ggtitle("Predictive distribution with G3") + 
  geom_vline(aes(xintercept = Y_test[21]), lty = 2, col = 2) +
  geom_ribbon(aes(x = x, ymin = ylow, ymax = yup), alpha = 0.25) +
  ylim(c(0, 0.70))+
  xlim(c(-15, 11))



#mixture of bernoulli----
Rcpp::sourceCpp('split_with_g_ngg.cpp')


set.seed(123)
group_1 <- cbind(rmvnorm(n=3775, mean=c(-2.7, 2.7), sigma=diag(0.4, 2)), 
                 sample(c(0, 1), size=3775, prob=c(0.9, 0.1), replace=TRUE),
                 sample(c(0, 1), size=3775, prob=c(0.1, 0.9), replace=TRUE))
logistic_values <- apply(group_1, 1, function(x) {
  round(1 / (1 + exp(rnorm(1, mean=1 + x %*% c(-2, -2, 1, 0), sd=0.5))))
})
group_1 <- cbind(logistic_values, group_1)


group_2<- cbind(rmvnorm(3775, c(0,0), diag(.4, 2)), 
                sample(c(0,1), size = 3775, prob = c(0.3, 0.7), replace = T),
                sample(c(0,1), size = 3775, prob = c(0.7, 0.3), replace = T))
logistic_values <- apply(group_2, 1, function(x) {
  round(1 / (1 + exp(rnorm(1, mean=-1 + x %*% c(2, -2, 1, -1), sd=0.5))))
})
group_2 <- cbind(logistic_values, group_2)


group_3 <- cbind(rmvnorm(2510, c(2.7,5), diag(.4, 2)), 
                 sample(c(0,1), size = 2510, prob = c(0.1, 0.9), replace = T),
                 sample(c(0,1), size = 2510, prob = c(0.9, 0.1), replace = T))
logistic_values <- apply(group_3, 1, function(x) {
  round(1 / (1 + exp(rnorm(1, mean=-1 + x %*% c(5, -2, -1, 1), sd=0.5))))
})
group_3 <- cbind(logistic_values, group_3)


data <- rbind(group_1, group_2, group_3)
new_column_names <- c("Y", "X1", "x2", "x3", "x4")
colnames(data) <- new_column_names

#scaling
data[,c(2,3)] <- scale(data[,c(2,3)])
clusters <- factor(rep(1:3, c(3775, 3775, 2510)))
#test set
sampled_test <- sample(nrow(data), 60)
data_test <- data[sampled_test,]
Y_test <- data_test[,1]
X_test <- data_test[,c(2,3,4,5)]
clusters_test <- clusters[sampled_test]
#training set
data_train <- data[-sampled_test,]
clusters_train <- clusters[-sampled_test]
Y_train <- data_train[,1]
X_train <- data_train[,c(2,3,4,5)]
sampled_train <- sample(nrow(data_train), 100)
data_sample <- data_train[sampled_train, ]
Y_sample <- data_sample[,1]
X_sample <- data_sample[,c(2,3,4,5)]
clusters_sample <- clusters_train[sampled_train]

histo_by_group <- ggplot(data.frame(data_train[,3], clusters_train), aes(x = data_train[,3], fill = clusters_train)) +
  geom_histogram(binwidth = 0.08, position = "identity", alpha = 1,color = "black")+
  labs(title = "Plot", x = "X2",fill = "clusters") +
  scale_fill_manual(values = c("1" = "lightblue",  
                               "2" = "darkblue",  
                               "3" = "darkgray",  
                               "4" = "lightgray",   
                               "5" = "#E0FFFF",
                               "6" = "#4682B4",
                               "7" = "#87CEFA",
                               "8" = "#87CEEB"
  ))
print(histo_by_group)

library(dplyr)
summary_X3 <- as.data.frame(data_train) %>%
  mutate(Cluster = clusters_train) %>%       
  group_by(Cluster, x3) %>%
  summarise(Count = n()) %>%
  mutate(Covariate = "X3")

# For X4
summary_X4 <- as.data.frame(data_train) %>%
  mutate(Cluster = clusters_train) %>%
  group_by(Cluster, x4) %>%
  summarise(Count = n()) %>%
  mutate(Covariate = "X4")

# Combine the summaries
summary_data <- bind_rows(summary_X3, summary_X4)

# Convert X3 and X4 to a factor for correct plotting
summary_data$X3_X4 <- factor(ifelse(is.na(summary_data$x3), summary_data$x4, summary_data$x3), 
                             labels = c("0", "1"))

# Plot the data
ggplot(summary_data, aes(x = Cluster, y = Count, fill = X3_X4)) +
  geom_bar(stat = "identity", position = "dodge", width = 0.9) +
  facet_wrap(~Covariate, scales = "free_y") +
  labs(
    title = "Count of 0s and 1s for Covariates X3 and X4 in Each Cluster",
    x = "Clusters",
    y = "Count",
    fill = "Value"
  ) +
  scale_fill_manual(values = c("0" = "#87CEFA",  # Light Blue
                               "1" = "darkblue"), # Steel Blue
                    name = "Values") +
  scale_y_continuous(limits = c(0, 3500))+
  theme_minimal()

table(data[,1])
table(data_train[which(clusters_train=="3"),1])
summary_Y <- as.data.frame(data_train) %>%
  mutate(Cluster = clusters_train) %>%       
  group_by(Cluster, Y) %>%
  summarise(Count = n()) %>%
  mutate(Covariate = "Y")
summary_Y$Y <- factor(summary_Y$Y, labels = c("0", "1"))

ggplot(summary_Y, aes(x = Cluster, y = Count, fill = Y)) +
  geom_bar(stat = "identity", position = "dodge", width = 0.5) +
  facet_wrap(~Covariate, scales = "free_y") +
  labs(
    title = "Count of 0s and 1s for Covariates X3 and X4 in Each Cluster",
    x = "Clusters",
    y = "Count",
    fill = "Value"
  ) +
  scale_fill_manual(values = c("0" = "#87CEFA",  # Light Blue
                               "1" = "darkblue"), # Steel Blue
                    name = "Values") +
  scale_y_continuous(limits = c(0, 3500))+
  theme_minimal()

#G1/G2/G3
invcov_cont <- solve(cov(data_train[,c(2,3)]))
espilon_star <- compute_lambda(data_train[,c(2,3,4,5)], nrep = 10000,invcov_cont, var_type)
lambda_true <- .001 / mean(espilon_star[,1])

#PPMX
#continuous covariates
#sigma2 known (prior on mu) (ppmx_n)
c1 <- 1
c2 <- 10
#m0 <- colMeans(data_rob[,c(2,3)])
#S0 <- apply(data_rob[,c(2,3)], 2, var)
#sigma2 unknown (prior on mu and sigma2) (ppmx_t)
a0 <- 5
k0 <- 1
#m <- colMeans(data_rob[,c(2,3)])
#b0 <- apply(data_rob[,c(2,3)], 2, var)
#binary covariates
a <- 0.1
b <- 0.1
#categorical covariates
categs_1 <- NULL
alpha_1 <- rep(0.1, length(categs_1))


ngg_params <- list(sigma = 0.2, k = 0.3)
P0_params <- list(a0 = 1, b0 = 1)
niter <- 3000
nburn <- 1000
nGibbs <- 0
thinGibbs <- 1
thin <- 1
var_type <- c(1,1,0,0)

g_params <- list(fun = "g0", lambda = 0.2, alpha = 1, cov = invcov_cont)
g_params <- list("ppmx_n", c1, c2, a, b, categs_1, alpha_1)
g_params <- list("ppmx_t", a0, k0, a, b, categs_1, alpha_1)

system.time(result_0 <- run_mcmc(data=data_sample, type=var_type, g_params=g_params, 
                                 ngg_params=ngg_params, P0_params=P0_params, niter=niter, 
                                 nburn=nburn, thin=thin, nGibbs = nGibbs, thin_Gibbs = thinGibbs))

freq_chain<-t(result_0$freq_chain)[-1,]
matplot(freq_chain, type = "l", xlab = "", ylab = "", main = "Split-Merge(0, 10) - G0", ylim = c(0, 1), col = "black", axes = FALSE)
axis(1)
axis(2, at = seq(0.0, 1.0, by = 0.2), labels = format(seq(0.0, 1.0, by = 0.2), nsmall = 1), las = 1)
box()

labels <- as.factor(result_0$best_clus_binder)
table(labels)
table(clusters_sample)
##
data_true <- cbind(data_sample, clusters_sample)
data_res <- cbind(data_sample,labels)
label_counts <- table(labels)
labels_to_keep <- names(label_counts[label_counts > 5])
filtered_labels <- labels[labels %in% labels_to_keep]
filt_data <- data_res[labels %in% labels_to_keep, ]
filt_data_true <- data_true[labels %in% labels_to_keep, ]

df_true <- data.frame(filt_data_true[,1], filt_data_true[,6])
df <- data.frame(filt_data[,1], filt_data[,6])
colnames(df_true) <- c("Y", "clus")
colnames(df) <- c("Y", "clus")
summary_Y <- as.data.frame(df) %>%
  mutate(Cluster = clus) %>%       
  group_by(Cluster, Y) %>%
  summarise(Count = n()) %>%
  mutate(Covariate = "Y")
summary_Y$Y <- factor(summary_Y$Y, labels = c("0", "1"))

ggplot(summary_Y, aes(x = Cluster, y = Count, fill = Y)) +
  geom_bar(stat = "identity", position = "dodge", width = 0.5) +
  facet_wrap(~Covariate, scales = "free_y") +
  labs(x = "Clusters",y = "Count",fill = "Value") +scale_fill_manual(values = c("0" = "#87CEFA",  # Light Blue
                               "1" = "darkblue"),  name = "Values") +
  scale_y_continuous(limits = c(0, 40))+
  theme_minimal()

summary_true <- as.data.frame(df_true) %>%
  mutate(Cluster = clus) %>%       
  group_by(Cluster, Y) %>%
  summarise(Count = n()) %>%
  mutate(Covariate = "Y")

summary_true$Y <- factor(summary_true$Y, labels = c("0", "1"))
ggplot(summary_true, aes(x = Cluster, y = Count, fill = Y)) +
  geom_bar(stat = "identity", position = "dodge", width = 0.5) +
  facet_wrap(~Covariate, scales = "free_y") +
  labs(x = "Clusters",y = "Count",fill = "Value") +scale_fill_manual(values = c("0" = "#87CEFA",  # Light Blue
                                                                                "1" = "darkblue"),  name = "Values") +
  scale_y_continuous(limits = c(0, 40))+
  theme_minimal()
##

#mixture of multivariate bernoulli----
Rcpp::sourceCpp('split_with_g_ngg.cpp')

set.seed(123)
k <- 5  # Number of components (clusters)
m <- 6  # Number of binary attributes
n <- 1060  # Total number of observations
p <- 4  # Number of covariates
n_per_cluster <- n / k  # Number of observations per cluster

# Component probabilities (all components are equally likely)
component_probs <- rep(1/k, k)

# Probabilities for each attribute given the component
prob_matrix <- matrix(c(
  0.85, 0.85, 0.85, 0.85, 0.85, 0.85,  # Component 2
  0.10, 0.85, 0.10, 0.10, 0.10, 0.85,  # Component 2
  0.85, 0.10, 0.10, 0.10, 0.85, 0.85,  # Component 3
  0.10, 0.10, 0.85, 0.10, 0.85, 0.85,  # Component 4
  0.85, 0.85, 0.85, 0.85, 0.10, 0.10   # Component 5
), nrow = k, byrow = TRUE)


data <- matrix(NA, nrow = n, ncol = m)
X <- matrix(NA, nrow = n, ncol = p)

# Define cluster-specific parameters for covariates
cov_params <- list(
  list(mean = c(0, 3), sd = c(1, 1)),
  list(mean = c(4, -2), sd = c(2, 1.5)),
  list(mean = c(-5, 8), sd = c(0.5, 2)),
  list(mean = c(2, 6), sd = c(1, 2)),
  list(mean = c(-3, 0), sd = c(1, 1.5))
)

prob_X3 <- c(0.1, 0.5, 0.9, 0.3, 0.7)  # Probabilities for X3 for each group
prob_X4 <- c(0.7, 0.1, 0.8, 0.5, 0.3)  # Probabilities for X4 for each group

# Generate the data
for (c in 1:k) {
  # Get the indices for the current cluster
  start_idx <- (c - 1) * n_per_cluster + 1
  end_idx <- c * n_per_cluster
  indices <- start_idx:end_idx
  
  # Sample binary attributes for the current cluster
  for (i in indices) {
    data[i, ] <- rbinom(m, size = 1, prob = prob_matrix[c, ])
  }
  
  # Generate covariates for the current cluster
  X[indices, 1] <- rnorm(n_per_cluster, mean = cov_params[[c]]$mean[1], sd = cov_params[[c]]$sd[1])
  X[indices, 2] <- runif(n_per_cluster, min = cov_params[[c]]$mean[2] - cov_params[[c]]$sd[2], max = cov_params[[c]]$mean[2] + cov_params[[c]]$sd[2])
  X[indices, 3] <- rbinom(n_per_cluster, size = 1, prob = prob_X3[c])
  X[indices, 4] <- rbinom(n_per_cluster, size = 1, prob = prob_X4[c])
  }

# Combine data and X into a single data frame
data <- as.data.frame(data)
X <- as.data.frame(X)

# Define column names for binary attributes and covariates
colnames(data) <- paste0("Y", 1:m)
colnames(X) <- paste0("X", 1:p)

# Combine binary attributes and covariates
data <- cbind(data, X)

#cluster labels
clusters <- factor(rep(1:k, each = n_per_cluster))

#scaling
data[,c(7,8)] <- scale(data[,c(7,8)])
#test set
sampled_test <- sample(nrow(data), 60)
data_test <- data[sampled_test,]
Y_test <- data_test[,c(1,2,3,4,5,6)]
X_test <- data_test[,c(7,8,9,10)]
clusters_test <- clusters[sampled_test]
#training set
data_train <- data[-sampled_test,]
clusters_train <- clusters[-sampled_test]
Y_train <- data_train[,c(1,2,3,4,5,6)]
X_train <- data_train[,c(7,8,9,10)]
sampled_train <- sample(nrow(data_train), 100)
data_sample <- data_train[sampled_train, ]
Y_sample <- data_sample[,c(1,2,3,4,5,6)]
X_sample <- data_sample[,c(7,8,9,10)]
clusters_sample <- clusters_train[sampled_train]

histo_by_group <- ggplot(data.frame(data_train$X2, clusters_train), aes(x = data_train$X2, fill = clusters_train)) +
  geom_histogram(binwidth = 0.08, position = "identity", alpha = 1,color = "black")+
  labs(title = "Plot", x = "X2") +
  scale_fill_manual(values = c("1" = "lightblue",  
                               "2" = "lightgray",  
                               "3" = "darkgray",  
                               "4" = "darkblue",   
                               "5" = "#E0FFFF",
                               "6" = "#4682B4",
                               "7" = "#87CEFA",
                               "8" = "#87CEEB"
                                ))
print(histo_by_group)

library(dplyr)
summary_X3 <- data_train %>%
  mutate(Cluster = clusters_train) %>%       
  group_by(Cluster, X3) %>%
  summarise(Count = n()) %>%
  mutate(Covariate = "X3")

# For X4
summary_X4 <- data_train %>%
  mutate(Cluster = clusters_train) %>%
  group_by(Cluster, X4) %>%
  summarise(Count = n()) %>%
  mutate(Covariate = "X4")

# Combine the summaries
summary_data <- bind_rows(summary_X3, summary_X4)

# Convert X3 and X4 to a factor for correct plotting
summary_data$X3_X4 <- factor(ifelse(is.na(summary_data$X3), summary_data$X4, summary_data$X3), 
                             labels = c("0", "1"))

# Plot the data
ggplot(summary_data, aes(x = Cluster, y = Count, fill = X3_X4)) +
  geom_bar(stat = "identity", position = "dodge") +
  facet_wrap(~Covariate, scales = "free_y") +
  labs(
    title = "Count of 0s and 1s for Covariates X3 and X4 in Each Cluster",
    x = "Clusters",
    y = "Count",
    fill = "Value"
  ) +
  scale_fill_manual(values = c("0" = "#87CEFA",  # Light Blue
                               "1" = "darkblue"), # Steel Blue
                    name = "Values") +
  theme_minimal()

#G1/G2/G3
invcov_cont <- solve(cov(data_train[,c(7,8)]))
espilon_star <- compute_lambda(data_train[,c(7,8,9,10)], nrep = 10000,invcov_cont, var_type)
lambda_true <- .001 / mean(espilon_star[,1])

#PPMX
#continuous covariates
#sigma2 known (prior on mu) (ppmx_n)
c1 <- 1
c2 <- 2
#m0 <- colMeans(data_rob[,c(2,3)])
#S0 <- apply(data_rob[,c(2,3)], 2, var)
#sigma2 unknown (prior on mu and sigma2) (ppmx_t)
a0 <- 1
k0 <- 6
#m <- colMeans(data_rob[,c(2,3)])
#b0 <- apply(data_rob[,c(2,3)], 2, var)
#binary covariates
a <- 0.1
b <- 0.1
#categorical covariates
categs_1 <- NULL
alpha_1 <- rep(0.1, length(categs_1))


ngg_params <- list(sigma = 0.2, k = 0.3)
P0_params <- list(B0 = matrix(1, nrow = 2, ncol = 6))
niter <- 5000
nburn <- 1000
nGibbs <- 0
thinGibbs <- 10
thin <- 1
var_type <- c(1,1,0,0)

g_params <- list(fun = "g1", lambda = 1, alpha = 1, cov = invcov_cont)
g_params <- list("ppmx_n", c1, c2, a, b, categs_1, alpha_1)
g_params <- list("ppmx_t", a0, k0, a, b, categs_1, alpha_1)

system.time(result_0 <- run_mcmc(data=as.matrix(data_sample), type=var_type, g_params=g_params, 
                                 ngg_params=ngg_params, P0_params=P0_params, niter=niter, 
                                 nburn=nburn, thin=thin, nGibbs = nGibbs, thin_Gibbs = thinGibbs))

freq_chain<-t(result_0$freq_chain)[-1,]
matplot(freq_chain, type = "l", xlab = "", ylab = "", main = "Split-Merge(0, 10) - G1", ylim = c(0, 1), col = "black", axes = FALSE)
axis(1)
axis(2, at = seq(0.0, 1.0, by = 0.2), labels = format(seq(0.0, 1.0, by = 0.2), nsmall = 1), las = 1)
box()

labels <- as.factor(result_0$best_clus_binder)
table(labels)
histo_by_group <- ggplot(as.data.frame(data_sample), aes(x = Y, fill = labels)) +
  geom_histogram(binwidth = 1, position = "identity", alpha = 0.7, color = "black") +
  labs(title = "Plot", x = "Y", y = "frequency") +
  scale_fill_discrete(name = "groups")
print(histo_by_group)


#mixture of multivariate normal----
Rcpp::sourceCpp('split_with_g_ngg.cpp')


set.seed(123)
group_1 <- cbind(rmvnorm(n=3775, mean=c(2,2),sigma= diag(.9, 2)), 
                 sample(c(0,1), size = 3775, prob = c(0.9, 0.1), replace = T),
                 sample(c(0,1), size = 3775, prob = c(0.9, 0.1), replace = T))
group_1 <- cbind(apply(group_1, 1, function(x) rnorm(1, 1 + x %*% c(5, 2, 1, 0), .5)),
                 apply(group_1, 1, function(x) rnorm(1, 3+ x %*% c(1, 2, 5, 0), 0.5)), group_1)

group_2<- cbind(rmvnorm(3775, c(-3,-3), diag(.9, 2)), 
                sample(c(0,1), size = 3775, prob = c(0.5, 0.5), replace = T),
                sample(c(0,1), size = 3775, prob = c(0.5, 0.5), replace = T))
group_2 <- cbind(apply(group_2, 1, function(x) rnorm(1, 4 + x %*% c(2, 2, 1, -1), .5)),
                 apply(group_2, 1, function(x) rnorm(1, 5 + x %*% c(-2, 2, 1, -1), .5)), group_2)

group_3 <- cbind(rmvnorm(2510, c(0,0), diag(.9, 2)), 
                 sample(c(0,1), size = 2510, prob = c(0.1, 0.9), replace = T),
                 sample(c(0,1), size = 2510, prob = c(0.1, 0.9), replace = T))
group_3 <- cbind(apply(group_3, 1, function(x) rnorm(1,  1 + x %*% c(-5, -2, -1, 1), .5)),
                 apply(group_3, 1, function(x) rnorm(1, -1 + x %*% c(-2, -2, -1, 1), 0.5)), group_3)

data <- rbind(group_1, group_2, group_3)
new_column_names <- c("Y1","Y2", "X1", "x2", "x3", "x4")
colnames(data) <- new_column_names
#scaling
data[,c(3,4)] <- scale(data[,c(3,4)])
clusters <- factor(rep(1:3, c(3775, 3775, 2510)))
#test set
sampled_test <- sample(nrow(data), 60)
data_test <- data[sampled_test,]
Y_test <- data_test[,c(1,2)]
X_test <- data_test[,c(3,4,5,6)]
clusters_test <- clusters[sampled_test]
#training set
data_train <- data[-sampled_test,]
clusters_train <- clusters[-sampled_test]
Y_train <- data_train[,c(1,2)]
X_train <- data_train[,c(3,4,5,6)]
sampled_train <- sample(nrow(data_train), 100)
data_sample <- data_train[sampled_train, ]
Y_sample <- data_sample[,c(1,2)]
X_sample <- data_sample[,c(3,4,5,6)]
clusters_sample <- clusters_train[sampled_train]


#x1-Y1
scatter_plot <- ggplot(as.data.frame(data_train), aes(x = X1, y = Y1, color = clusters_train)) +
  geom_point(alpha = 0.3) +
  labs(title = "", x = "X1", y = "Y1") +
  scale_color_manual(values = c("1"="#87CEFA","2"= "darkblue","3"= "blue"), name="groups")
print(scatter_plot)

#x2-Y1
scatter_plot <- ggplot(as.data.frame(data_train), aes(x = x2, y = Y1, color = clusters_train)) +
  geom_point(alpha = 0.3) +
  labs(title = "", x = "X2", y = "Y1") +
  scale_color_manual(values = c("1"="#87CEFA","2"= "darkblue","3"= "blue"), name="groups")
print(scatter_plot)

#x3-Y1
scatter_plot <- ggplot(as.data.frame(data_train), aes(x = x3, y = Y1, color = clusters_train)) +
  geom_point(alpha = 0.3) +
  labs(title = "", x = "X3", y = "Y") +
  scale_color_manual(values = c("1"="#87CEFA","2"= "darkblue","3"= "blue"), name="groups")
print(scatter_plot)

#x4-Y1
scatter_plot <- ggplot(as.data.frame(data_train), aes(x = x4, y = Y1, color = clusters_train)) +
  geom_point(alpha = 0.3) +
  labs(title = "", x = "X4", y = "Y") +
  scale_color_manual(values = c("1"="#87CEFA","2"= "darkblue","3"= "blue"), name="groups")
print(scatter_plot)

####
#x1-Y2
scatter_plot <- ggplot(as.data.frame(data_train), aes(x = X1, y = Y2, color = clusters_train)) +
  geom_point(alpha = 0.3) +
  labs(title = "", x = "X1", y = "Y2") +
  scale_color_manual(values = c("1"="#87CEFA","2"= "darkblue","3"= "blue"), name="groups")
print(scatter_plot)

#x2-Y2
scatter_plot <- ggplot(as.data.frame(data_train), aes(x = x2, y = Y2, color = clusters_train)) +
  geom_point(alpha = 0.3) +
  labs(title = "", x = "X2", y = "Y2") +
  scale_color_manual(values = c("1"="#87CEFA","2"= "darkblue","3"= "blue"), name="groups")
print(scatter_plot)

#x3-Y2
scatter_plot <- ggplot(as.data.frame(data_train), aes(x = x3, y = Y2, color = clusters_train)) +
  geom_point(alpha = 0.3) +
  labs(title = "", x = "X3", y = "Y2") +
  scale_color_manual(values = c("1"="#87CEFA","2"= "darkblue","3"= "blue"), name="groups")
print(scatter_plot)

#x4-Y2
scatter_plot <- ggplot(as.data.frame(data_train), aes(x = x4, y = Y2, color = clusters_train)) +
  geom_point(alpha = 0.3) +
  labs(title = "", x = "X4", y = "Y2") +
  scale_color_manual(values =c("1"="#87CEFA","2"= "darkblue","3"= "blue"), name="groups")
print(scatter_plot)

##
data <- rbind(group_1, group_2, group_3)
new_column_names <- c("Y1","Y2", "X1", "x2", "x3", "x4")
colnames(data) <- new_column_names

#scaling
data[,c(3,4)] <- scale(data[,c(3,4)])
clusters <- factor(rep(1:3, c(3775, 3775, 2510)))
#test set
sampled_test <- sample(nrow(data), 60)
data_test <- data[sampled_test,]
Y_test <- data_test[,c(1,2)]
X_test <- data_test[,c(3,4,5,6)]
clusters_test <- clusters[sampled_test]
#training set
data_train <- data[-sampled_test,]
clusters_train <- clusters[-sampled_test]
Y_train <- data_train[,c(1,2)]
X_train <- data_train[,c(3,4,5,6)]
sampled_train <- sample(nrow(data_train), 100)
data_sample <- data_train[sampled_train, ]
Y_sample <- data_sample[,c(1,2)]
X_sample <- data_sample[,c(3,4,5,6)]
clusters_sample <- clusters_train[sampled_train]

ggplot(data.frame(data[,c(1,2)], clusters), aes(x = Y1, y = Y2, color = clusters)) +
  geom_point(size = 2, alpha = 0.3) +               # Scatter plot with different colors for each group
  labs(title = "", x = "Y1", y = "Y2", color = "Original clusters") +
  theme_minimal() +                                 # Minimal theme for a clean look
  scale_color_manual(values = c("1"="#87CEFA","2"= "darkblue","3"= "blue")) + 
  ylim(-10, 20)+
  xlim(-20, 40)+
  theme(legend.position = "top") 

ggplot(data.frame(data_sample[,c(1,2)], clusters_sample), aes(x = Y1, y = Y2, color = clusters_sample)) +
  geom_point(size = 2, alpha = 0.9) +               # Scatter plot with different colors for each group
  labs(title = "", x = "Y1", y = "Y2", color = "Original clusters") +
  theme_minimal() +                                 # Minimal theme for a clean look
  scale_color_manual(values = c("1"="#87CEFA","2"= "darkblue","3"= "blue")) + 
  ylim(-10, 20)+
  xlim(-20, 40)+
  theme(legend.position = "top") 

var_type <- c(1,1,0,0)
#G1/G2/G3
invcov_cont <- solve(cov(data_train[,c(3,4)]))
espilon_star <- compute_lambda(data_train[,c(3,4)], nrep = 1000,invcov_cont, var_type)
lambda_true <- .001 / mean(espilon_star[,1])

#PPMX
#continuous covariates
#sigma2 known (prior on mu) (ppmx_n)
c1 <- 2
c2 <- 1
#m0 <- colMeans(data_rob[,c(2,3)])
#S0 <- apply(data_rob[,c(2,3)], 2, var)
#sigma2 unknown (prior on mu and sigma2) (ppmx_t)
a0 <- 1
k0 <- 4
#m <- colMeans(data_rob[,c(2,3)])
#b0 <- apply(data_rob[,c(2,3)], 2, var)
#binary covariates
a <- 0.1
b <- 0.1
#categorical covariates
categs_1 <- NULL
alpha_1 <- rep(0.1, length(categs_1))


ngg_params <- list(sigma = 0.2, k = 0.3)
mu0 <- colMeans(data_train[,c(1,2)])
S0 <- 3*cov(data_train[,c(1,2)])
P0_params <- list(mu0 = mu0, k0 = 1, S0 = S0, v0 = 2+3)
niter <- 5000
nburn <- 1000
nGibbs <- 0
thinGibbs <- 10
thin <- 1
var_type <- c(1,1,0,0)

g_params <- list(fun = "g0", lambda = 1, alpha = 1, cov = invcov_cont)
g_params <- list("ppmx_n", c1, c2, a, b, categs_1, alpha_1)
g_params <- list("ppmx_t", a0, k0, a, b, categs_1, alpha_1)

system.time(result_0 <- run_mcmc(data=data_sample, type=var_type, g_params=g_params, 
                                 ngg_params=ngg_params, P0_params=P0_params, niter=niter, 
                                 nburn=nburn, thin=thin, nGibbs = nGibbs, thin_Gibbs = thinGibbs))



freq_chain<-t(result_0$freq_chain)[-1,]
matplot(freq_chain, type = "l", xlab = "", ylab = "", main = "Split-Merge(0, 10) - G0", ylim = c(0, 1), col = "black", axes = FALSE)
axis(1)
axis(2, at = seq(0.0, 1.0, by = 0.2), labels = format(seq(0.0, 1.0, by = 0.2), nsmall = 1), las = 1)
box()

labels <- as.factor(result_0$best_clus_binder)
table(labels)

ggplot(data.frame(Y_sample, labels), aes(x = Y1, y = Y2, color = labels)) +
  geom_point(size = 2, alpha = 0.6) +               # Scatter plot with different colors for each group
  labs(title = "Samples from 3 Bivariate Normal Distributions", x = "Y1", y = "Y2") +
  theme_minimal() +                                 # Minimal theme for a clean look
  scale_color_manual(values = c("0" = "blue", "1" = "green", "2" = "red", "3" = "yellow", "4" = "purple")) +
  theme(legend.position = "top") 



#PREDICTION
grid_y1 <- seq(min(Y_train[,1]), max(Y_train[,1]), by = 0.3)
grid_y2 <- seq(min(Y_train[,2]), max(Y_train[,2]), by = 0.2)
niter <- 3000
nburn <- 2000
thin <- 1
thinGibbs <- 1
nGibbs <- 0
g_params <- list(fun = "g0", lambda = 1, alpha = 1, cov = invcov_cont)
g_params <- list("ppmx_n", c1, c2, a, b, categs_1, alpha_1)
g_params <- list("ppmx_t", a0, k0, a, b, categs_1, alpha_1)

system.time(result_pred <- run_mcmc_pred_multi(data_sample, var_type, g_params, ngg_params,
                             P0_params, as.matrix(Y_test), as.matrix(X_test[-c(1:1),]), grid_y1, grid_y2, niter, nburn, thin, nGibbs, thinGibbs,
                             TRUE, FALSE, FALSE, TRUE))
result_pred$


freq_chain<-t(result_pred$freq_chain)[-1,]
matplot(freq_chain, type = "l", xlab = "Iteration", main = "Gibbs sampling (3000)",ylim = c(0, 1), col = "black")

labels <- as.factor(result_pred$best_clus_binder)
table(labels)
Y_pred_test1 <- result_pred$y_pred1
Y_pred_test2 <- result_pred$y_pred2

labels <- as.factor(result_pred$best_clus_binder)
labels[which(labels==3)] <- 1
library(reshape2)
grid_data <- melt(result_pred$pred_grid)
colnames(grid_data) <- c("Y1_index", "Y2_index", "density")
grid_data$Y1 <- grid_y1[grid_data$Y1_index]
grid_data$Y2 <- grid_y2[grid_data$Y2_index]
new_point <- data.frame(Y1 = Y_test[2,1], Y2 = Y_test[2,2])
ggplot() +
  geom_point(data = data.frame(Y_sample, labels), aes(x = Y1, y = Y2, color = labels), size = 2, alpha = 0.5) +
  geom_contour(data = grid_data, aes(x = Y1, y = Y2, z = density), color = "black", bins = 10) +  # Contour lines
  geom_point(data = new_point, aes(x = Y1, y = Y2), color = "purple", size = 4, shape = 17) +  # Add new point
  labs(title = "Predictive distribution with G0", x = "Y1", y = "Y2") +
  theme_minimal() +
  scale_color_manual(values = c("0"="#87CEFA","1" = "darkblue")) +
  theme(legend.position = "top")+
  xlim(-20, 30) +
  ylim(-5, 15)

contour(grid_y1, grid_y2, result_pred$pred_grid, main = "Predictive distribution")


#GESTIONAL AGE----
Rcpp::sourceCpp('split_with_g_ngg.cpp')


data <- read_xlsx("data_gest.xlsx")
premature <- data$premature
data$dde <- log(data$dde)
#scaling
data[,5] <- scale(data[,5])
data[,c(4,5)] <- scale(data[,c(4,5)])
Y <- data$gest
## Plot dataset ##
ggplot(data, aes(x = gest)) +
  geom_histogram(binwidth = 1, fill = "blue", color = "black") +
  labs(title = "Distribuzione di gest", x = "gest", y = "Frequenza")

ggplot(data, aes(x = gest)) +
  geom_histogram(binwidth = 1, fill = "blue", color = "black") +
  labs(title = "Distribuzione di gest suddivisa per smoke", x = "gest", y = "Frequenza") +
  facet_wrap(~ smoke)


histo <- ggplot(data.frame(Y, data$smoke), aes(x = Y, fill = as.factor(data$smoke))) +
  geom_histogram(binwidth = 1, position = "identity", alpha = 0.7,color = "black")+
  labs(title = "Plot", x = "Y", y = "frequence") +
  scale_fill_manual(values = c("red", "green"), name = "groups")
print(histo)

ggplot(data, aes(x = dde, y = gest)) +
  geom_point(color = "red") +
  labs(title = "Relazione tra dde e gest", x = "dde", y = "gest")

ggplot(data, aes(x = weigth, y = gest)) +
  geom_point(color = "green") +
  labs(title = "Relazione tra weight e gest", x = "weight", y = "gest")


data$smoke <- as.factor(data$smoke)
ggplot(data, aes(x = gest, fill = smoke)) +
  geom_histogram(binwidth = 1, color = "black", position = "stack") +
  facet_wrap(~ hosp) +
  labs(title = "Distribuzione di gest per ospedale", x = "Gest", y = "Frequenza") +
  scale_fill_manual(values = c("blue", "red"), name = "Smoke") +
  theme_minimal()


ggplot(data, aes(x = smoke, y = gest)) +
  geom_boxplot(fill = "lightblue", color = "black") +
  labs(title = "Distribuzione di gest per smoke", x = "Smoke", y = "Gest") +
  theme_minimal()

library(GGally)
ggpairs(data, columns = c("gest", "dde", "weigth"))
kruskal.test(gest ~ hosp, data = data)
kruskal.test(gest ~ smoke, data = data)


data <- data[,-6]
data <- data[,-2]
data <- data[,-3]
#PPMX
#continuous covariates
#sigma2 known (prior on mu) (ppmx_n)
c1 <- 1
c2 <- 2
#m0 <- colMeans(data_rob[,c(2,3)])
#S0 <- apply(data_rob[,c(2,3)], 2, var)
#sigma2 unknown (prior on mu and sigma2) (ppmx_t)
a0 <- 1
k0 <- 2
#m <- colMeans(data_rob[,c(2,3)])
#b0 <- apply(data_rob[,c(2,3)], 2, var)
#binary covariates
a <- 0.1
b <- 0.1
#categorical covariates
categs_1 <- unique(data$hosp)
alpha_1 <- rep(0.1, length(categs_1))


#REGRESSION
invcov_cont <- solve(cov(data[,c(3,4)]))
ngg_params <- list(sigma = 0.2, k = 0.3)

beta0 <- numeric(4)
B0 <- diag(4)/0.01
P0_params <- list(beta0 = beta0, B0 = B0, a0 = 2, b0 = 1)


niter <- 1000
nburn <- 500
nGibbs <- 0
thinGibbs <- 10
thin <- 1

data_smoke <- data[,c(1,3)]
data_dde <- data[,c(1,3,4,5)]
var_type <- c(0,1,1)
Y_test <- 0
X_test <-c(0,3.2,75)

grid <- seq(min(Y), max(Y), by = 0.2)

g_params <- list(fun = "g0",lambda = 0.5, alpha = 1, cov = invcov_cont)
g_params <- list("ppmx_n", c1, c2, a, b, categs_1, alpha_1)
g_params <- list("ppmx_t", a0, k0, a, b, categs_1, alpha_1)

result_pred <- run_mcmc_pred(as.matrix(data_dde), var_type, g_params, ngg_params,
                             P0_params, Y_test, t(as.matrix(X_test)), grid, niter, nburn, thin, nGibbs, thinGibbs,
                             TRUE, FALSE, FALSE, TRUE, TRUE)

freq_chain<-t(result_pred$freq_chain)[-1,]
matplot(freq_chain, type = "l", xlab = "Iteration", main = "Split & Merge (3000, 0, 1)",ylim = c(0, 1))

labels <- as.factor(result_pred$best_clus_binder)
table(labels)
histo_by_group <- ggplot(as.data.frame(Y), aes(x = Y, fill = labels)) +
  geom_histogram(binwidth = 1, position = "identity", alpha = 0.7, color = "black") +
  labs(title = "Plot", x = "Y", y = "frequency") +
  scale_fill_discrete(name = "groups")
print(histo_by_group)

ggplot(data.frame(x = grid, y = c(colMeans(result_pred$pred_grid)), 
                  ylow = apply(result_pred$pred_grid, 2, quantile, p = 0.05), 
                  yup = apply(result_pred$pred_grid, 2, quantile, p = 0.95))) +
  theme_bw() +
  geom_line(aes(x = x, y = y)) + 
  xlab("y") +
  ylab("density") +
  ggtitle("Predictive distribution") + 
  geom_vline(aes(xintercept = Y_test[1]), lty = 2, col = 2) +
  geom_ribbon(aes(x = x, ymin = ylow, ymax = yup), alpha = 0.25) +
  ylim(c(0, 0.5))

#NO REGRESSION
invcov_cont <- solve(cov(data[,c(3,4)]))
invcov_cont <- matrix(1/var(data[,3]))
ngg_params <- list(sigma = 0.2, k = 0.3)
P0_params <- list(mu0 = mean(data$gest), lambda0 = 0.01, a0 = 1, b0 = 5)

niter <- 2000
nburn <- 1000
nGibbs <- 0
thinGibbs <- 10
thin <- 1
var_type <- c(0,1,1)

grid <- seq(min(data$gest), max(data$gest), by = 0.1)

g_params <- list(fun = "g1", lambda = 1, alpha = 1, cov = invcov_cont)
g_params <- list("ppmx_n", c1, c2, a, b, categs_1, alpha_1)
g_params <- list("ppmx_t", a0, k0, a, b, categs_1, alpha_1)

Y_test <- 0
X_test <-c(0, -2, 2)
result_pred <- run_mcmc_pred(as.matrix(data[sample(1:nrow(data),1000),]), var_type, g_params, ngg_params,
                             P0_params, Y_test, matrix(X_test, nrow = 1), grid, niter, nburn, thin, nGibbs, thinGibbs,
                             TRUE, FALSE, FALSE, TRUE)

freq_chain<-t(result_pred$freq_chain)[-1,]
matplot(freq_chain, type = "l", xlab = "Iteration", main = "Split & Merge (3000, 0, 1)",ylim = c(0, 1))

labels <- as.factor(result_pred$best_clus_binder)
table(labels)

### plot :
labels <- as.factor(result_pred$best_clus_binder)
table(labels)

data$cluster <- labels

histo_by_group <- ggplot(as.data.frame(data), aes(x = gest, fill = labels)) +
  geom_histogram(binwidth = 1, position = "identity", alpha = 0.7, color = "black") +
  labs(title = "Plot", x = "Y", y = "frequency") +
  scale_fill_discrete(name = "groups")
print(histo_by_group)

freq_chain<-t(result_pred$freq_chain)[-1,]
matplot(freq_chain, type = "l", xlab = "Iteration", main = " Gibbs sampler: iter=5000",ylim = c(0, 1))

ggplot(data, aes(x = dde, y = gest, color = cluster)) +
  geom_point(size = 2) +
  labs(title = "Distribuzione dei Cluster", x = "log DDE", y = "gest") +
  theme_minimal()

ggplot(data, aes(x = weigth, y = gest, color = cluster)) +
  geom_point(size=2) +
  labs(title = "Relazione tra weight e gest", x = "weight", y = "gest") +
  theme_minimal()


ggplot(data, aes(x = as.factor(cluster), y = gest, fill = as.factor(cluster))) +
  geom_boxplot(color = "black") +
  labs(title = "Distribuzione di gest per Cluster", x = "Cluster", y = "Gest") +
  scale_fill_manual(values = c("red", "blue", "green", "purple"), name = "Cluster") +
  theme_minimal()

library(GGally)
ggpairs(data, columns = c("gest", "dde", "weigth"), aes(color = as.factor(cluster)))

ggplot(data, aes(x = smoke, y = gest, fill = cluster)) +
  geom_boxplot(color = "black") +
  labs(title = "Distribuzione di gest per smoke", x = "Smoke", y = "Gest") +
  scale_fill_manual(values = c("red", "blue", "green", "purple"), name = "Cluster") +
  theme_minimal()

table(data$smoke[which(labels==2)])
table(data$smoke[which(labels==1)])
table(data$smoke[which(labels==0)])
table(data$smoke[which(labels==3)])

histo_by_group <- ggplot(as.data.frame(data), aes(x = smoke,  color = cluster)) +
  geom_histogram(binwidth = 1, position = "identity", alpha = 0.7) +
  labs(title = "Plot", x = "smoke", y = "frequency") +
  scale_fill_discrete(name = "groups")
print(histo_by_group)

histo_by_group <- ggplot(as.data.frame(data), aes(x = as.factor(smoke), fill = as.factor(cluster))) +
  geom_histogram(stat = "count", position = "dodge", alpha = 0.8, color = "black") +
  labs(title = "Distribuzione dei Fumatori nei Cluster", x = "Fumatore", y = "Frequenza") +
  scale_fill_brewer(palette = "Set2", name = "Cluster") +  
  geom_text(stat = "count", aes(label = ..count..), vjust = -0.5, position = position_dodge(0.9)) +  
  theme_minimal(base_size = 15) + 
  theme(legend.position = "top") 
print(histo_by_group)


#PREDICTION
grid <- seq(min(Y), max(Y), by = 0.1)
ngg_params <- list(sigma = 0.2, k = 0.3)
P0_params <- list(mu0 = mean(Y), lambda0 = 0.01, a0 = 1, b0 = 5)
niter <- 2000
nburn <- 1500
thin <- 1
nGibbs <- 0
thinGibbs <- 10
var_type <- c(0,1,1)

g_params <- list(fun = "g1", lambda = 1.5, alpha = 1, cov = invcov_cont)
g_params <- list("ppmx_n", c1, c2, a, b, categs_1, alpha_1)
g_params <- list("ppmx_t", a0, k0, a, b, categs_1, alpha_1)


Y_test <- 0
X_test <-c(0, 2.8, 70)
X_test <-c(0, 3.2, 70)
X_test <-c(0, 3.6, 70)
X_test <-c(0, 3.2, 58)
X_test <-c(0, 3.2, 81)
X_test <-c(1, 3.2, 70)

X_test <-c(0, 2.8, 81)
X_test <-c(1, 3.6, 58)


X_test <-c(0, 2.8, 70)
result_0_28_70 <- run_mcmc_pred(as.matrix(data), var_type, g_params, ngg_params,
                             P0_params, Y_test, matrix(X_test,1,3,byrow = TRUE), grid, niter, nburn, thin, nGibbs, thinGibbs,
                             TRUE, FALSE, FALSE, TRUE)
X_test <-c(0, 3.2, 70)
result_0_32_70 <- run_mcmc_pred(as.matrix(data), var_type, g_params, ngg_params,
                                   P0_params, Y_test, matrix(X_test,1,3,byrow=TRUE), grid, niter, nburn, thin, nGibbs, thinGibbs,
                                   TRUE, FALSE, FALSE, TRUE)
X_test <-c(0, 3.6, 70)
result_0_36_70 <- run_mcmc_pred(as.matrix(data), var_type, g_params, ngg_params,
                                    P0_params, Y_test, matrix(X_test,1,3,byrow=TRUE), grid, niter, nburn, thin, nGibbs, thinGibbs,
                                    TRUE, FALSE, FALSE, TRUE)

X_test <-c(0, 3.2, 58)
result_0_32_58 <- run_mcmc_pred(as.matrix(data), var_type, g_params, ngg_params,
                                P0_params, Y_test, matrix(X_test,1,3,byrow = TRUE), grid, niter, nburn, thin, nGibbs, thinGibbs,
                                TRUE, FALSE, FALSE, TRUE)
X_test <-c(0, 3.2, 81)
result_0_32_81 <- run_mcmc_pred(as.matrix(data), var_type, g_params, ngg_params,
                                P0_params, Y_test, matrix(X_test,1,3,byrow=TRUE), grid, niter, nburn, thin, nGibbs, thinGibbs,
                                TRUE, FALSE, FALSE, TRUE)

X_test <-c(1, 3.2, 70)
result_1_32_70 <- run_mcmc_pred(as.matrix(data), var_type, g_params, ngg_params,
                                P0_params, Y_test, matrix(X_test,1,3,byrow=TRUE), grid, niter, nburn, thin, nGibbs, thinGibbs,
                                TRUE, FALSE, FALSE, TRUE)


X_test <-c(1, 3.6, 58)
result_1_36_58 <- run_mcmc_pred(as.matrix(data), var_type, g_params, ngg_params,
                                P0_params, Y_test, matrix(X_test,1,3,byrow=TRUE), grid, niter, nburn, thin, nGibbs, thinGibbs,
                                TRUE, FALSE, FALSE, TRUE)
X_test <-c(0, 2.8, 81)
result_0_28_81 <- run_mcmc_pred(as.matrix(data), var_type, g_params, ngg_params,
                                P0_params, Y_test, matrix(X_test,1,3,byrow=TRUE), grid, niter, nburn, thin, nGibbs, thinGibbs,
                                TRUE, FALSE, FALSE, TRUE)



data_xy <- data.frame(
  x = grid,
  y_0_28_70 = colMeans(result_0_28_70$pred_grid),
  y_0_32_70 = colMeans(result_0_32_70$pred_grid),
  y_0_36_70 = colMeans(result_0_36_70$pred_grid),
  y_0_32_58 = colMeans(result_0_32_58$pred_grid),
  y_0_32_81 = colMeans(result_0_32_81$pred_grid),
  y_1_32_70 = colMeans(result_1_32_70$pred_grid),
  y_1_36_58 = colMeans(result_1_36_58$pred_grid),
  y_0_28_81 = colMeans(result_0_28_81$pred_grid)
)

# Plot with both color and linetype mapped to the legend
library(ggplot2)

ggplot(data_xy, aes(x = x)) +
  theme_bw() +
  # Define each line with unique linetype, size, and color, using a combined label in aes
  geom_line(aes(y = y_0_28_70, linetype = "(0, 2.8, 70)", size = "(0, 2.8, 70)", color = "(0, 2.8, 70)")) +
  geom_line(aes(y = y_0_32_70, linetype = "(0, 3.2, 70)", size = "(0, 3.2, 70)", color = "(0, 3.2, 70)")) +
  geom_line(aes(y = y_0_36_70, linetype = "(0, 3.6, 70)", size = "(0, 3.6, 70)", color = "(0, 3.6, 70)")) +
  geom_line(aes(y = y_0_32_58, linetype = "(0, 3.2, 58)", size = "(0, 3.2, 58)", color = "(0, 3.2, 58)")) +
  geom_line(aes(y = y_0_32_81, linetype = "(0, 3.2, 81)", size = "(0, 3.2, 81)", color = "(0, 3.2, 81)")) +
  geom_line(aes(y = y_1_32_70, linetype = "(1, 3.2, 70)", size = "(1, 3.2, 70)", color = "(1, 3.2, 70)")) +
  geom_line(aes(y = y_1_36_58, linetype = "(1, 3.6, 58)", size = "(1, 3.6, 58)", color = "(1, 3.6, 58)")) +
  geom_line(aes(y = y_0_28_81, linetype = "(0, 2.8, 81)", size = "(0, 2.8, 81)", color = "(0, 2.8, 81)")) +
  xlab("Gestational age (weeks)") +
  ylab("Density") +
  ggtitle("Predictive Distribution") +
  ylim(c(0, 0.28)) +
  # Define custom linetypes, sizes, and colors with simplified labels
  scale_linetype_manual(
    values = c("(0, 2.8, 70)" = "solid",
               "(0, 3.2, 70)" = "dashed",
               "(0, 3.6, 70)" = "dotted",
               "(0, 3.2, 58)" = "dotdash",
               "(0, 3.2, 81)" = "longdash",
               "(1, 3.2, 70)" = "twodash",
               "(1, 3.6, 58)" = "solid",
               "(0, 2.8, 81)" = "dashed")) +
  scale_size_manual(
    values = c("(0, 2.8, 70)" = 0.7,
               "(0, 3.2, 70)" = 0.7,
               "(0, 3.6, 70)" = 0.7,
               "(0, 3.2, 58)" = 0.7,
               "(0, 3.2, 81)" = 0.7,
               "(1, 3.2, 70)" = 0.7,
               "(1, 3.6, 58)" = 0.7,
               "(0, 2.8, 81)" = 0.7)) +
  scale_color_manual(
    values = c("(0, 2.8, 70)" = "black",
               "(0, 3.2, 70)" = "black",
               "(0, 3.6, 70)" = "black",
               "(0, 3.2, 58)" = "blue",
               "(0, 3.2, 81)" = "blue",
               "(1, 3.2, 70)" = "green",
               "(1, 3.6, 58)" = "red",
               "(0, 2.8, 81)" = "red")) +
  # Guide to combine linetype, size, and color legends into one
  guides(
    linetype = guide_legend(title = ""),
    size = guide_legend(title = ""),
    color = guide_legend(title = "")
  )


###

result_pred <- run_mcmc_pred(as.matrix(data_smoke), var_type, g_params, ngg_params,
                             P0_params, Y_test, as.matrix(X_test), grid, niter, nburn, thin, nGibbs, thinGibbs,
                             TRUE, FALSE, FALSE, TRUE, TRUE)

freq_chain<-t(result_pred$freq_chain)[-1,]
matplot(freq_chain, type = "l", xlab = "Iteration", main = "Split & Merge (3000, 0, 1)",ylim = c(0, 1))
labels <- as.factor(result_pred$best_clus_binder)
table(labels)

ggplot(data.frame(x = grid, y = c(colMeans(result_pred$pred_grid)), 
                  ylow = apply(result_pred$pred_grid, 2, quantile, p = 0.05), 
                  yup = apply(result_pred$pred_grid, 2, quantile, p = 0.95))) +
  theme_bw() +
  geom_line(aes(x = x, y = y)) + 
  xlab("y") +
  ylab("density") +
  ggtitle("Predictive distribution") + 
  geom_vline(aes(xintercept = Y_test[1]), lty = 2, col = 2) +
  geom_ribbon(aes(x = x, ymin = ylow, ymax = yup), alpha = 0.25) +
  ylim(c(0, 1))

##MULTIVARIATE
##SEPARATELY----
Rcpp::sourceCpp('split_with_g_ngg.cpp')
data <- read_xlsx("data_gest.xlsx")
premature <- data$premature
data$dde <- log(data$dde)
#scaling
data[,5] <- scale(data[,5])
data<-data[,-6]
data<-cbind(data$dde,data)
data_hosp<-data[,-5]
data<-data_hosp[,-3]
colnames(data)[1]<-"dde"
colnames(data_hosp)[1]<-"dde"

invcov_cont <- matrix(1/var(data[,4]))
espilon_star <- compute_lambda(data[,c(3,4)], nrep = 1000,invcov_cont, var_type)
lambda_true <- .001 / mean(espilon_star[,1])

#PPMX
#continuous covariates
#sigma2 known (prior on mu) (ppmx_n)
c1 <- 1
c2 <- 2
#m0 <- colMeans(data_rob[,c(2,3)])
#S0 <- apply(data_rob[,c(2,3)], 2, var)
#sigma2 unknown (prior on mu and sigma2) (ppmx_t)
a0 <- 1
k0 <- 2
#m <- colMeans(data_rob[,c(2,3)])
#b0 <- apply(data_rob[,c(2,3)], 2, var)
#binary covariates
a <- 0.1
b <- 0.1
#categorical covariates
categs_1 <- unique(data_hosp$hosp)
alpha_1 <- rep(0.1, length(categs_1))


smokers <- data[which(data$smoke==1),c(1,2,4)]
Y1 <- smokers[,c(1,2)]
non_smokers <- data[which(data$smoke==0),c(1,2,4)]
Y0<- non_smokers[,c(1,2)]

ngg_params <- list(sigma = 0.2, k = 0.3)
mu0 = colMeans((non_smokers[,c(1,2)])) #c(0,0)
S0 = 3*cov(as.matrix(non_smokers[,c(1,2)])) #diag(diag(cov(as.matrix(data[,c(1,2)]))))
P0_params <- list(mu0 = mu0, k0 = 1/10, S0 = S0, v0 = 2+3)
niter <- 3000
nburn <- 2000
nGibbs <- 0
thinGibbs <- 1
thin <- 1
var_type <- c(1)

g_params <- list(fun = "g3", lambda = 0.08, alpha = 1, cov = invcov_cont)
g_params <- list("ppmx_n", c1, c2, a, b, categs_1, alpha_1)
g_params <- list("ppmx_t", a0, k0, a, b, categs_1, alpha_1)

g_params <- list(fun = "g0", lambda = 1.5, alpha = 1, cov = invcov_cont)
nGibbs <- 0
thinGibbs <- 1
system.time(result_0_0_1 <- run_mcmc(data=as.matrix(non_smokers), type=var_type, g_params=g_params, 
                                 ngg_params=ngg_params, P0_params=P0_params, niter=niter, 
                                 nburn=nburn, thin=thin, nGibbs = nGibbs, thin_Gibbs = thinGibbs))
g_params <- list(fun = "g0", lambda = 1, alpha = 1, cov = invcov_cont)
nGibbs <- 1
thinGibbs <- 10
system.time(result_0_1_10 <- run_mcmc(data=as.matrix(non_smokers), type=var_type, g_params=g_params, 
                                 ngg_params=ngg_params, P0_params=P0_params, niter=niter, 
                                 nburn=nburn, thin=thin, nGibbs = nGibbs, thin_Gibbs = thinGibbs))
g_params <- list(fun = "g0", lambda = 1.5, alpha = 1, cov = invcov_cont)
nGibbs <- 0
thinGibbs <- 10
system.time(result_0_0_10 <- run_mcmc(data=as.matrix(non_smokers), type=var_type, g_params=g_params, 
                                 ngg_params=ngg_params, P0_params=P0_params, niter=niter, 
                                 nburn=nburn, thin=thin, nGibbs = nGibbs, thin_Gibbs = thinGibbs))

g_params <- list(fun = "g1", lambda = 1, alpha = 1, cov = invcov_cont)
nGibbs <- 0
thinGibbs <- 1
system.time(result_1_0_1 <- run_mcmc(data=as.matrix(non_smokers), type=var_type, g_params=g_params, 
                                     ngg_params=ngg_params, P0_params=P0_params, niter=niter, 
                                     nburn=nburn, thin=thin, nGibbs = nGibbs, thin_Gibbs = thinGibbs))
g_params <- list(fun = "g1", lambda = 1, alpha = 1, cov = invcov_cont)
nGibbs <- 1
thinGibbs <- 10
system.time(result_1_1_10 <- run_mcmc(data=as.matrix(non_smokers), type=var_type, g_params=g_params, 
                                      ngg_params=ngg_params, P0_params=P0_params, niter=niter, 
                                      nburn=nburn, thin=thin, nGibbs = nGibbs, thin_Gibbs = thinGibbs))
g_params <- list(fun = "g1", lambda = 1.5, alpha = 1, cov = invcov_cont)
nGibbs <- 0
thinGibbs <- 10
system.time(result_1_0_10 <- run_mcmc(data=as.matrix(non_smokers), type=var_type, g_params=g_params, 
                                      ngg_params=ngg_params, P0_params=P0_params, niter=niter, 
                                      nburn=nburn, thin=thin, nGibbs = nGibbs, thin_Gibbs = thinGibbs))
time <- data.frame(
  "time G0" = c(156.16, 56.99, 46.19),  # First column of values
  "time G1" = c(691.70, 166.88, 133.41),  # Second column of values
  row.names = c("Gibbs sampler (3000)", "Split & Merge (3000, 1, 10)", "Split & Merge (3000, 0, 10)")
)
colnames(time)<- c("time G0", "time G1")

freq_chain<-t(result_smoke_11$freq_chain)[-1,]
matplot(freq_chain, type = "l", xlab = "Iteration", main = "Non smokers, Gibbs sampler (3000)",ylim = c(0, 1), )

labels <- as.factor(result_smoke_11$best_clus_binder)
table(labels)


ggplot(data.frame(Y1, labels), aes(x = dde, y = gest, color = labels)) +
  geom_point(size = 2, alpha = 0.6) +               # Scatter plot with different colors for each group
  labs( x = "log(DDE)", y = "gestional time") +
  theme_minimal() +                                 # Minimal theme for a clean look
  scale_color_manual(values = c("0" = "blue", "1" = "green", "2" = "red", "3" = "yellow", "4" = "violet", "5" = "black", "6" = "lightblue", "7" = "orange")) +
  theme(legend.position = "top") 

##analysis on found clusters

labels <- as.factor(result_smoke_11$best_clus_binder)
table(labels)

data_res <- cbind(smokers,labels)

label_counts <- table(labels)
labels_to_keep <- names(label_counts[label_counts > 10])
filtered_labels <- labels[labels %in% labels_to_keep]
filt_data <- data_res[labels %in% labels_to_keep, ]


ggplot(as.data.frame(filt_data), aes(x = gest, fill = labels)) +
  geom_histogram(binwidth = 0.4, position = "identity", alpha = 0.6, color = "black") +
  labs(title = "smokers", x = "gestional age (weeks)", y = "frequency") +
  scale_color_manual(values = c("0" = "#ADD8E6",  # Light Blue
                                "1" = "#90EE90",  # Light Green
                                "2" = "#FFB6C1",  # Light Red (pinkish)
                                "3" = "#FFFFE0",  # Light Yellow
                                "4" = "#EE82EE",  # Light Violet
                                "5" = "#FFDAB9",  # Light Orange (peach puff)
                                "6" = "#F08080",   # Light Coral
                                "7" = "#E0FFFF",   # Light Cyan
                                "8" = "#DDA0DD"   # Light Purple (plum)
  ))+
  scale_y_continuous(limits = c(0, 105))+
  theme(legend.position = "none")

ggplot(as.data.frame(filt_data), aes(x = dde, fill = labels)) +
  geom_histogram(binwidth = 0.1, position = "identity", alpha = 0.7, color = "black") +
  labs(title = "non smokers", x = "logDDE (μg/l)", y = "frequency") +
  scale_color_manual(values = c("0" = "#ADD8E6",  # Light Blue
                                "1" = "#90EE90",  # Light Green
                                "2" = "#FFB6C1",  # Light Red (pinkish)
                                "3" = "#FFFFE0",  # Light Yellow
                                "4" = "#EE82EE",  # Light Violet
                                "5" = "#FFDAB9",  # Light Orange (peach puff)
                                "6" = "#F08080",   # Light Coral
                                "7" = "#E0FFFF",   # Light Cyan
                                "8" = "#DDA0DD"   # Light Purple (plum)
  ))+
  scale_y_continuous(limits = c(0, 50))+
  theme(legend.position = "none")

ggplot(filt_data, aes(x = dde, y = gest, color = labels)) +
  geom_point(size = 2) +
  labs(title = "non smokers", x = "logDDE (μg/l)", y = "gestional age (weeks)") +
  scale_color_manual(values = c("0" = "#ADD8E6",  # Light Blue
                                "1" = "#90EE90",  # Light Green
                                "2" = "#FFB6C1",  # Light Red (pinkish)
                                "3" = "#FFFFE0",  # Light Yellow
                                "4" = "#EE82EE",  # Light Violet
                                "5" = "#FFDAB9",  # Light Orange (peach puff)
                                "6" = "#F08080",   # Light Coral
                                "7" = "#E0FFFF",   # Light Cyan
                                "8" = "#DDA0DD"   # Light Purple (plum)
  ))+
  scale_y_continuous(limits = c(27, 46))+
  theme(legend.position = "none")

ggplot(filt_data, aes(x = weigth, y = gest, color = labels)) +
  geom_point(size=2) +
  labs( title = "non smokers",x = "weight (Kg)", y = "gestional age (weeks)") +
  scale_color_manual(values = c("0" = "#ADD8E6",  # Light Blue
                                "1" = "#90EE90",  # Light Green
                                "2" = "#FFB6C1",  # Light Red (pinkish)
                                "3" = "#FFFFE0",  # Light Yellow
                                "4" = "#EE82EE",  # Light Violet
                                "5" = "#FFDAB9",  # Light Orange (peach puff)
                                "6" = "#F08080",   # Light Coral
                                "7" = "#E0FFFF",   # Light Cyan
                                "8" = "#DDA0DD"   # Light Purple (plum)
  ))+
  scale_y_continuous(limits = c(27, 46))+
  theme(legend.position = "none")

ggplot(filt_data, aes(x = weigth, y = dde, color = labels)) +
  geom_point(size=2) +
  labs(title = "non smokers",x = "weight (Kg)", y = "logDDE (μg/l)") +
  scale_color_manual(values = c("0" = "#ADD8E6",  # Light Blue
                                "1" = "#90EE90",  # Light Green
                                "2" = "#FFB6C1",  # Light Red (pinkish)
                                "3" = "#FFFFE0",  # Light Yellow
                                "4" = "#EE82EE",  # Light Violet
                                "5" = "#FFDAB9",  # Light Orange (peach puff)
                                "6" = "#F08080",   # Light Coral
                                "7" = "#E0FFFF",   # Light Cyan
                                "8" = "#DDA0DD"   # Light Purple (plum)
  ))+
  scale_y_continuous(limits = c(1, 6))+
  theme(legend.position = "none")

ggplot(filt_data, aes(x = gest, fill = labels)) +
  geom_histogram(binwidth = 1, color = "black") +
  facet_wrap(~ hosp) +
  labs(title = "Distribuzione di gest per ospedale", x = "Gest", y = "Frequenza") +
  scale_color_manual(values = c("0" = "#ADD8E6",  # Light Blue
                                "1" = "#90EE90",  # Light Green
                                "2" = "#FFB6C1",  # Light Red (pinkish)
                                "3" = "#FFFFE0",  # Light Yellow
                                "4" = "#EE82EE",  # Light Violet
                                "5" = "#FFDAB9",  # Light Orange (peach puff)
                                "6" = "#F08080",   # Light Coral
                                "7" = "#E0FFFF",   # Light Cyan
                                "8" = "#DDA0DD"   # Light Purple (plum)
  ))+
  theme(legend.position = "none")

ggplot(filt_data, aes(x = as.factor(labels), y = gest, fill = as.factor(labels))) +
  geom_boxplot(color = "black", alpha = 0.5) +
  labs(title, "non smokers",x = "Cluster", y = "Gestional age (weeks)") +
  scale_color_manual(values = c("0" = "#ADD8E6",  # Light Blue
                                "1" = "#90EE90",  # Light Green
                                "2" = "#FFB6C1",  # Light Red (pinkish)
                                "3" = "#FFFFE0",  # Light Yellow
                                "4" = "#EE82EE",  # Light Violet
                                "5" = "#FFDAB9",  # Light Orange (peach puff)
                                "6" = "#F08080",   # Light Coral
                                "7" = "#E0FFFF",   # Light Cyan
                                "8" = "#DDA0DD"   # Light Purple (plum)
  ))+
  scale_y_continuous(limits = c(27, 45))+
  theme(legend.position = "none")

ggplot(filt_data, aes(x = as.factor(labels), y = dde, fill = as.factor(labels))) +
  geom_boxplot(color = "black", alpha = 0.5) +
  labs(title= "non smokers",x = "Cluster", y = "logDDE (μg/l)") +
  scale_color_manual(values = c("0" = "#ADD8E6",  # Light Blue
                                "1" = "#90EE90",  # Light Green
                                "2" = "#FFB6C1",  # Light Red (pinkish)
                                "3" = "#FFFFE0",  # Light Yellow
                                "4" = "#EE82EE",  # Light Violet
                                "5" = "#FFDAB9",  # Light Orange (peach puff)
                                "6" = "#F08080",   # Light Coral
                                "7" = "#E0FFFF",   # Light Cyan
                                "8" = "#DDA0DD"   # Light Purple (plum)
  ))+
  scale_y_continuous(limits = c(0, 5.5))+
  theme(legend.position = "none")

ggplot(filt_data, aes(x = as.factor(labels), y = weigth, fill = as.factor(labels))) +
  geom_boxplot(color = "black", alpha = 0.5) +
  labs(title = "non smokers",x = "Cluster", y = "weight (Kg)") +
  scale_color_manual(values = c("0" = "#ADD8E6",  # Light Blue
                                "1" = "#90EE90",  # Light Green
                                "2" = "#FFB6C1",  # Light Red (pinkish)
                                "3" = "#FFFFE0",  # Light Yellow
                                "4" = "#EE82EE",  # Light Violet
                                "5" = "#FFDAB9",  # Light Orange (peach puff)
                                "6" = "#F08080",   # Light Coral
                                "7" = "#E0FFFF",   # Light Cyan
                                "8" = "#DDA0DD"   # Light Purple (plum)
  ))+
  scale_y_continuous(limits = c(2, 137))+
  theme(legend.position = "none")

library(GGally)
ggpairs(filt_data, columns = c("gest", "dde", "weigth"), aes(color = as.factor(labels)))


table(data$smoke[which(labels==2)])
table(data_sample$smoke[which(labels==1)])
table(data_sample$smoke[which(labels==0)])
table(data$smoke[which(labels==3)])

#
library(dplyr)
df <- data.frame(filt_data$smoke, filt_data$labels)
colnames(df) <- c("Smoke", "groups")
group_counts <- df %>%
  group_by(groups, Smoke) %>%
  summarise(count = n(), .groups = 'drop')

# Print the summary counts
print(group_counts)

# Plot the results using ggplot2
ggplot(group_counts, aes(x = groups, y = count, fill = as.factor(Smoke))) +
  geom_bar(stat = "identity", position = "dodge", alpha = 0.6) +  # Dodge for side-by-side bars
  labs(x = "Group", y = "Count", fill = "Smoke") +
  theme_minimal()
#

histo_by_group <- ggplot(as.data.frame(data_sample), aes(x = as.factor(smoke), fill = as.factor(labels))) +
  geom_histogram(stat = "count", position = "dodge", alpha = 0.8, color = "black") +
  labs(title = "Distribuzione dei Fumatori nei Cluster", x = "Fumatore", y = "Frequenza") +
  scale_fill_brewer(palette = "Set2", name = "Cluster") +  
  geom_text(stat = "count", aes(label = ..count..), vjust = -0.5, position = position_dodge(0.9)) +  
  theme_minimal(base_size = 15) + 
  theme(legend.position = "top") 
print(histo_by_group)

#prediction

grid_y1 <- seq(min(data$dde), max(data$dde), by = 0.03)
grid_y2 <- seq(min(data$gest), 47, by = 0.1)
ngg_params <- list(sigma = 0.2, k = 0.3)
mu0 = colMeans(smokers[,c(1,2)]) #c(0,0)
S0 = 3*cov(as.matrix(smokers[,c(1,2)])) #diag(diag(cov(as.matrix(data[,c(1,2)]))))
P0_params <- list(mu0 = mu0, k0 = 1/10, S0 = S0, v0 = 2+3)

niter <- 2000
nburn <- 1500
thin <- 1
nGibbs <- 0
thinGibbs <- 10
var_type <- c(1)

g_params <- list(fun = "g1", lambda = 1, alpha = 1, cov = invcov_cont)
g_params <- list("ppmx_n", c1, c2, a, b, categs_1, alpha_1)
g_params <- list("ppmx_t", a0, k0, a, b, categs_1, alpha_1)

####
Xnew <- matrix(c(70,58,81), 3, 1, byrow = TRUE)
nGibbs <- 0
thinGibbs <- 1

g_params <- list(fun = "g0", lambda = 1, alpha = 1, cov = invcov_cont)
system.time(result_smoke_0 <- run_mcmc_pred_multi(as.matrix(smokers), var_type, g_params, ngg_params,
                                                  P0_params, matrix(c(0), nrow=1), Xnew,
                                                  grid_y1, grid_y2, niter, nburn, thin, nGibbs, thinGibbs,
                                                  TRUE, FALSE, FALSE, TRUE))


g_params <- list(fun = "g1", lambda = 1, alpha = 1, cov = invcov_cont)
system.time(result_smoke_11 <- run_mcmc_pred_multi(as.matrix(smokers), var_type, g_params, ngg_params,
                                                  P0_params, matrix(c(0), nrow=1), Xnew,
                                                  grid_y1, grid_y2, niter, nburn, thin, nGibbs, thinGibbs,
                                                  TRUE, FALSE, FALSE, TRUE))


mu0 = colMeans(non_smokers[,c(1,2)]) #c(0,0)
S0 = 3*cov(as.matrix(non_smokers[,c(1,2)])) #diag(diag(cov(as.matrix(data[,c(1,2)]))))
P0_params <- list(mu0 = mu0, k0 = 1/10, S0 = S0, v0 = 2+3)

g_params <- list(fun = "g0", lambda = 1, alpha = 1, cov = invcov_cont)
system.time(result_nosmoke_0 <- run_mcmc_pred_multi(as.matrix(non_smokers), var_type, g_params, ngg_params,
                                                  P0_params, matrix(c(0), nrow=1), Xnew,
                                                  grid_y1, grid_y2, niter, nburn, thin, nGibbs, thinGibbs,
                                                  TRUE, FALSE, FALSE, TRUE))


g_params <- list(fun = "g1", lambda = 1, alpha = 1, cov = invcov_cont)
system.time(result_nosmoke_11 <- run_mcmc_pred_multi(as.matrix(non_smokers), var_type, g_params, ngg_params,
                                                   P0_params, matrix(c(0), nrow=1), Xnew,
                                                   grid_y1, grid_y2, niter, nburn, thin, nGibbs, thinGibbs,
                                                   TRUE, FALSE, FALSE, TRUE))


####

freq_chain<-t(result_pred_11$freq_chain)[-1,]
matplot(freq_chain, type = "l", xlab = "Iteration", main = "Gibbs sampler (2000) with G0",ylim = c(0, 1))

labels <- as.factor(result_nosmoke_0$best_clus_binder)
table(labels)

data_res <- cbind(non_smokers,labels)
label_counts <- table(labels)
labels_to_keep <- names(label_counts[label_counts > 1])
filtered_labels <- labels[labels %in% labels_to_keep]
filt_data <- data_res[labels %in% labels_to_keep, ]

pred_yes <- result_smoke_11$pred_grid
pred_no <- result_nosmoke_11$pred_grid

library(reshape2)
grid_data <- melt(pred[,,1])
colnames(grid_data) <- c("Y1_index", "Y2_index", "density")
grid_data$Y1 <- grid_y1[grid_data$Y1_index]
grid_data$Y2 <- grid_y2[grid_data$Y2_index]
new_point <- data.frame(Y1 = filt_data[1,1], Y2 = filt_data[1,2])
ggplot() +
  geom_point(data = data.frame(filt_data[,c(1,2,4)]), aes(x = dde, y = gest, color = labels), size = 2, alpha = 0.3) +
  geom_contour(data = grid_data, aes(x = Y1, y = Y2, z = density), color = "black", bins = 10) +  # Contour lines
  #geom_point(data = new_point, aes(x = Y1, y = Y2), color = "purple", size = 4, shape = 17) +  # Add new point
  labs(title = "non smoker", x = "log(DDE)", y = "gestional time") +
  theme_minimal() +
  scale_color_manual(values = c("0" = "#ADD8E6",  # Light Blue
                                "1" = "#90EE90",  # Light Green
                                "2" = "#FFB6C1",  # Light Red (pinkish)
                                "3" = "#FFFFE0",  # Light Yellow
                                "4" = "#EE82EE",  # Light Violet
                                "5" = "#FFDAB9",  # Light Orange (peach puff)
                                "6" = "#F08080",   # Light Coral
                                "7" = "#E0FFFF",   # Light Cyan
                                "8" = "#DDA0DD",   # Light Purple (plum)
                                "9" = "#D8BFD8",  # Thistle (Light Violet)
                                "10" = "#FFE4B5",  # Moccasin (Light Orange)
                                "11" = "#FFB347",  # Apricot (Light Coral variant)
                                "12" = "#AFEEEE",  # Pale Turquoise (Light Cyan variant)
                                "13" = "#E6E6FA",   # Lavender (Light Purple variant)
                                "14" = "#B0E0E6",  # Powder Blue
                                "15" = "#98FB98",  # Pale Green
                                "16" = "#FFCCCB",  # Light Red (Salmon Pink)
                                "17" = "#FFFACD"  # Lemon Chiffon (Light Yellow)
  ))+
  theme(legend.position = "none")

contour(grid_y1, grid_y2, pred[,,1], main = "non smoker , 70 kg")



grid_data_yes <- melt(pred_yes[,,3])
grid_data_no <- melt(pred_no[,,3])
colnames(grid_data_yes) <- c("Y1_index", "Y2_index", "density")
colnames(grid_data_no) <- c("Y1_index", "Y2_index", "density")
grid_data_yes$Y1 <- grid_y1[grid_data_yes$Y1_index]
grid_data_yes$Y2 <- grid_y2[grid_data_yes$Y2_index]
grid_data_no$Y1 <- grid_y1[grid_data_no$Y1_index]
grid_data_no$Y2 <- grid_y2[grid_data_no$Y2_index]

ggplot() +
  geom_point(data = data.frame(data[,c(1,2,3)]), aes(x = dde, y = gest, color = as.factor(smoke)), size = 2, alpha = 0.1) +
  geom_contour(data = grid_data_yes, aes(x = Y1, y = Y2, z = density), color = "black", bins = 7, lwd = 0.8) +
  geom_contour(data = grid_data_no, aes(x = Y1, y = Y2, z = density), color = "#DAA520", bins = 7, lwd = 0.8) +# Contour lines
  #geom_point(data = new_point, aes(x = Y1, y = Y2), color = "purple", size = 4, shape = 17) +  # Add new point
  labs(x = "log(DDE) (μg/l)", y = "gestational age (weeks)", title = "81 Kg") +
  theme_minimal() +
  scale_color_manual(values = c("0" = "#DAA520",
                                "1" = "black"))+
  theme(legend.position = "none",
        plot.title = element_text(hjust = 0.5))

##TOGETHER----
Rcpp::sourceCpp('split_with_g_ngg.cpp')
data <- read_xlsx("data_gest.xlsx")
premature <- data$premature
data$dde <- log(data$dde)
#scaling
data[,5] <- scale(data[,5])
data<-data[,-6]
data<-cbind(data$dde,data)
data_hosp<-data[,-5]
data<-data_hosp[,-3]
colnames(data)[1]<-"dde"
colnames(data_hosp)[1]<-"dde"

invcov_cont <- matrix(1/var(data[,4]))
espilon_star <- compute_lambda(data[,c(3,4)], nrep = 1000,invcov_cont, var_type)
lambda_true <- .001 / mean(espilon_star[,1])

#PPMX
#continuous covariates
#sigma2 known (prior on mu) (ppmx_n)
c1 <- 1
c2 <- 2
#m0 <- colMeans(data_rob[,c(2,3)])
#S0 <- apply(data_rob[,c(2,3)], 2, var)
#sigma2 unknown (prior on mu and sigma2) (ppmx_t)
a0 <- 1
k0 <- 2
#m <- colMeans(data_rob[,c(2,3)])
#b0 <- apply(data_rob[,c(2,3)], 2, var)
#binary covariates
a <- 0.1
b <- 0.1
#categorical covariates
categs_1 <- unique(data_hosp$hosp)
alpha_1 <- rep(0.1, length(categs_1))

sample_rows <- sample(1:nrow(data),1000)
data_sample <- data[sample_rows,]
Y_sample <- data_sample[,c(1,2)]
Y <- data[,c(1,2)]
X <- data[,c(3,4)]

ngg_params <- list(sigma = 0.2, k = 0.3)
mu0 = colMeans((data[,c(1,2)])) #c(0,0)
S0 = 3*cov(as.matrix(data[,c(1,2)])) #diag(diag(cov(as.matrix(data[,c(1,2)]))))
P0_params <- list(mu0 = mu0, k0 = 1/10, S0 = S0, v0 = 2+3)
niter <- 5000
nburn <- 1000
nGibbs <- 0
thinGibbs <- 10
thin <- 1
var_type <- c(0,1)

g_params <- list(fun = "g1", lambda = 1, alpha = 1, cov = invcov_cont)
g_params <- list("ppmx_n", c1, c2, a, b, categs_1, alpha_1)
g_params <- list("ppmx_t", a0, k0, a, b, categs_1, alpha_1)

system.time(result_0 <- run_mcmc(data=as.matrix(data), type=var_type, g_params=g_params, 
                                 ngg_params=ngg_params, P0_params=P0_params, niter=niter, 
                                 nburn=nburn, thin=thin, nGibbs = nGibbs, thin_Gibbs = thinGibbs))


freq_chain<-t(result_0$freq_chain)[-1,]
matplot(freq_chain, type = "l", xlab = "", ylab = "", main = "Split-Merge(0, 10) - G1", ylim = c(0, 1), col = "black", axes = FALSE)
axis(1)
axis(2, at = seq(0.0, 1.0, by = 0.2), labels = format(seq(0.0, 1.0, by = 0.2), nsmall = 1), las = 1)
box()

labels <- as.factor(result_0$best_clus_binder)
table(labels)


ggplot(data.frame(Y, labels), aes(x = dde, y = gest, color = labels)) +
  geom_point(size = 2, alpha = 0.6) +               # Scatter plot with different colors for each group
  labs( x = "log(DDE)", y = "gestional time") +
  theme_minimal() +                                 # Minimal theme for a clean look
  scale_color_manual(values = c("0" = "blue", "1" = "green", "2" = "red", "3" = "yellow", "4" = "violet", "5" = "black", "6" = "lightblue", "7" = "orange")) +
  theme(legend.position = "top") 

##analysis on found clusters

labels <- as.factor(result_0$best_clus_binder)
table(labels)

data_res <- cbind(data,labels)

label_counts <- table(labels)
labels_to_keep <- names(label_counts[label_counts > 24])
filtered_labels <- labels[labels %in% labels_to_keep]
filt_data <- data_res[labels %in% labels_to_keep, ]

table(filt_data[,5])
ggplot(as.data.frame(filt_data), aes(x = gest, fill = filtered_labels)) +
  geom_histogram(binwidth = 0.3, position = "identity", alpha = 0.6, color = "black") +
  labs(title = "", x = "gestational age (weeks)", y = "frequency") +
  scale_fill_manual(values = c("5" = "#E0FFFF",  
                               "1" = "darkblue",  
                               "2" = "darkgray",  
                               "3" = "lightgreen",   
                               "4" = "#4682B4",
                               "0" = "blue",
                               "6" = "#87CEFA",
                               "7" = "lightgray",
                               "8" = "darkgreen"
  ))+
  scale_y_continuous(limits = c(0, 130))+
  theme(legend.position = "none")

ggplot(as.data.frame(filt_data), aes(x = dde, fill = filtered_labels)) +
  geom_histogram(binwidth = 0.1, position = "identity", alpha = 0.7, color = "black") +
  labs(title = "", x = "log(DDE) (µg/l)", y = "frequency") +
  scale_fill_manual(values = c("5" = "#E0FFFF",  
                               "1" = "darkblue",  
                               "2" = "darkgray",  
                               "3" = "lightgreen",   
                               "4" = "#4682B4",
                               "5" = "blue",
                               "6" = "#87CEFA",
                               "7" = "lightgray",
                               "8" = "darkgreen"
  ))+
  scale_y_continuous(limits = c(0, 60))+
  theme(legend.position = "none")

ggplot(as.data.frame(filt_data), aes(x = weigth, fill = filtered_labels)) +
  geom_histogram(binwidth = 3, position = "identity", alpha = 0.7, color = "black") +
  labs(title = "", x = "weight (Kg)", y = "frequency") +
  scale_fill_manual(values = c("5" = "#E0FFFF",  
                               "1" = "darkblue",  
                               "2" = "darkgray",  
                               "3" = "lightgreen",   
                               "4" = "#4682B4",
                               "0" = "blue",
                               "6" = "#87CEFA",
                               "7" = "lightgray",
                               "8" = "darkgreen"
  ))+
  scale_y_continuous(limits = c(0, 75))+
  theme(legend.position = "none")

ggplot(filt_data, aes(x = dde, y = gest, color = filtered_labels)) +
  geom_point(size = 2) +
  labs(title = "", x = "log(DDE) (µg/l)", y = "gestational age (weeks)") +
  scale_color_manual(values = c("5" = "#E0FFFF",  
                                "1" = "darkblue",  
                                "2" = "darkgray",  
                                "3" = "lightgreen",   
                                "4" = "#4682B4",
                                "0" = "blue",
                                "6" = "#87CEFA",
                                "7" = "lightgray",
                                "8" = "darkgreen"
  ))+
  scale_y_continuous(limits = c(27, 46))+
  theme(legend.position = "none")

ggplot(filt_data, aes(x = weigth, y = gest, color = filtered_labels)) +
  geom_point(size=2) +
  labs( x = "weight (Kg)", y = "gestational age (weeks)") +
  scale_color_manual(values = c("5" = "#E0FFFF",  
                                "1" = "darkblue",  
                                "2" = "darkgray",  
                                "3" = "lightgreen",   
                                "4" = "#4682B4",
                                "0" = "blue",
                                "6" = "#87CEFA",
                                "7" = "lightgray",
                                "8" = "darkgreen"
  ))+
  scale_y_continuous(limits = c(27, 46))+
  theme(legend.position = "none")

ggplot(filt_data, aes(x = weigth, y = dde, color = filtered_labels)) +
  geom_point(size=2) +
  labs(x = "weight (Kg)", y = "log(DDE) (µg/l)") +
  scale_color_manual(values = c("5" = "#E0FFFF",  
                                "1" = "darkblue",  
                                "2" = "darkgray",  
                                "3" = "lightgreen",   
                                "4" = "#4682B4",
                                "0" = "blue",
                                "6" = "#87CEFA",
                                "7" = "lightgray",
                                "8" = "darkgreen"
  ))+
  scale_y_continuous(limits = c(1, 6))+
  theme(legend.position = "none")


ggplot(filt_data, aes(x = as.factor(filtered_labels), y = gest, fill = as.factor(filtered_labels))) +
  geom_boxplot(color = "black", alpha = 0.5) +
  labs(x = "Cluster", y = "gestational age (weeks)") +
  scale_fill_manual(values = c("5" = "#E0FFFF",  
                               "1" = "darkblue",  
                               "2" = "darkgray",  
                               "3" = "lightgreen",   
                               "4" = "#4682B4",
                               "0" = "blue",
                               "6" = "#87CEFA",
                               "7" = "lightgray",
                               "8" = "darkgreen"
  ))+
  scale_y_continuous(limits = c(27, 45))+
  theme(legend.position = "none")

ggplot(filt_data, aes(x = as.factor(labels), y = dde, fill = as.factor(labels))) +
  geom_boxplot(color = "black", alpha = 0.5) +
  labs(x = "Cluster", y = "log(DDE) (µg/l)") +
  scale_fill_manual(values = c("5" = "#E0FFFF",  
                               "1" = "darkblue",  
                               "2" = "darkgray",  
                               "3" = "lightgreen",   
                               "4" = "#4682B4",
                               "0" = "blue",
                               "6" = "#87CEFA",
                               "7" = "lightgray",
                               "8" = "darkgreen"
  ))+
  scale_y_continuous(limits = c(0, 5.5))+
  theme(legend.position = "none")

ggplot(filt_data, aes(x = as.factor(labels), y = weigth, fill = as.factor(labels))) +
  geom_boxplot(color = "black", alpha = 0.5) +
  labs(x = "Cluster", y = "weight (Kg)") +
  scale_fill_manual(values = c("5" = "#E0FFFF",  
                               "1" = "darkblue",  
                               "2" = "darkgray",  
                               "3" = "lightgreen",   
                               "4" = "#4682B4",
                               "0" = "blue",
                               "6" = "#87CEFA",
                               "7" = "lightgray",
                               "8" = "darkgreen"
  ))+
  scale_y_continuous(limits = c(-3, 140))+
  theme(legend.position = "none")

library(GGally)
ggpairs(filt_data, columns = c("gest", "dde", "weigth"), aes(color = as.factor(labels)))


table(data$smoke[which(labels==2)])
table(data_sample$smoke[which(labels==1)])
table(data_sample$smoke[which(labels==0)])
table(data$smoke[which(labels==3)])

#
library(dplyr)
df <- data.frame(filt_data$smoke, filt_data$labels)
colnames(df) <- c("Smoke", "groups")
group_counts <- df %>%
  group_by(groups, Smoke) %>%
  summarise(count = n(), .groups = 'drop')

# Print the summary counts
print(group_counts)

# Plot the results using ggplot2
ggplot(group_counts, aes(x = groups, y = count, fill = as.factor(Smoke))) +
  geom_bar(stat = "identity", position = "dodge", alpha = 1) +  # Dodge for side-by-side bars
  labs(x = "Cluster", y = "Count", fill = "Smoke") +
  scale_fill_manual(values = c("0" = "#87CEFA",  # Light Blue
                               "1" = "darkblue"))+
  theme_minimal()
#

histo_by_group <- ggplot(as.data.frame(data_sample), aes(x = as.factor(smoke), fill = as.factor(labels))) +
  geom_histogram(stat = "count", position = "dodge", alpha = 0.8, color = "black") +
  labs(title = "Distribuzione dei Fumatori nei Cluster", x = "Fumatore", y = "Frequenza") +
  scale_fill_brewer(palette = "Set2", name = "Cluster") +  
  geom_text(stat = "count", aes(label = ..count..), vjust = -0.5, position = position_dodge(0.9)) +  
  theme_minimal(base_size = 15) + 
  theme(legend.position = "top") 
print(histo_by_group)

#prediction

grid_y1 <- seq(min(data$dde), max(data$dde), by = 0.03)
grid_y2 <- seq(min(data$gest), 47, by = 0.1)
ngg_params <- list(sigma = 0.2, k = 0.3)
mu0 = colMeans(data[,c(1,2)]) #c(0,0)
S0 = 3*cov(as.matrix(data[,c(1,2)])) #diag(diag(cov(as.matrix(data[,c(1,2)]))))
P0_params <- list(mu0 = mu0, k0 = 1/10, S0 = S0, v0 = 2+3)

niter <- 2000
nburn <- 1500
thin <- 1
nGibbs <- 0
thinGibbs <- 10
var_type <- c(0,1)

g_params <- list(fun = "g1", lambda = 1, alpha = 1, cov = invcov_cont)
g_params <- list("ppmx_n", c1, c2, a, b, categs_1, alpha_1)
g_params <- list("ppmx_t", a0, k0, a, b, categs_1, alpha_1)

###
Xnew <- matrix(c(1,70,0,70,1,58,1,81,0,58,0,81), 6, 2, byrow = TRUE)
Xnew <- matrix(c(1,0), 1, 2, byrow = TRUE)
Xnew <- matrix(c(1,0,0,0,1,-3,1,3,0,-3,0,3), 6, 2, byrow = TRUE)

system.time(result_ppmx <- run_mcmc_pred_multi(as.matrix(data), var_type, g_params, ngg_params,
                                                 P0_params, matrix(c(0,0), nrow=1), Xnew,
                                                 grid_y1, grid_y2, niter, nburn, thin, nGibbs, thinGibbs,
                                                 TRUE, FALSE, FALSE, TRUE))


###

freq_chain<-t(result_pred_11$freq_chain)[-1,]
matplot(freq_chain, type = "l", xlab = "Iteration", main = "Gibbs sampler (2000) with G0",ylim = c(0, 1))

labels <- as.factor(result_ppmx$best_clus_binder)
table(labels)

data_res <- cbind(data,labels)
label_counts <- table(labels)
labels_to_keep <- names(label_counts[label_counts > 10])
filtered_labels <- labels[labels %in% labels_to_keep]
filt_data <- data_res[labels %in% labels_to_keep, ]

pred <- result_ppmx$pred_grid

library(reshape2)
grid_data <- melt(pred[,,6])
colnames(grid_data) <- c("Y1_index", "Y2_index", "density")
grid_data$Y1 <- grid_y1[grid_data$Y1_index]
grid_data$Y2 <- grid_y2[grid_data$Y2_index]
new_point <- data.frame(Y1 = filt_data[1,1], Y2 = filt_data[1,2])
ggplot() +
  geom_point(data = data.frame(filt_data[,c(1,2,5)]), aes(x = dde, y = gest, color = labels), size = 2, alpha = 0.1) +
  geom_contour(data = grid_data, aes(x = Y1, y = Y2, z = density), color = "black", bins = 12, lwd = 0.6) +  # Contour lines
  #geom_point(data = new_point, aes(x = Y1, y = Y2), color = "purple", size = 4, shape = 17) +  # Add new point
  labs(title = "Non-Smoker, 81 kg ", x = "log(DDE) (µg/l)", y = "gestational age (weeks)") +
  theme_minimal()+
  scale_color_manual(values = c("5" = "#E0FFFF",  
                                "1" = "darkblue",  
                                "2" = "darkgray",  
                                "3" = "lightgreen",   
                                "4" = "#4682B4",
                                "0" = "blue",
                                "6" = "#87CEFA",
                                "7" = "lightgray",
                                "8" = "darkgreen"
  ))+
  theme(legend.position = "none")+
  theme(plot.title = element_text(hjust = 0.5))
contour(grid_y1, grid_y2, pred[,,1], main = "non smoker , 70 kg")







