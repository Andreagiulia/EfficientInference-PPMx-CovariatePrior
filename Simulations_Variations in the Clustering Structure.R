library(Rcpp)
library(RcppProgress)
library(mvtnorm)
library(ggplot2)
library(coda)
library(mcclust)
library(sn)
library(readxl)

#Data Generation: (x1,x2)~N, x3~Ber, x4~Ber ; Y ~ N(beta'x, 0.5).
#Model Assumption: Gaussian Mixture components with unknown means and variances,
#                  Y ~ N(beta'x, 0.5)

Rcpp::sourceCpp('GibbsSampler_SplitandMerge.cpp')

##### 50,50,50 -----
set.seed(123)
group_1 <- cbind(rmvnorm(n=3775, mean=c(-3,3),sigma= diag(.5, 2)), 
                 sample(c(0,1), size = 3775, prob = c(0.9, 0.1), replace = T),
                 sample(c(0,1), size = 3775, prob = c(0.9, 0.1), replace = T))
group_1 <- cbind(apply(group_1, 1, function(x) rnorm(1, 1 + x %*% c(5, 2, 1, 0), .5)), group_1)

group_2<- cbind(rmvnorm(3775, c(0,0), diag(.5, 2)), 
                sample(c(0,1), size = 3775, prob = c(0.5, 0.5), replace = T),
                sample(c(0,1), size = 3775, prob = c(0.5, 0.5), replace = T))
group_2 <- cbind(apply(group_2, 1, function(x) rnorm(1, 4 + x %*% c(2, -2, 1, -1), .5)), group_2)

group_3 <- cbind(rmvnorm(3775, c(3,3), diag(.5, 2)), 
                 sample(c(0,1), size = 3775, prob = c(0.1, 0.9), replace = T),
                 sample(c(0,1), size = 3775, prob = c(0.1, 0.9), replace = T))
group_3 <- cbind(apply(group_3, 1, function(x) rnorm(1,  -1 + x %*% c(-5, -2, -1, 1), .5)), group_3)

data <- rbind(group_1, group_2, group_3)
new_column_names <- c("Y", "X1", "x2", "x3", "x4")
colnames(data) <- new_column_names

#scaling
data[,c(2,3)] <- scale(data[,c(2,3)])
clusters <- factor(rep(1:3, c(3775, 3775, 3775)))
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
sampled_train <- sample(nrow(data_train), 150)  #cambio
data_sample <- data_train[sampled_train, ]
Y_sample <- data_sample[,1]
X_sample <- data_sample[,c(2,3,4,5)]
clusters_sample <- clusters_train[sampled_train]

histo_by_group <- ggplot(data.frame(Y_train, clusters_train), aes(x = Y_train, fill = clusters_train)) +
  geom_histogram(binwidth = 1, position = "identity", alpha = 0.7,color = "black")+
  labs(title = "Plot", x = "Y", y = "frequence") +
  scale_fill_manual(values = c("1" = "blue","2" = "darkblue","3" = "#87CEFA"), name = "groups")
print(histo_by_group)

col <-c("1" = "lightblue",  
        "2" = "darkblue",  
        "3" = "darkgray",  
        "4" = "lightgray",   
        "5" = "#E0FFFF",
        "6" = "#4682B4",
        "7" = "#87CEFA",
        "8" = "#87CEEB")
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
k0 <- 4
#m <- colMeans(data_rob[,c(2,3)])
#b0 <- apply(data_rob[,c(2,3)], 2, var)
#binary covariates
a <- 0.1
b <- 0.1
#categorical covariates
categs_1 <- NULL
alpha_1 <- rep(0.1, length(categs_1))

g_params <- list(fun = "g0", lambda = 1, alpha = 1, cov = invcov_cont)
g_params <- list(fun = "g1", lambda = 1, alpha = 1, cov = invcov_cont)
g_params <- list(fun = "g2", lambda = 1, alpha = 1, cov = invcov_cont)
g_params <- list(fun = "g3", lambda = 0.1, alpha = 1, cov = invcov_cont)
g_params <- list("ppmx_n", c1, c2, a, b, categs_1, alpha_1)
g_params <- list("ppmx_t", a0, k0, a, b, categs_1, alpha_1)


P0_params <- list(mu0 = mean(Y_sample), lambda0 = 0.01, a0 = 1, b0 = 15)
ngg_params <- list(sigma = 0.2, k = 0.3)
niter <- 3000
nburn <- 2000
thin <- 1
nGibbs <- 0
thinGibbs <-10
var_type <- c(1,1,0,0)

system.time(result_0 <- run_mcmc(data=as.matrix(data_sample), type=var_type, g_params=g_params, 
                                 ngg_params=ngg_params, P0_params=P0_params, niter=niter, 
                                 nburn=nburn, thin=thin, nGibbs = nGibbs, thin_Gibbs = thinGibbs))

freq_chain<-t(result_0$freq_chain)[-1,]
matplot(freq_chain, type = "l", xlab = "Iteration", main = " Gibbs sampling with N-NIG",ylim = c(0, 1), col = "black")

labels <- as.factor(result_0$best_clus_binder)
table(labels)
histo_by_group <- ggplot(as.data.frame(data_sample), aes(x = Y, fill = labels)) +
  geom_histogram(binwidth = 1, position = "identity", alpha = 0.7, color = "black") +
  labs(title = "Plot", x = "Y", y = "frequency") +
  scale_fill_discrete(name = "groups")
print(histo_by_group)

arandi(clusters_sample, result_0$best_clus_binder)

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
g_params <- list(fun = "g3", lambda = 0.1, alpha = 1, cov = invcov_cont)
time_g3 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list("ppmx_n", c1, c2, a, b, categs_1, alpha_1)
time_ppmx_n <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list("ppmx_t", a0, k0, a, b, categs_1, alpha_1)
time_ppmx_t <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
time_Gibbs <- data.frame(time = c(time_g0, time_g1, time_g2, time_g3,
                                  time_ppmx_n, time_ppmx_t))
rownames(time_Gibbs) <- c("g0", "g1", "g2", "g3", "ppmx_n", "ppmx_t")



nGibbs <- 1
thinGibbs <- 10
itermean <- 5
g_params <- list(fun = "g0", lambda = 1, alpha = 1, cov = invcov_cont)
time_g0 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g1", lambda = 1, alpha = 1, cov = invcov_cont)
time_g1 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g2", lambda = 1, alpha = 1, cov = invcov_cont)
time_g2 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g3", lambda = 0.1, alpha = 1, cov = invcov_cont)
time_g3 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list("ppmx_n", c1, c2, a, b, categs_1, alpha_1)
time_ppmx_n <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list("ppmx_t", a0, k0, a, b, categs_1, alpha_1)
time_ppmx_t <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
time_Split_1_10 <- data.frame(time = c(time_g0, time_g1, time_g2, time_g3,
                                       time_ppmx_n, time_ppmx_t))
rownames(time_Split_1_10) <- c("g0", "g1", "g2", "g3", "ppmx_n", "ppmx_t")


nGibbs <- 0
thinGibbs <- 10
itermean <- 5
g_params <- list(fun = "g0", lambda = 1, alpha = 1, cov = invcov_cont)
time_g0 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g1", lambda = 1, alpha = 1, cov = invcov_cont)
time_g1 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g2", lambda = 1, alpha = 1, cov = invcov_cont)
time_g2 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g3", lambda = 0.1, alpha = 1, cov = invcov_cont)
time_g3 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list("ppmx_n", c1, c2, a, b, categs_1, alpha_1)
time_ppmx_n <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list("ppmx_t", a0, k0, a, b, categs_1, alpha_1)
time_ppmx_t <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
time_Split_0_10 <- data.frame(time = c(time_g0, time_g1, time_g2, time_g3,
                                       time_ppmx_n, time_ppmx_t))
rownames(time_Split_0_10) <- c("g0", "g1", "g2", "g3", "ppmx_n", "ppmx_t")


nGibbs <- 0
thinGibbs <- 5
itermean <- 5
g_params <- list(fun = "g0", lambda = 1, alpha = 1, cov = invcov_cont)
time_g0 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g1", lambda = 1, alpha = 1, cov = invcov_cont)
time_g1 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g2", lambda = 1, alpha = 1, cov = invcov_cont)
time_g2 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g3", lambda = 0.1, alpha = 1, cov = invcov_cont)
time_g3 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list("ppmx_n", c1, c2, a, b, categs_1, alpha_1)
time_ppmx_n <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list("ppmx_t", a0, k0, a, b, categs_1, alpha_1)
time_ppmx_t <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
time_Split_0_5 <- data.frame(time = c(time_g0, time_g1, time_g2, time_g3,
                                      time_ppmx_n, time_ppmx_t))
rownames(time_Split_0_5) <- c("g0", "g1", "g2", "g3", "ppmx_n", "ppmx_t")


Gibbs <- cbind(comb_Gibbs, time_Gibbs)
rownames(Gibbs)<- c("G0", "G1", "G2", "G3", "N-N", "N-NIG")
colnames(Gibbs)<- c("ARI","LPML","RMSE", "#clus", "time (sec)")
penultima_posizione <- ncol(comb_Split_0_10)-2

Split_0_10 <- cbind(comb_Split_0_10[1:penultima_posizione], time = time_Split_0_10$time, comb_Split_0_10[(penultima_posizione+1):ncol(comb_Split_0_10)])
rownames(Split_0_10)<- c("G0", "G1", "G2", "G3", "N-N", "N-NIG")
colnames(Split_0_10)<- c("ARI","LPML","RMSE", "#clus", "time (sec)", "accept_rate split", "accept_rate merge")

Split_1_10 <- cbind(comb_Split_1_10[1:penultima_posizione], time = time_Split_1_10$time, comb_Split_1_10[(penultima_posizione+1):ncol(comb_Split_1_10)])
rownames(Split_1_10)<- c("G0", "G1", "G2", "G3", "N-N", "N-NIG")
colnames(Split_1_10)<- c("ARI","LPML","RMSE", "#clus", "time (sec)", "accept_rate split", "accept_rate merge")

Split_0_5 <- cbind(comb_Split_0_5[1:penultima_posizione], time = time_Split_0_5$time, comb_Split_0_5[(penultima_posizione+1):ncol(comb_Split_0_5)])
rownames(Split_0_5)<- c("G0", "G1", "G2", "G3", "N-N", "N-NIG")
colnames(Split_0_5)<- c("ARI","LPML","RMSE", "#clus", "time (sec)", "accept_rate split", "accept_rate merge")

Gibbs <- round(Gibbs,3)
Split_1_10 <- round(Split_1_10, 3)
Split_0_5 <- round(Split_0_5, 3)
Split_0_10 <- round(Split_0_10, 3)


save.image("regression.RData")
#PREDICTION
grid_y1 <- seq(min(Y_train), max(Y_train), by = 0.3)
grid_y2 <- c(0)
niter <- 2000
nburn <- 1000
thin <- 1
nGibbs <- 0
thinGibbs <-1

g_params <- list(fun = "g1", lambda = 1, alpha = 1, cov = invcov_cont)
g_params <- list(fun = "g2", lambda = 1, alpha = 1, cov = invcov_cont)
g_params <- list(fun = "g3", lambda = 0.03, alpha = 1, cov = invcov_cont)
g_params <- list(fun = "g0", lambda = 1, alpha = 1, cov = invcov_cont)
g_params <- list("ppmx_n", c1, c2, a, b, categs_1, alpha_1)
g_params <- list("ppmx_t", a0, k0, a, b, categs_1, alpha_1)

result_pred <- run_mcmc_pred(as.matrix(data_sample), var_type, g_params, ngg_params,
                             P0_params, Y_test[-c(1,2,3,4)], as.matrix(X_test[-c(1,2,3,4),]), grid_y1, niter, nburn, thin, nGibbs, thinGibbs,
                             TRUE, FALSE, FALSE, TRUE)

ggplot(data.frame(x = grid_y1, y = c(colMeans(result_pred$pred_grid)), 
                  ylow = apply(result_pred$pred_grid, 2, quantile, p = 0.05), 
                  yup = apply(result_pred$pred_grid, 2, quantile, p = 0.95))) +
  theme_bw() +
  geom_line(aes(x = x, y = y)) + 
  xlab("y") +
  ylab("density") +
  ggtitle("Predictive distribution with G0") + 
  geom_vline(aes(xintercept = Y_test[5]), lty = 2, col = 2) +
  geom_ribbon(aes(x = x, ymin = ylow, ymax = yup), alpha = 0.25) +
  ylim(c(0, 0.30))+
  xlim(c(-38,10))


#METRICSS
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

nGibbs <- 1
thinGibbs <- 10
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
g_params <- list("ppmx_t",  a0, k0, a, b, categs_1, alpha_1)
metrics_t <- compute_metrics(as.matrix(data_train), clusters_train, var_type, grid_y1, grid_y2,
                             g_params, ngg_params, P0_params, niter, nburn, thin,itermean, nGibbs, thinGibbs,
                             FALSE,FALSE,TRUE,TRUE)
df1 <- data.frame(metrics_0)
df2 <- data.frame(metrics_1)
df3 <- data.frame(metrics_2)
df4 <- data.frame(metrics_3)
df5 <- data.frame(metrics_n)
df6 <- data.frame(metrics_t)
comb_Split_1_10 <- rbind(df1, df2, df3, df4, df5, df6)  
rownames(comb_Split_1_10) <- c("g0", "g1", "g2", "g3", "ppmx_n", "ppmx_t")

nGibbs <- 0
thinGibbs <- 5
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
g_params <- list("ppmx_t",  a0, k0, a, b, categs_1, alpha_1)
metrics_t <- compute_metrics(as.matrix(data_train), clusters_train, var_type, grid_y1, grid_y2,
                             g_params, ngg_params, P0_params, niter, nburn, thin,itermean, nGibbs, thinGibbs,
                             FALSE,FALSE,TRUE,TRUE)
df1 <- data.frame(metrics_0)
df2 <- data.frame(metrics_1)
df3 <- data.frame(metrics_2)
df4 <- data.frame(metrics_3)
df5 <- data.frame(metrics_n)
df6 <- data.frame(metrics_t)
comb_Split_0_5 <- rbind(df1, df2, df3, df4, df5, df6)  
rownames(comb_Split_0_5) <- c("g0", "g1", "g2", "g3", "ppmx_n", "ppmx_t")

nGibbs <- 0
thinGibbs <- 10
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
comb_Split_0_10 <- rbind(df1, df2, df3, df4, df5, df6)  
rownames(comb_Split_0_10) <- c("g0", "g1", "g2", "g3","ppmx_n", "ppmx_t")




  #OK #OK
##### 120,60,20 -----
set.seed(123)
#200 = tot
group_1 <- cbind(rmvnorm(n=6000, mean=c(-3,3),sigma= diag(.5, 2)), 
                 sample(c(0,1), size = 6000, prob = c(0.9, 0.1), replace = T),
                 sample(c(0,1), size = 6000, prob = c(0.9, 0.1), replace = T))
group_1 <- cbind(apply(group_1, 1, function(x) rnorm(1, 1 + x %*% c(5, 2, 1, 0), .5)), group_1)

group_2<- cbind(rmvnorm(3000, c(0,0), diag(.5, 2)), 
                sample(c(0,1), size = 3000, prob = c(0.5, 0.5), replace = T),
                sample(c(0,1), size = 3000, prob = c(0.5, 0.5), replace = T))
group_2 <- cbind(apply(group_2, 1, function(x) rnorm(1, 4 + x %*% c(2, -2, 1, -1), .5)), group_2)

group_3 <- cbind(rmvnorm(1000, c(3,3), diag(.5, 2)), 
                 sample(c(0,1), size = 1000, prob = c(0.1, 0.9), replace = T),
                 sample(c(0,1), size = 1000, prob = c(0.1, 0.9), replace = T))
group_3 <- cbind(apply(group_3, 1, function(x) rnorm(1,  -1 + x %*% c(-5, -2, -1, 1), .5)), group_3)

data <- rbind(group_1, group_2, group_3)
new_column_names <- c("Y", "X1", "x2", "x3", "x4")
colnames(data) <- new_column_names

#scaling
data[,c(2,3)] <- scale(data[,c(2,3)])
clusters <- factor(rep(1:3, c(6000, 3000, 1000)))
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
sampled_train <- sample(nrow(data_train), 200)  #cambio
data_sample <- data_train[sampled_train, ]
Y_sample <- data_sample[,1]
X_sample <- data_sample[,c(2,3,4,5)]
clusters_sample <- clusters_train[sampled_train]

histo_by_group <- ggplot(data.frame(Y_train, clusters_train), aes(x = Y_train, fill = clusters_train)) +
  geom_histogram(binwidth = 1, position = "identity", alpha = 0.7,color = "black")+
  labs(title = "Plot", x = "Y", y = "frequence") +
  scale_fill_manual(values = c("1" = "blue","2" = "darkblue","3" = "#87CEFA"), name = "groups")
print(histo_by_group)

col <-c("1" = "lightblue",  
        "2" = "darkblue",  
        "3" = "darkgray",  
        "4" = "lightgray",   
        "5" = "#E0FFFF",
        "6" = "#4682B4",
        "7" = "#87CEFA",
        "8" = "#87CEEB")
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
a0 <- 8
k0 <- 1
#m <- colMeans(data_rob[,c(2,3)])
#b0 <- apply(data_rob[,c(2,3)], 2, var)
#binary covariates
a <- 0.1
b <- 0.1
#categorical covariates
categs_1 <- NULL
alpha_1 <- rep(0.1, length(categs_1))

g_params <- list(fun = "g0", lambda = 1, alpha = 1, cov = invcov_cont)
g_params <- list(fun = "g1", lambda = 1, alpha = 1, cov = invcov_cont)
g_params <- list(fun = "g2", lambda = 1, alpha = 1, cov = invcov_cont)
g_params <- list(fun = "g3", lambda = 0.05, alpha = 1, cov = invcov_cont)
g_params <- list("ppmx_n", c1, c2, a, b, categs_1, alpha_1)
g_params <- list("ppmx_t", a0, k0, a, b, categs_1, alpha_1)


P0_params <- list(mu0 = mean(Y_sample), lambda0 = 0.01, a0 = 1, b0 = 15)
ngg_params <- list(sigma = 0.2, k = 0.3)
niter <- 3000
nburn <- 2000
thin <- 1
nGibbs <- 0
thinGibbs <-10
var_type <- c(1,1,0,0)

system.time(result_0 <- run_mcmc(data=as.matrix(data_sample), type=var_type, g_params=g_params, 
                                 ngg_params=ngg_params, P0_params=P0_params, niter=niter, 
                                 nburn=nburn, thin=thin, nGibbs = nGibbs, thin_Gibbs = thinGibbs))

freq_chain<-t(result_0$freq_chain)[-1,]
matplot(freq_chain, type = "l", xlab = "Iteration", main = " Gibbs (3000, 0, 1) with G0",ylim = c(0, 1), col = "black")

labels <- as.factor(result_0$best_clus_binder)
table(labels)
histo_by_group <- ggplot(as.data.frame(data_sample), aes(x = Y, fill = labels)) +
  geom_histogram(binwidth = 1, position = "identity", alpha = 0.7, color = "black") +
  labs(title = "Plot", x = "Y", y = "frequency") +
  scale_fill_discrete(name = "groups")
print(histo_by_group)

arandi(clusters_sample, result_0$best_clus_binder)


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
g_params <- list(fun = "g3", lambda = 0.05, alpha = 1, cov = invcov_cont)
time_g3 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list("ppmx_n", c1, c2, a, b, categs_1, alpha_1)
time_ppmx_n <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list("ppmx_t", a0, k0, a, b, categs_1, alpha_1)
time_ppmx_t <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
time_Gibbs <- data.frame(time = c(time_g0, time_g1, time_g2, time_g3,
                                  time_ppmx_n, time_ppmx_t))
rownames(time_Gibbs) <- c("g0", "g1", "g2", "g3", "ppmx_n", "ppmx_t")



nGibbs <- 1
thinGibbs <- 10
itermean <- 5
g_params <- list(fun = "g0", lambda = 1, alpha = 1, cov = invcov_cont)
time_g0 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g1", lambda = 1, alpha = 1, cov = invcov_cont)
time_g1 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g2", lambda = 1, alpha = 1, cov = invcov_cont)
time_g2 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g3", lambda = 0.05, alpha = 1, cov = invcov_cont)
time_g3 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list("ppmx_n", c1, c2, a, b, categs_1, alpha_1)
time_ppmx_n <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list("ppmx_t", a0, k0, a, b, categs_1, alpha_1)
time_ppmx_t <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
time_Split_1_10 <- data.frame(time = c(time_g0, time_g1, time_g2, time_g3,
                                       time_ppmx_n, time_ppmx_t))
rownames(time_Split_1_10) <- c("g0", "g1", "g2", "g3", "ppmx_n", "ppmx_t")


nGibbs <- 0
thinGibbs <- 10
itermean <- 5
g_params <- list(fun = "g0", lambda = 1, alpha = 1, cov = invcov_cont)
time_g0 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g1", lambda = 1, alpha = 1, cov = invcov_cont)
time_g1 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g2", lambda = 1, alpha = 1, cov = invcov_cont)
time_g2 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g3", lambda = 0.05, alpha = 1, cov = invcov_cont)
time_g3 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list("ppmx_n", c1, c2, a, b, categs_1, alpha_1)
time_ppmx_n <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list("ppmx_t", a0, k0, a, b, categs_1, alpha_1)
time_ppmx_t <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
time_Split_0_10 <- data.frame(time = c(time_g0, time_g1, time_g2, time_g3,
                                       time_ppmx_n, time_ppmx_t))
rownames(time_Split_0_10) <- c("g0", "g1", "g2", "g3", "ppmx_n", "ppmx_t")


nGibbs <- 0
thinGibbs <- 5
itermean <- 5
g_params <- list(fun = "g0", lambda = 1, alpha = 1, cov = invcov_cont)
time_g0 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g1", lambda = 1, alpha = 1, cov = invcov_cont)
time_g1 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g2", lambda = 1, alpha = 1, cov = invcov_cont)
time_g2 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g3", lambda = 0.05, alpha = 1, cov = invcov_cont)
time_g3 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list("ppmx_n", c1, c2, a, b, categs_1, alpha_1)
time_ppmx_n <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list("ppmx_t", a0, k0, a, b, categs_1, alpha_1)
time_ppmx_t <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
time_Split_0_5 <- data.frame(time = c(time_g0, time_g1, time_g2, time_g3,
                                      time_ppmx_n, time_ppmx_t))
rownames(time_Split_0_5) <- c("g0", "g1", "g2", "g3", "ppmx_n", "ppmx_t")


Gibbs <- cbind(comb_Gibbs, time_Gibbs)
rownames(Gibbs)<- c("G0", "G1", "G2", "G3", "N-N", "N-NIG")
colnames(Gibbs)<- c("ARI","LPML","RMSE", "#clus", "time (sec)")
penultima_posizione <- ncol(comb_Split_0_10)-2

Split_0_10 <- cbind(comb_Split_0_10[1:penultima_posizione], time = time_Split_0_10$time, comb_Split_0_10[(penultima_posizione+1):ncol(comb_Split_0_10)])
rownames(Split_0_10)<- c("G0", "G1", "G2", "G3", "N-N", "N-NIG")
colnames(Split_0_10)<- c("ARI","LPML","RMSE", "#clus", "time (sec)", "accept_rate split", "accept_rate merge")

Split_1_10 <- cbind(comb_Split_1_10[1:penultima_posizione], time = time_Split_1_10$time, comb_Split_1_10[(penultima_posizione+1):ncol(comb_Split_1_10)])
rownames(Split_1_10)<- c("G0", "G1", "G2", "G3", "N-N", "N-NIG")
colnames(Split_1_10)<- c("ARI","LPML","RMSE", "#clus", "time (sec)", "accept_rate split", "accept_rate merge")

Split_0_5 <- cbind(comb_Split_0_5[1:penultima_posizione], time = time_Split_0_5$time, comb_Split_0_5[(penultima_posizione+1):ncol(comb_Split_0_5)])
rownames(Split_0_5)<- c("G0", "G1", "G2", "G3", "N-N", "N-NIG")
colnames(Split_0_5)<- c("ARI","LPML","RMSE", "#clus", "time (sec)", "accept_rate split", "accept_rate merge")

Gibbs <- round(Gibbs,3)
Split_1_10 <- round(Split_1_10, 3)
Split_0_5 <- round(Split_0_5, 3)
Split_0_10 <- round(Split_0_10, 3)


save.image("regression.RData")
#PREDICTION
grid_y1 <- seq(min(Y_train), max(Y_train), by = 0.3)
grid_y2 <- c(0)
niter <- 2000
nburn <- 1000
thin <- 1
nGibbs <- 0
thinGibbs <-1

g_params <- list(fun = "g1", lambda = 1, alpha = 1, cov = invcov_cont)
g_params <- list(fun = "g2", lambda = 1, alpha = 1, cov = invcov_cont)
g_params <- list(fun = "g3", lambda = 0.05, alpha = 1, cov = invcov_cont)
g_params <- list(fun = "g0", lambda = 1, alpha = 1, cov = invcov_cont)
g_params <- list("ppmx_n", c1, c2, a, b, categs_1, alpha_1)
g_params <- list("ppmx_t", a0, k0, a, b, categs_1, alpha_1)

result_pred <- run_mcmc_pred(as.matrix(data_sample), var_type, g_params, ngg_params,
                             P0_params, Y_test[-c(1,2,3,4)], as.matrix(X_test[-c(1,2,3,4),]), grid_y1, niter, nburn, thin, nGibbs, thinGibbs,
                             TRUE, FALSE, FALSE, TRUE)

ggplot(data.frame(x = grid_y1, y = c(colMeans(result_pred$pred_grid)), 
                  ylow = apply(result_pred$pred_grid, 2, quantile, p = 0.05), 
                  yup = apply(result_pred$pred_grid, 2, quantile, p = 0.95))) +
  theme_bw() +
  geom_line(aes(x = x, y = y)) + 
  xlab("y") +
  ylab("density") +
  ggtitle("Predictive distribution with G0") + 
  geom_vline(aes(xintercept = Y_test[5]), lty = 2, col = 2) +
  geom_ribbon(aes(x = x, ymin = ylow, ymax = yup), alpha = 0.25) +
  ylim(c(0, 0.30))+
  xlim(c(-38,10))


#METRICSS
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
g_params <- list(fun = "g3", lambda = 0.05, alpha = 1, cov = invcov_cont)
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

nGibbs <- 1
thinGibbs <- 10
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
g_params <- list(fun = "g3", lambda = 0.05, alpha = 1, cov = invcov_cont)
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
comb_Split_1_10 <- rbind(df1, df2, df3, df4, df5, df6)  
rownames(comb_Split_1_10) <- c("g0", "g1", "g2", "g3", "ppmx_n", "ppmx_t")

nGibbs <- 0
thinGibbs <- 5
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
g_params <- list(fun = "g3", lambda = 0.05, alpha = 1, cov = invcov_cont)
metrics_3 <- compute_metrics(as.matrix(data_train), clusters_train, var_type, grid_y1, grid_y2,
                             g_params, ngg_params, P0_params, niter, nburn, thin,itermean, nGibbs, thinGibbs,
                             FALSE,FALSE,TRUE,TRUE)
g_params <- list("ppmx_n", c1, c2, a, b, categs_1, alpha_1)
metrics_n <- compute_metrics(as.matrix(data_train), clusters_train, var_type, grid_y1, grid_y2,
                             g_params, ngg_params, P0_params, niter, nburn, thin,itermean, nGibbs, thinGibbs,
                             FALSE,FALSE,TRUE,TRUE)
g_params <- list("ppmx_t",a0, k0,a, b, categs_1, alpha_1)
metrics_t <- compute_metrics(as.matrix(data_train), clusters_train, var_type, grid_y1, grid_y2,
                             g_params, ngg_params, P0_params, niter, nburn, thin,itermean, nGibbs, thinGibbs,
                             FALSE,FALSE,TRUE,TRUE)
df1 <- data.frame(metrics_0)
df2 <- data.frame(metrics_1)
df3 <- data.frame(metrics_2)
df4 <- data.frame(metrics_3)
df5 <- data.frame(metrics_n)
df6 <- data.frame(metrics_t)
comb_Split_0_5 <- rbind(df1, df2, df3, df4, df5, df6)  
rownames(comb_Split_0_5) <- c("g0", "g1", "g2", "g3", "ppmx_n", "ppmx_t")

nGibbs <- 0
thinGibbs <- 10
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
g_params <- list(fun = "g3", lambda = 0.05, alpha = 1, cov = invcov_cont)
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
comb_Split_0_10 <- rbind(df1, df2, df3, df4, df5, df6)  
rownames(comb_Split_0_10) <- c("g0", "g1", "g2", "g3","ppmx_n", "ppmx_t")


 #OK  #OK #OK
##### 200,100,30,10 -----
set.seed(123)
#340 = tot
group_1 <- cbind(rmvnorm(n=5882, mean=c(-3,3),sigma= diag(.5, 2)), 
                 sample(c(0,1), size = 5882, prob = c(0.9, 0.1), replace = T),
                 sample(c(0,1), size = 5882, prob = c(0.9, 0.1), replace = T))
group_1 <- cbind(apply(group_1, 1, function(x) rnorm(1, 1 + x %*% c(5, 2, 1, 0), .5)), group_1)

group_2<- cbind(rmvnorm(2940, c(0,0), diag(.5, 2)), 
                sample(c(0,1), size = 2940, prob = c(0.5, 0.5), replace = T),
                sample(c(0,1), size = 2940, prob = c(0.5, 0.5), replace = T))
group_2 <- cbind(apply(group_2, 1, function(x) rnorm(1, 4 + x %*% c(2, -2, 1, -1), .5)), group_2)

group_3 <- cbind(rmvnorm(980, c(3,3), diag(.5, 2)), 
                 sample(c(0,1), size = 980, prob = c(0.1, 0.9), replace = T),
                 sample(c(0,1), size = 980, prob = c(0.1, 0.9), replace = T))
group_3 <- cbind(apply(group_3, 1, function(x) rnorm(1,  -1 + x %*% c(-5, -2, -1, 1), .5)), group_3)

group_4 <- cbind(rmvnorm(330, c(6, -2), diag(.2, 2)), 
                 sample(c(0,1), size = 330, prob = c(0.7, 0.3), replace = TRUE),
                 sample(c(0,1), size = 330, prob = c(0.7, 0.3), replace = TRUE))
group_4 <- cbind(apply(group_4, 1, function(x) rnorm(1, 2 + x %*% c(3, 4, -1, 2), .5)), group_4)


data <- rbind(group_1, group_2, group_3, group_4)
new_column_names <- c("Y", "X1", "x2", "x3", "x4")
colnames(data) <- new_column_names

#scaling
data[,c(2,3)] <- scale(data[,c(2,3)])
clusters <- factor(rep(1:4, c(5882, 2940, 980,330)))
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
sampled_train <- sample(nrow(data_train), 340)  #cambio
data_sample <- data_train[sampled_train, ]
Y_sample <- data_sample[,1]
X_sample <- data_sample[,c(2,3,4,5)]
clusters_sample <- clusters_train[sampled_train]

histo_by_group <- ggplot(data.frame(Y_train, clusters_train), aes(x = Y_train, fill = clusters_train)) +
  geom_histogram(binwidth = 1, position = "identity", alpha = 0.7,color = "black")+
  labs(title = "Plot", x = "Y", y = "frequence") +
  scale_fill_manual(values = c("1" = "blue","2" = "darkblue","3" = "#87CEFA", "4"="darkgray"), name = "groups")
print(histo_by_group)

col <-c("1" = "lightblue",  
        "2" = "darkblue",  
        "3" = "darkgray",  
        "4" = "lightgray",   
        "5" = "#E0FFFF",
        "6" = "#4682B4",
        "7" = "#87CEFA",
        "8" = "#87CEEB")
##
#x1-Y
scatter_plot <- ggplot(as.data.frame(data_train), aes(x = X1, y = Y, color = clusters_train)) +
  geom_point(alpha = 0.6) +
  labs(title = "plot X1-Y", x = "X1", y = "Y") +
  scale_color_manual(values = c("1" = "blue","2" = "darkblue","3" = "#87CEFA",  "4"="darkgray"), name="groups")
print(scatter_plot)

#x2-Y1
scatter_plot <- ggplot(as.data.frame(data_train), aes(x = x2, y = Y, color = clusters_train)) +
  geom_point(alpha = 0.6) +
  labs(title = "plot X2-Y", x = "X2", y = "Y") +
  scale_color_manual(values = c("1" = "blue","2" = "darkblue","3" = "#87CEFA",  "4"="darkgray"), name="groups")
print(scatter_plot)

#x3-Y1
scatter_plot <- ggplot(as.data.frame(data_train), aes(x = x3, y = Y, color = clusters_train)) +
  geom_point(alpha = 0.6) +
  labs(title = "plot X3-Y", x = "X3", y = "Y") +
  scale_color_manual(values = c("1" = "blue","2" = "darkblue","3" = "#87CEFA",  "4"="darkgray"), name="groups")
print(scatter_plot)

#x4-Y1
scatter_plot <- ggplot(as.data.frame(data_train), aes(x = x4, y = Y, color = clusters_train)) +
  geom_point(alpha = 0.6) +
  labs(title = "plot X4-Y", x = "X4", y = "Y") +
  scale_color_manual(values = c("1" = "blue","2" = "darkblue","3" = "#87CEFA",  "4"="darkgray"), name="groups")
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
k0 <- 8
#m <- colMeans(data_rob[,c(2,3)])
#b0 <- apply(data_rob[,c(2,3)], 2, var)
#binary covariates
a <- 0.1
b <- 0.1
#categorical covariates
categs_1 <- NULL
alpha_1 <- rep(0.1, length(categs_1))

g_params <- list(fun = "g0", lambda = 1, alpha = 1, cov = invcov_cont)
g_params <- list(fun = "g1", lambda = 0.5, alpha = 1, cov = invcov_cont)
g_params <- list(fun = "g2", lambda = 1, alpha = 1, cov = invcov_cont)
g_params <- list(fun = "g3", lambda = 0.05, alpha = 1, cov = invcov_cont)
g_params <- list("ppmx_n", c1, c2, a, b, categs_1, alpha_1)
g_params <- list("ppmx_t", a0, k0, a, b, categs_1, alpha_1)


P0_params <- list(mu0 = mean(Y_sample), lambda0 = 0.01, a0 = 1, b0 = 15)
ngg_params <- list(sigma = 0.2, k = 0.3)
niter <- 3000
nburn <- 2000
thin <- 1
nGibbs <- 0
thinGibbs <-10
var_type <- c(1,1,0,0)

system.time(result_0 <- run_mcmc(data=as.matrix(data_sample), type=var_type, g_params=g_params, 
                                 ngg_params=ngg_params, P0_params=P0_params, niter=niter, 
                                 nburn=nburn, thin=thin, nGibbs = nGibbs, thin_Gibbs = thinGibbs))

freq_chain<-t(result_0$freq_chain)[-1,]
matplot(freq_chain, type = "l", xlab = "Iteration", main = " Gibbs (3000, 0, 1) with G0",ylim = c(0, 1), col = "black")

labels <- as.factor(result_0$best_clus_binder)
table(labels)
histo_by_group <- ggplot(as.data.frame(data_sample), aes(x = Y, fill = labels)) +
  geom_histogram(binwidth = 1, position = "identity", alpha = 0.7, color = "black") +
  labs(title = "Plot", x = "Y", y = "frequency") +
  scale_fill_discrete(name = "groups")
print(histo_by_group)

arandi(clusters_sample, result_0$best_clus_binder)


niter <- 3000
nburn <- 2000
thinGibbs <- 1
nGibbs <- 0
thin <- 1
itermean <- 5
g_params <- list(fun = "g0", lambda = 1, alpha = 1, cov = invcov_cont)
time_g0 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g1", lambda = 0.5, alpha = 1, cov = invcov_cont)
time_g1 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g2", lambda = 1, alpha = 1, cov = invcov_cont)
time_g2 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g3", lambda = 0.05, alpha = 1, cov = invcov_cont)
time_g3 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list("ppmx_n", c1, c2, a, b, categs_1, alpha_1)
time_ppmx_n <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list("ppmx_t", a0, k0, a, b, categs_1, alpha_1)
time_ppmx_t <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
time_Gibbs <- data.frame(time = c(time_g0, time_g1, time_g2, time_g3,
                                  time_ppmx_n, time_ppmx_t))
rownames(time_Gibbs) <- c("g0", "g1", "g2", "g3", "ppmx_n", "ppmx_t")



nGibbs <- 1
thinGibbs <- 10
itermean <- 5
g_params <- list(fun = "g0", lambda = 1, alpha = 1, cov = invcov_cont)
time_g0 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g1", lambda = 0.5, alpha = 1, cov = invcov_cont)
time_g1 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g2", lambda = 1, alpha = 1, cov = invcov_cont)
time_g2 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g3", lambda = 0.05, alpha = 1, cov = invcov_cont)
time_g3 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list("ppmx_n", c1, c2, a, b, categs_1, alpha_1)
time_ppmx_n <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list("ppmx_t", a0, k0, a, b, categs_1, alpha_1)
time_ppmx_t <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
time_Split_1_10 <- data.frame(time = c(time_g0, time_g1, time_g2, time_g3,
                                       time_ppmx_n, time_ppmx_t))
rownames(time_Split_1_10) <- c("g0", "g1", "g2", "g3", "ppmx_n", "ppmx_t")


nGibbs <- 0
thinGibbs <- 10
itermean <- 5
g_params <- list(fun = "g0", lambda = 1, alpha = 1, cov = invcov_cont)
time_g0 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g1", lambda = 0.5, alpha = 1, cov = invcov_cont)
time_g1 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g2", lambda = 1, alpha = 1, cov = invcov_cont)
time_g2 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g3", lambda = 0.05, alpha = 1, cov = invcov_cont)
time_g3 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list("ppmx_n", c1, c2, a, b, categs_1, alpha_1)
time_ppmx_n <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list("ppmx_t", a0, k0, a, b, categs_1, alpha_1)
time_ppmx_t <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
time_Split_0_10 <- data.frame(time = c(time_g0, time_g1, time_g2, time_g3,
                                       time_ppmx_n, time_ppmx_t))
rownames(time_Split_0_10) <- c("g0", "g1", "g2", "g3", "ppmx_n", "ppmx_t")


nGibbs <- 0
thinGibbs <- 5
itermean <- 5
g_params <- list(fun = "g0", lambda = 1, alpha = 1, cov = invcov_cont)
time_g0 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g1", lambda = 0.5, alpha = 1, cov = invcov_cont)
time_g1 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g2", lambda = 1, alpha = 1, cov = invcov_cont)
time_g2 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g3", lambda = 0.05, alpha = 1, cov = invcov_cont)
time_g3 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list("ppmx_n", c1, c2, a, b, categs_1, alpha_1)
time_ppmx_n <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list("ppmx_t", a0, k0, a, b, categs_1, alpha_1)
time_ppmx_t <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
time_Split_0_5 <- data.frame(time = c(time_g0, time_g1, time_g2, time_g3,
                                      time_ppmx_n, time_ppmx_t))
rownames(time_Split_0_5) <- c("g0", "g1", "g2", "g3", "ppmx_n", "ppmx_t")


Gibbs <- cbind(comb_Gibbs, time_Gibbs)
rownames(Gibbs)<- c("G0", "G1", "G2", "G3", "N-N", "N-NIG")
colnames(Gibbs)<- c("ARI","LPML","RMSE", "#clus", "time (sec)")
penultima_posizione <- ncol(comb_Split_0_10)-2

Split_0_10 <- cbind(comb_Split_0_10[1:penultima_posizione], time = time_Split_0_10$time, comb_Split_0_10[(penultima_posizione+1):ncol(comb_Split_0_10)])
rownames(Split_0_10)<- c("G0", "G1", "G2", "G3", "N-N", "N-NIG")
colnames(Split_0_10)<- c("ARI","LPML","RMSE", "#clus", "time (sec)", "accept_rate split", "accept_rate merge")

Split_1_10 <- cbind(comb_Split_1_10[1:penultima_posizione], time = time_Split_1_10$time, comb_Split_1_10[(penultima_posizione+1):ncol(comb_Split_1_10)])
rownames(Split_1_10)<- c("G0", "G1", "G2", "G3", "N-N", "N-NIG")
colnames(Split_1_10)<- c("ARI","LPML","RMSE", "#clus", "time (sec)", "accept_rate split", "accept_rate merge")

Split_0_5 <- cbind(comb_Split_0_5[1:penultima_posizione], time = time_Split_0_5$time, comb_Split_0_5[(penultima_posizione+1):ncol(comb_Split_0_5)])
rownames(Split_0_5)<- c("G0", "G1", "G2", "G3", "N-N", "N-NIG")
colnames(Split_0_5)<- c("ARI","LPML","RMSE", "#clus", "time (sec)", "accept_rate split", "accept_rate merge")

Gibbs <- round(Gibbs,3)
Split_1_10 <- round(Split_1_10, 3)
Split_0_5 <- round(Split_0_5, 3)
Split_0_10 <- round(Split_0_10, 3)


save.image("regression.RData")
#PREDICTION
grid_y1 <- seq(min(Y_train), max(Y_train), by = 0.3)
grid_y2 <- c(0)
niter <- 2000
nburn <- 1000
thin <- 1
nGibbs <- 0
thinGibbs <-1

g_params <- list(fun = "g1", lambda = 1, alpha = 1, cov = invcov_cont)
g_params <- list(fun = "g2", lambda = 1, alpha = 1, cov = invcov_cont)
g_params <- list(fun = "g3", lambda = 0.03, alpha = 1, cov = invcov_cont)
g_params <- list(fun = "g0", lambda = 1, alpha = 1, cov = invcov_cont)
g_params <- list("ppmx_n", c1, c2, a, b, categs_1, alpha_1)
g_params <- list("ppmx_t", a0, k0, a, b, categs_1, alpha_1)

result_pred <- run_mcmc_pred(as.matrix(data_sample), var_type, g_params, ngg_params,
                             P0_params, Y_test[-c(1,2,3,4)], as.matrix(X_test[-c(1,2,3,4),]), grid_y1, niter, nburn, thin, nGibbs, thinGibbs,
                             TRUE, FALSE, FALSE, TRUE)

ggplot(data.frame(x = grid_y1, y = c(colMeans(result_pred$pred_grid)), 
                  ylow = apply(result_pred$pred_grid, 2, quantile, p = 0.05), 
                  yup = apply(result_pred$pred_grid, 2, quantile, p = 0.95))) +
  theme_bw() +
  geom_line(aes(x = x, y = y)) + 
  xlab("y") +
  ylab("density") +
  ggtitle("Predictive distribution with G0") + 
  geom_vline(aes(xintercept = Y_test[5]), lty = 2, col = 2) +
  geom_ribbon(aes(x = x, ymin = ylow, ymax = yup), alpha = 0.25) +
  ylim(c(0, 0.30))+
  xlim(c(-38,10))


#METRICSS
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
g_params <- list(fun = "g1", lambda = 0.5, alpha = 1, cov = invcov_cont)
metrics_1 <- compute_metrics(as.matrix(data_train), clusters_train, var_type, grid_y1, grid_y2,
                             g_params, ngg_params, P0_params, niter, nburn, thin,itermean, nGibbs, thinGibbs,
                             FALSE,FALSE,TRUE,TRUE)
g_params <- list(fun = "g2", lambda = 1, alpha = 1, cov = invcov_cont)
metrics_2 <- compute_metrics(as.matrix(data_train), clusters_train, var_type, grid_y1, grid_y2,
                             g_params, ngg_params, P0_params, niter, nburn, thin,itermean, nGibbs, thinGibbs,
                             FALSE,FALSE,TRUE,TRUE)
g_params <- list(fun = "g3", lambda = 0.05, alpha = 1, cov = invcov_cont)
metrics_3 <- compute_metrics(as.matrix(data_train), clusters_train, var_type, grid_y1, grid_y2,
                             g_params, ngg_params, P0_params, niter, nburn, thin,itermean, nGibbs, thinGibbs,
                             FALSE,FALSE,TRUE,TRUE)
itermean <- 10
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

nGibbs <- 1
thinGibbs <- 10
itermean <- 30
g_params <- list(fun = "g0", lambda = 1, alpha = 1, cov = invcov_cont)
metrics_0 <- compute_metrics(as.matrix(data_train), clusters_train, var_type, grid_y1, grid_y2,
                             g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs,
                             FALSE,FALSE,TRUE,TRUE)
g_params <- list(fun = "g1", lambda = 0.5, alpha = 1, cov = invcov_cont)
metrics_1 <- compute_metrics(as.matrix(data_train), clusters_train, var_type, grid_y1, grid_y2,
                             g_params, ngg_params, P0_params, niter, nburn, thin,itermean, nGibbs, thinGibbs,
                             FALSE,FALSE,TRUE,TRUE)
g_params <- list(fun = "g2", lambda = 1, alpha = 1, cov = invcov_cont)
metrics_2 <- compute_metrics(as.matrix(data_train), clusters_train, var_type, grid_y1, grid_y2,
                             g_params, ngg_params, P0_params, niter, nburn, thin,itermean, nGibbs, thinGibbs,
                             FALSE,FALSE,TRUE,TRUE)
g_params <- list(fun = "g3", lambda = 0.05, alpha = 1, cov = invcov_cont)
metrics_3 <- compute_metrics(as.matrix(data_train), clusters_train, var_type, grid_y1, grid_y2,
                             g_params, ngg_params, P0_params, niter, nburn, thin,itermean, nGibbs, thinGibbs,
                             FALSE,FALSE,TRUE,TRUE)
itermean <- 10
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
comb_Split_1_10 <- rbind(df1, df2, df3, df4, df5, df6)  
rownames(comb_Split_1_10) <- c("g0", "g1", "g2", "g3", "ppmx_n", "ppmx_t")

nGibbs <- 0
thinGibbs <- 5
itermean <- 30
g_params <- list(fun = "g0", lambda = 1, alpha = 1, cov = invcov_cont)
metrics_0 <- compute_metrics(as.matrix(data_train), clusters_train, var_type, grid_y1, grid_y2,
                             g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs,
                             FALSE,FALSE,TRUE,TRUE)
g_params <- list(fun = "g1", lambda = 0.5, alpha = 1, cov = invcov_cont)
metrics_1 <- compute_metrics(as.matrix(data_train), clusters_train, var_type, grid_y1, grid_y2,
                             g_params, ngg_params, P0_params, niter, nburn, thin,itermean, nGibbs, thinGibbs,
                             FALSE,FALSE,TRUE,TRUE)
g_params <- list(fun = "g2", lambda = 1, alpha = 1, cov = invcov_cont)
metrics_2 <- compute_metrics(as.matrix(data_train), clusters_train, var_type, grid_y1, grid_y2,
                             g_params, ngg_params, P0_params, niter, nburn, thin,itermean, nGibbs, thinGibbs,
                             FALSE,FALSE,TRUE,TRUE)
g_params <- list(fun = "g3", lambda = 0.05, alpha = 1, cov = invcov_cont)
metrics_3 <- compute_metrics(as.matrix(data_train), clusters_train, var_type, grid_y1, grid_y2,
                             g_params, ngg_params, P0_params, niter, nburn, thin,itermean, nGibbs, thinGibbs,
                             FALSE,FALSE,TRUE,TRUE)
itermean <- 10
g_params <- list("ppmx_n", c1, c2, a, b, categs_1, alpha_1)
metrics_n <- compute_metrics(as.matrix(data_train), clusters_train, var_type, grid_y1, grid_y2,
                             g_params, ngg_params, P0_params, niter, nburn, thin,itermean, nGibbs, thinGibbs,
                             FALSE,FALSE,TRUE,TRUE)
g_params <- list("ppmx_t",a0, k0,a, b, categs_1, alpha_1)
metrics_t <- compute_metrics(as.matrix(data_train), clusters_train, var_type, grid_y1, grid_y2,
                             g_params, ngg_params, P0_params, niter, nburn, thin,itermean, nGibbs, thinGibbs,
                             FALSE,FALSE,TRUE,TRUE)
df1 <- data.frame(metrics_0)
df2 <- data.frame(metrics_1)
df3 <- data.frame(metrics_2)
df4 <- data.frame(metrics_3)
df5 <- data.frame(metrics_n)
df6 <- data.frame(metrics_t)
comb_Split_0_5 <- rbind(df1, df2, df3, df4, df5, df6)  
rownames(comb_Split_0_5) <- c("g0", "g1", "g2", "g3", "ppmx_n", "ppmx_t")

nGibbs <- 0
thinGibbs <- 10
itermean <- 30
g_params <- list(fun = "g0", lambda = 1, alpha = 1, cov = invcov_cont)
metrics_0 <- compute_metrics(as.matrix(data_train), clusters_train, var_type, grid_y1, grid_y2,
                             g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs,
                             FALSE,FALSE,TRUE,TRUE)
g_params <- list(fun = "g1", lambda = 0.5, alpha = 1, cov = invcov_cont)
metrics_1 <- compute_metrics(as.matrix(data_train), clusters_train, var_type, grid_y1, grid_y2,
                             g_params, ngg_params, P0_params, niter, nburn, thin,itermean, nGibbs, thinGibbs,
                             FALSE,FALSE,TRUE,TRUE)
g_params <- list(fun = "g2", lambda = 1, alpha = 1, cov = invcov_cont)
metrics_2 <- compute_metrics(as.matrix(data_train), clusters_train, var_type, grid_y1, grid_y2,
                             g_params, ngg_params, P0_params, niter, nburn, thin,itermean, nGibbs, thinGibbs,
                             FALSE,FALSE,TRUE,TRUE)
g_params <- list(fun = "g3", lambda = 0.05, alpha = 1, cov = invcov_cont)
metrics_3 <- compute_metrics(as.matrix(data_train), clusters_train, var_type, grid_y1, grid_y2,
                             g_params, ngg_params, P0_params, niter, nburn, thin,itermean, nGibbs, thinGibbs,
                             FALSE,FALSE,TRUE,TRUE)
itermean <- 10
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
comb_Split_0_10 <- rbind(df1, df2, df3, df4, df5, df6)  
rownames(comb_Split_0_10) <- c("g0", "g1", "g2", "g3","ppmx_n", "ppmx_t")

niter <- 3000
nburn <- 2000
thinGibbs <- 1
nGibbs <- 0
thin <- 1
itermean <- 5
g_params <- list(fun = "g0", lambda = 1, alpha = 1, cov = invcov_cont)
time_g0 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g1", lambda = 0.5, alpha = 1, cov = invcov_cont)
time_g1 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g2", lambda = 1, alpha = 1, cov = invcov_cont)
time_g2 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g3", lambda = 0.05, alpha = 1, cov = invcov_cont)
time_g3 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list("ppmx_n", c1, c2, a, b, categs_1, alpha_1)
time_ppmx_n <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list("ppmx_t", a0, k0, a, b, categs_1, alpha_1)
time_ppmx_t <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
time_Gibbs <- data.frame(time = c(time_g0, time_g1, time_g2, time_g3,
                                  time_ppmx_n, time_ppmx_t))
rownames(time_Gibbs) <- c("g0", "g1", "g2", "g3", "ppmx_n", "ppmx_t")



nGibbs <- 1
thinGibbs <- 10
itermean <- 5
g_params <- list(fun = "g0", lambda = 1, alpha = 1, cov = invcov_cont)
time_g0 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g1", lambda = 0.5, alpha = 1, cov = invcov_cont)
time_g1 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g2", lambda = 1, alpha = 1, cov = invcov_cont)
time_g2 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g3", lambda = 0.05, alpha = 1, cov = invcov_cont)
time_g3 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list("ppmx_n", c1, c2, a, b, categs_1, alpha_1)
time_ppmx_n <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list("ppmx_t", a0, k0, a, b, categs_1, alpha_1)
time_ppmx_t <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
time_Split_1_10 <- data.frame(time = c(time_g0, time_g1, time_g2, time_g3,
                                       time_ppmx_n, time_ppmx_t))
rownames(time_Split_1_10) <- c("g0", "g1", "g2", "g3", "ppmx_n", "ppmx_t")


nGibbs <- 0
thinGibbs <- 10
itermean <- 5
g_params <- list(fun = "g0", lambda = 1, alpha = 1, cov = invcov_cont)
time_g0 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g1", lambda = 0.5, alpha = 1, cov = invcov_cont)
time_g1 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g2", lambda = 1, alpha = 1, cov = invcov_cont)
time_g2 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g3", lambda = 0.05, alpha = 1, cov = invcov_cont)
time_g3 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list("ppmx_n", c1, c2, a, b, categs_1, alpha_1)
time_ppmx_n <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list("ppmx_t", a0, k0, a, b, categs_1, alpha_1)
time_ppmx_t <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
time_Split_0_10 <- data.frame(time = c(time_g0, time_g1, time_g2, time_g3,
                                       time_ppmx_n, time_ppmx_t))
rownames(time_Split_0_10) <- c("g0", "g1", "g2", "g3", "ppmx_n", "ppmx_t")


nGibbs <- 0
thinGibbs <- 5
itermean <- 5
g_params <- list(fun = "g0", lambda = 1, alpha = 1, cov = invcov_cont)
time_g0 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g1", lambda = 0.5, alpha = 1, cov = invcov_cont)
time_g1 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g2", lambda = 1, alpha = 1, cov = invcov_cont)
time_g2 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g3", lambda = 0.05, alpha = 1, cov = invcov_cont)
time_g3 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list("ppmx_n", c1, c2, a, b, categs_1, alpha_1)
time_ppmx_n <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list("ppmx_t", a0, k0, a, b, categs_1, alpha_1)
time_ppmx_t <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
time_Split_0_5 <- data.frame(time = c(time_g0, time_g1, time_g2, time_g3,
                                      time_ppmx_n, time_ppmx_t))
rownames(time_Split_0_5) <- c("g0", "g1", "g2", "g3", "ppmx_n", "ppmx_t")


Gibbs <- cbind(comb_Gibbs, time_Gibbs)
rownames(Gibbs)<- c("G0", "G1", "G2", "G3", "N-N", "N-NIG")
colnames(Gibbs)<- c("ARI","LPML","RMSE", "#clus", "time (sec)")
penultima_posizione <- ncol(comb_Split_0_10)-2

Split_0_10 <- cbind(comb_Split_0_10[1:penultima_posizione], time = time_Split_0_10$time, comb_Split_0_10[(penultima_posizione+1):ncol(comb_Split_0_10)])
rownames(Split_0_10)<- c("G0", "G1", "G2", "G3", "N-N", "N-NIG")
colnames(Split_0_10)<- c("ARI","LPML","RMSE", "#clus", "time (sec)", "accept_rate split", "accept_rate merge")

Split_1_10 <- cbind(comb_Split_1_10[1:penultima_posizione], time = time_Split_1_10$time, comb_Split_1_10[(penultima_posizione+1):ncol(comb_Split_1_10)])
rownames(Split_1_10)<- c("G0", "G1", "G2", "G3", "N-N", "N-NIG")
colnames(Split_1_10)<- c("ARI","LPML","RMSE", "#clus", "time (sec)", "accept_rate split", "accept_rate merge")

Split_0_5 <- cbind(comb_Split_0_5[1:penultima_posizione], time = time_Split_0_5$time, comb_Split_0_5[(penultima_posizione+1):ncol(comb_Split_0_5)])
rownames(Split_0_5)<- c("G0", "G1", "G2", "G3", "N-N", "N-NIG")
colnames(Split_0_5)<- c("ARI","LPML","RMSE", "#clus", "time (sec)", "accept_rate split", "accept_rate merge")

Gibbs <- round(Gibbs,3)
Split_1_10 <- round(Split_1_10, 3)
Split_0_5 <- round(Split_0_5, 3)
Split_0_10 <- round(Split_0_10, 3)


save.image("provan2.RData")  #UNA MERDINA
##### 50,50,50,50,50,50 -----
set.seed(123)
#300 = tot

n <- 1666
group_1 <- cbind(rmvnorm(n, mean=c(-8, 8), sigma= diag(0.5, 2)), 
                 sample(c(0,1), size = n, prob = c(0.9, 0.1), replace = TRUE),
                 sample(c(0,1), size = n, prob = c(0.9, 0.1), replace = TRUE))
group_1 <- cbind(apply(group_1, 1, function(x) rnorm(1, 1 + x %*% c(5, 2, 1, 0), 0.5)), group_1)

group_2 <- cbind(rmvnorm(n, mean=c(-4, -4), sigma= diag(0.5, 2)), 
                 sample(c(0,1), size = n, prob = c(0.5, 0.5), replace = TRUE),
                 sample(c(0,1), size = n, prob = c(0.5, 0.5), replace = TRUE))
group_2 <- cbind(apply(group_2, 1, function(x) rnorm(1, 4 + x %*% c(2, -2, 1, -1), 0.5)), group_2)

group_3 <- cbind(rmvnorm(n, mean=c(6, 8), sigma= diag(0.5, 2)), 
                 sample(c(0,1), size = n, prob = c(0.1, 0.9), replace = TRUE),
                 sample(c(0,1), size = n, prob = c(0.1, 0.9), replace = TRUE))
group_3 <- cbind(apply(group_3, 1, function(x) rnorm(1, -1 + x %*% c(-5, -2, -1, 1), 0.5)), group_3)

group_4 <- cbind(rmvnorm(n, mean=c(6, -6), sigma= diag(0.5, 2)), 
                 sample(c(0,1), size = n, prob = c(0.7, 0.3), replace = TRUE),
                 sample(c(0,1), size = n, prob = c(0.7, 0.3), replace = TRUE))
group_4 <- cbind(apply(group_4, 1, function(x) rnorm(1, 2 + x %*% c(3, 4, -1, 2), 0.5)), group_4)

group_5 <- cbind(rmvnorm(n, mean=c(7 ,-7), sigma= diag(0.5, 2)), 
                 sample(c(0,1), size = n, prob = c(0.8, 0.2), replace = TRUE),
                 sample(c(0,1), size = n, prob = c(0.8, 0.2), replace = TRUE))
group_5 <- cbind(apply(group_5, 1, function(x) rnorm(1, 3 + x %*% c(2, 1, 1, 0), 0.5)), group_5)

group_6 <- cbind(rmvnorm(n, mean=c(12, 4), sigma= diag(0.5, 2)), 
                 sample(c(0,1), size = n, prob = c(0.6, 0.4), replace = TRUE),
                 sample(c(0,1), size = n, prob = c(0.6, 0.4), replace = TRUE))
group_6 <- cbind(apply(group_6, 1, function(x) rnorm(1, -2 + x %*% c(-4, -3, 1, 0), 0.5)), group_6)


data <- rbind(group_1, group_2, group_3, group_4, group_5, group_6)
new_column_names <- c("Y", "X1", "x2", "x3", "x4")
colnames(data) <- new_column_names

#scaling
data[,c(2,3)] <- scale(data[,c(2,3)])
clusters <- factor(rep(1:6, c(1666, 1666, 1666,1666,1666,1666)))
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
sampled_train <- sample(nrow(data_train), 300)  #cambio
data_sample <- data_train[sampled_train, ]
Y_sample <- data_sample[,1]
X_sample <- data_sample[,c(2,3,4,5)]
clusters_sample <- clusters_train[sampled_train]

histo_by_group <- ggplot(data.frame(Y_train, clusters_train), aes(x = Y_train, fill = clusters_train)) +
  geom_histogram(binwidth = 1, position = "identity", alpha = 0.7,color = "black")+
  labs(title = "Plot", x = "Y", y = "frequence") +
  scale_fill_manual(values = c("1" = "blue","2" = "darkblue","3" = "#87CEFA", "4"="darkgray","5" = "#E0FFFF","6" = "lightgray"), name = "groups")
print(histo_by_group)

col <-c("1" = "lightblue",  
        "2" = "darkblue",  
        "3" = "darkgray",  
        "4" = "lightgray",   
        "5" = "#E0FFFF",
        "6" = "#4682B4",
        "7" = "#87CEFA",
        "8" = "#87CEEB")
##
#x1-Y
scatter_plot <- ggplot(as.data.frame(data_train), aes(x = X1, y = Y, color = clusters_train)) +
  geom_point(alpha = 0.6) +
  labs(title = "plot X1-Y", x = "X1", y = "Y") +
  scale_color_manual(values = c("1" = "blue","2" = "darkblue","3" = "#87CEFA",  "4"="darkgray","5" = "#E0FFFF","6" = "lightgray"), name="groups")
print(scatter_plot)

#x2-Y1
scatter_plot <- ggplot(as.data.frame(data_train), aes(x = x2, y = Y, color = clusters_train)) +
  geom_point(alpha = 0.6) +
  labs(title = "plot X2-Y", x = "X2", y = "Y") +
  scale_color_manual(values = c("1" = "blue","2" = "darkblue","3" = "#87CEFA",  "4"="darkgray","5" = "#E0FFFF","6" = "lightgray"), name="groups")
print(scatter_plot)

#x3-Y1
scatter_plot <- ggplot(as.data.frame(data_train), aes(x = x3, y = Y, color = clusters_train)) +
  geom_point(alpha = 0.6) +
  labs(title = "plot X3-Y", x = "X3", y = "Y") +
  scale_color_manual(values = c("1" = "blue","2" = "darkblue","3" = "#87CEFA",  "4"="darkgray","5" = "#E0FFFF","6" ="lightgray"), name="groups")
print(scatter_plot)

#x4-Y1
scatter_plot <- ggplot(as.data.frame(data_train), aes(x = x4, y = Y, color = clusters_train)) +
  geom_point(alpha = 0.6) +
  labs(title = "plot X4-Y", x = "X4", y = "Y") +
  scale_color_manual(values = c("1" = "blue","2" = "darkblue","3" = "#87CEFA",  "4"="darkgray","5" = "#E0FFFF","6" = "lightgray"), name="groups")
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
a0 <- 8
k0 <- 1
#m <- colMeans(data_rob[,c(2,3)])
#b0 <- apply(data_rob[,c(2,3)], 2, var)
#binary covariates
a <- 0.1
b <- 0.1
#categorical covariates
categs_1 <- NULL
alpha_1 <- rep(0.1, length(categs_1))

g_params <- list(fun = "g0", lambda = 1, alpha = 1, cov = invcov_cont)
g_params <- list(fun = "g1", lambda = 1, alpha = 1, cov = invcov_cont)
g_params <- list(fun = "g2", lambda = 1, alpha = 1, cov = invcov_cont)
g_params <- list(fun = "g3", lambda = 0.05, alpha = 1, cov = invcov_cont)
g_params <- list("ppmx_n", c1, c2, a, b, categs_1, alpha_1)
g_params <- list("ppmx_t", a0, k0, a, b, categs_1, alpha_1)


P0_params <- list(mu0 = mean(Y_sample), lambda0 = 0.01, a0 = 1, b0 = 15)
ngg_params <- list(sigma = 0.2, k = 0.3)
niter <- 3000
nburn <- 2000
thin <- 1
nGibbs <- 0
thinGibbs <-1
var_type <- c(1,1,0,0)

system.time(result_0 <- run_mcmc(data=as.matrix(data_sample), type=var_type, g_params=g_params, 
                                 ngg_params=ngg_params, P0_params=P0_params, niter=niter, 
                                 nburn=nburn, thin=thin, nGibbs = nGibbs, thin_Gibbs = thinGibbs))

freq_chain<-t(result_0$freq_chain)[-1,]
matplot(freq_chain, type = "l", xlab = "Iteration", main = " Gibbs (3000, 0, 1) with G0",ylim = c(0, 1), col = "black")

labels <- as.factor(result_0$best_clus_binder)
table(labels)
histo_by_group <- ggplot(as.data.frame(data_sample), aes(x = Y, fill = labels)) +
  geom_histogram(binwidth = 1, position = "identity", alpha = 0.7, color = "black") +
  labs(title = "Plot", x = "Y", y = "frequency") +
  scale_fill_discrete(name = "groups")
print(histo_by_group)

arandi(clusters_sample, result_0$best_clus_binder)



niter <- 3000
nburn <- 2000
thinGibbs <- 1
nGibbs <- 0
thin <- 1
itermean <- 5
g_params <- list(fun = "g0", lambda = 1, alpha = 1, cov = invcov_cont)
time_g0 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g1", lambda = 0.5, alpha = 1, cov = invcov_cont)
time_g1 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g2", lambda = 1, alpha = 1, cov = invcov_cont)
time_g2 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g3", lambda = 0.03, alpha = 1, cov = invcov_cont)
time_g3 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list("ppmx_n", c1, c2, a, b, categs_1, alpha_1)
time_ppmx_n <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list("ppmx_t", a0, k0, a, b, categs_1, alpha_1)
time_ppmx_t <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
time_Gibbs <- data.frame(time = c(time_g0, time_g1, time_g2, time_g3,
                                  time_ppmx_n, time_ppmx_t))
rownames(time_Gibbs) <- c("g0", "g1", "g2", "g3", "ppmx_n", "ppmx_t")



nGibbs <- 1
thinGibbs <- 10
itermean <- 5
g_params <- list(fun = "g0", lambda = 1, alpha = 1, cov = invcov_cont)
time_g0 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g1", lambda = 0.5, alpha = 1, cov = invcov_cont)
time_g1 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g2", lambda = 1, alpha = 1, cov = invcov_cont)
time_g2 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g3", lambda = 0.03, alpha = 1, cov = invcov_cont)
time_g3 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list("ppmx_n", c1, c2, a, b, categs_1, alpha_1)
time_ppmx_n <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list("ppmx_t", a0, k0, a, b, categs_1, alpha_1)
time_ppmx_t <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
time_Split_1_10 <- data.frame(time = c(time_g0, time_g1, time_g2, time_g3,
                                       time_ppmx_n, time_ppmx_t))
rownames(time_Split_1_10) <- c("g0", "g1", "g2", "g3", "ppmx_n", "ppmx_t")


nGibbs <- 0
thinGibbs <- 10
itermean <- 5
g_params <- list(fun = "g0", lambda = 1, alpha = 1, cov = invcov_cont)
time_g0 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g1", lambda = 0.5, alpha = 1, cov = invcov_cont)
time_g1 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g2", lambda = 1, alpha = 1, cov = invcov_cont)
time_g2 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g3", lambda = 0.03, alpha = 1, cov = invcov_cont)
time_g3 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list("ppmx_n", c1, c2, a, b, categs_1, alpha_1)
time_ppmx_n <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list("ppmx_t", a0, k0, a, b, categs_1, alpha_1)
time_ppmx_t <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
time_Split_0_10 <- data.frame(time = c(time_g0, time_g1, time_g2, time_g3,
                                       time_ppmx_n, time_ppmx_t))
rownames(time_Split_0_10) <- c("g0", "g1", "g2", "g3", "ppmx_n", "ppmx_t")


nGibbs <- 0
thinGibbs <- 5
itermean <- 5
g_params <- list(fun = "g0", lambda = 1, alpha = 1, cov = invcov_cont)
time_g0 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g1", lambda = 0.5, alpha = 1, cov = invcov_cont)
time_g1 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g2", lambda = 1, alpha = 1, cov = invcov_cont)
time_g2 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g3", lambda = 0.03, alpha = 1, cov = invcov_cont)
time_g3 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list("ppmx_n", c1, c2, a, b, categs_1, alpha_1)
time_ppmx_n <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list("ppmx_t", a0, k0, a, b, categs_1, alpha_1)
time_ppmx_t <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
time_Split_0_5 <- data.frame(time = c(time_g0, time_g1, time_g2, time_g3,
                                      time_ppmx_n, time_ppmx_t))
rownames(time_Split_0_5) <- c("g0", "g1", "g2", "g3", "ppmx_n", "ppmx_t")


Gibbs <- cbind(comb_Gibbs, time_Gibbs)
rownames(Gibbs)<- c("G0", "G1", "G2", "G3", "N-N", "N-NIG")
colnames(Gibbs)<- c("ARI","LPML","RMSE", "#clus", "time (sec)")
penultima_posizione <- ncol(comb_Split_0_10)-2

Split_0_10 <- cbind(comb_Split_0_10[1:penultima_posizione], time = time_Split_0_10$time, comb_Split_0_10[(penultima_posizione+1):ncol(comb_Split_0_10)])
rownames(Split_0_10)<- c("G0", "G1", "G2", "G3", "N-N", "N-NIG")
colnames(Split_0_10)<- c("ARI","LPML","RMSE", "#clus", "time (sec)", "accept_rate split", "accept_rate merge")

Split_1_10 <- cbind(comb_Split_1_10[1:penultima_posizione], time = time_Split_1_10$time, comb_Split_1_10[(penultima_posizione+1):ncol(comb_Split_1_10)])
rownames(Split_1_10)<- c("G0", "G1", "G2", "G3", "N-N", "N-NIG")
colnames(Split_1_10)<- c("ARI","LPML","RMSE", "#clus", "time (sec)", "accept_rate split", "accept_rate merge")

Split_0_5 <- cbind(comb_Split_0_5[1:penultima_posizione], time = time_Split_0_5$time, comb_Split_0_5[(penultima_posizione+1):ncol(comb_Split_0_5)])
rownames(Split_0_5)<- c("G0", "G1", "G2", "G3", "N-N", "N-NIG")
colnames(Split_0_5)<- c("ARI","LPML","RMSE", "#clus", "time (sec)", "accept_rate split", "accept_rate merge")

Gibbs <- round(Gibbs,3)
Split_1_10 <- round(Split_1_10, 3)
Split_0_5 <- round(Split_0_5, 3)
Split_0_10 <- round(Split_0_10, 3)


save.image("regression.RData")
#PREDICTION
grid_y1 <- seq(min(Y_train), max(Y_train), by = 0.3)
grid_y2 <- c(0)
niter <- 2000
nburn <- 1000
thin <- 1
nGibbs <- 0
thinGibbs <-1

g_params <- list(fun = "g1", lambda = 1, alpha = 1, cov = invcov_cont)
g_params <- list(fun = "g2", lambda = 1, alpha = 1, cov = invcov_cont)
g_params <- list(fun = "g3", lambda = 0.03, alpha = 1, cov = invcov_cont)
g_params <- list(fun = "g0", lambda = 1, alpha = 1, cov = invcov_cont)
g_params <- list("ppmx_n", c1, c2, a, b, categs_1, alpha_1)
g_params <- list("ppmx_t", a0, k0, a, b, categs_1, alpha_1)

result_pred <- run_mcmc_pred(as.matrix(data_sample), var_type, g_params, ngg_params,
                             P0_params, Y_test[-c(1,2,3,4)], as.matrix(X_test[-c(1,2,3,4),]), grid_y1, niter, nburn, thin, nGibbs, thinGibbs,
                             TRUE, FALSE, FALSE, TRUE)

ggplot(data.frame(x = grid_y1, y = c(colMeans(result_pred$pred_grid)), 
                  ylow = apply(result_pred$pred_grid, 2, quantile, p = 0.05), 
                  yup = apply(result_pred$pred_grid, 2, quantile, p = 0.95))) +
  theme_bw() +
  geom_line(aes(x = x, y = y)) + 
  xlab("y") +
  ylab("density") +
  ggtitle("Predictive distribution with G0") + 
  geom_vline(aes(xintercept = Y_test[5]), lty = 2, col = 2) +
  geom_ribbon(aes(x = x, ymin = ylow, ymax = yup), alpha = 0.25) +
  ylim(c(0, 0.30))+
  xlim(c(-38,10))


#METRICSS
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
g_params <- list(fun = "g3", lambda = 0.05, alpha = 1, cov = invcov_cont)
metrics_3 <- compute_metrics(as.matrix(data_train), clusters_train, var_type, grid_y1, grid_y2,
                             g_params, ngg_params, P0_params, niter, nburn, thin,itermean, nGibbs, thinGibbs,
                             FALSE,FALSE,TRUE,TRUE)
itermean <- 10
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

nGibbs <- 1
thinGibbs <- 10
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
g_params <- list(fun = "g3", lambda = 0.05, alpha = 1, cov = invcov_cont)
metrics_3 <- compute_metrics(as.matrix(data_train), clusters_train, var_type, grid_y1, grid_y2,
                             g_params, ngg_params, P0_params, niter, nburn, thin,itermean, nGibbs, thinGibbs,
                             FALSE,FALSE,TRUE,TRUE)
itermean <- 10
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
comb_Split_1_10 <- rbind(df1, df2, df3, df4, df5, df6)  
rownames(comb_Split_1_10) <- c("g0", "g1", "g2", "g3", "ppmx_n", "ppmx_t")

nGibbs <- 0
thinGibbs <- 5
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
g_params <- list(fun = "g3", lambda = 0.05, alpha = 1, cov = invcov_cont)
metrics_3 <- compute_metrics(as.matrix(data_train), clusters_train, var_type, grid_y1, grid_y2,
                             g_params, ngg_params, P0_params, niter, nburn, thin,itermean, nGibbs, thinGibbs,
                             FALSE,FALSE,TRUE,TRUE)
itermean <- 10
g_params <- list("ppmx_n", c1, c2, a, b, categs_1, alpha_1)
metrics_n <- compute_metrics(as.matrix(data_train), clusters_train, var_type, grid_y1, grid_y2,
                             g_params, ngg_params, P0_params, niter, nburn, thin,itermean, nGibbs, thinGibbs,
                             FALSE,FALSE,TRUE,TRUE)
g_params <- list("ppmx_t",a0, k0,a, b, categs_1, alpha_1)
metrics_t <- compute_metrics(as.matrix(data_train), clusters_train, var_type, grid_y1, grid_y2,
                             g_params, ngg_params, P0_params, niter, nburn, thin,itermean, nGibbs, thinGibbs,
                             FALSE,FALSE,TRUE,TRUE)
df1 <- data.frame(metrics_0)
df2 <- data.frame(metrics_1)
df3 <- data.frame(metrics_2)
df4 <- data.frame(metrics_3)
df5 <- data.frame(metrics_n)
df6 <- data.frame(metrics_t)
comb_Split_0_5 <- rbind(df1, df2, df3, df4, df5, df6)  
rownames(comb_Split_0_5) <- c("g0", "g1", "g2", "g3", "ppmx_n", "ppmx_t")

nGibbs <- 0
thinGibbs <- 10
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
g_params <- list(fun = "g3", lambda = 0.05, alpha = 1, cov = invcov_cont)
metrics_3 <- compute_metrics(as.matrix(data_train), clusters_train, var_type, grid_y1, grid_y2,
                             g_params, ngg_params, P0_params, niter, nburn, thin,itermean, nGibbs, thinGibbs,
                             FALSE,FALSE,TRUE,TRUE)
itermean <- 10
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
comb_Split_0_10 <- rbind(df1, df2, df3, df4, df5, df6)  
rownames(comb_Split_0_10) <- c("g0", "g1", "g2", "g3","ppmx_n", "ppmx_t")


niter <- 3000
nburn <- 2000
thinGibbs <- 1
nGibbs <- 0
thin <- 1
itermean <- 5
g_params <- list(fun = "g0", lambda = 1, alpha = 1, cov = invcov_cont)
time_g0 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g1", lambda = 0.5, alpha = 1, cov = invcov_cont)
time_g1 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g2", lambda = 1, alpha = 1, cov = invcov_cont)
time_g2 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g3", lambda = 0.03, alpha = 1, cov = invcov_cont)
time_g3 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list("ppmx_n", c1, c2, a, b, categs_1, alpha_1)
time_ppmx_n <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list("ppmx_t", a0, k0, a, b, categs_1, alpha_1)
time_ppmx_t <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
time_Gibbs <- data.frame(time = c(time_g0, time_g1, time_g2, time_g3,
                                  time_ppmx_n, time_ppmx_t))
rownames(time_Gibbs) <- c("g0", "g1", "g2", "g3", "ppmx_n", "ppmx_t")



nGibbs <- 1
thinGibbs <- 10
itermean <- 5
g_params <- list(fun = "g0", lambda = 1, alpha = 1, cov = invcov_cont)
time_g0 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g1", lambda = 0.5, alpha = 1, cov = invcov_cont)
time_g1 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g2", lambda = 1, alpha = 1, cov = invcov_cont)
time_g2 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g3", lambda = 0.03, alpha = 1, cov = invcov_cont)
time_g3 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list("ppmx_n", c1, c2, a, b, categs_1, alpha_1)
time_ppmx_n <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list("ppmx_t", a0, k0, a, b, categs_1, alpha_1)
time_ppmx_t <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
time_Split_1_10 <- data.frame(time = c(time_g0, time_g1, time_g2, time_g3,
                                       time_ppmx_n, time_ppmx_t))
rownames(time_Split_1_10) <- c("g0", "g1", "g2", "g3", "ppmx_n", "ppmx_t")


nGibbs <- 0
thinGibbs <- 10
itermean <- 5
g_params <- list(fun = "g0", lambda = 1, alpha = 1, cov = invcov_cont)
time_g0 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g1", lambda = 0.5, alpha = 1, cov = invcov_cont)
time_g1 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g2", lambda = 1, alpha = 1, cov = invcov_cont)
time_g2 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g3", lambda = 0.03, alpha = 1, cov = invcov_cont)
time_g3 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list("ppmx_n", c1, c2, a, b, categs_1, alpha_1)
time_ppmx_n <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list("ppmx_t", a0, k0, a, b, categs_1, alpha_1)
time_ppmx_t <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
time_Split_0_10 <- data.frame(time = c(time_g0, time_g1, time_g2, time_g3,
                                       time_ppmx_n, time_ppmx_t))
rownames(time_Split_0_10) <- c("g0", "g1", "g2", "g3", "ppmx_n", "ppmx_t")


nGibbs <- 0
thinGibbs <- 5
itermean <- 5
g_params <- list(fun = "g0", lambda = 1, alpha = 1, cov = invcov_cont)
time_g0 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g1", lambda = 0.5, alpha = 1, cov = invcov_cont)
time_g1 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g2", lambda = 1, alpha = 1, cov = invcov_cont)
time_g2 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g3", lambda = 0.03, alpha = 1, cov = invcov_cont)
time_g3 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list("ppmx_n", c1, c2, a, b, categs_1, alpha_1)
time_ppmx_n <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list("ppmx_t", a0, k0, a, b, categs_1, alpha_1)
time_ppmx_t <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
time_Split_0_5 <- data.frame(time = c(time_g0, time_g1, time_g2, time_g3,
                                      time_ppmx_n, time_ppmx_t))
rownames(time_Split_0_5) <- c("g0", "g1", "g2", "g3", "ppmx_n", "ppmx_t")


Gibbs <- cbind(comb_Gibbs, time_Gibbs)
rownames(Gibbs)<- c("G0", "G1", "G2", "G3", "N-N", "N-NIG")
colnames(Gibbs)<- c("ARI","LPML","RMSE", "#clus", "time (sec)")
penultima_posizione <- ncol(comb_Split_0_10)-2

Split_0_10 <- cbind(comb_Split_0_10[1:penultima_posizione], time = time_Split_0_10$time, comb_Split_0_10[(penultima_posizione+1):ncol(comb_Split_0_10)])
rownames(Split_0_10)<- c("G0", "G1", "G2", "G3", "N-N", "N-NIG")
colnames(Split_0_10)<- c("ARI","LPML","RMSE", "#clus", "time (sec)", "accept_rate split", "accept_rate merge")

Split_1_10 <- cbind(comb_Split_1_10[1:penultima_posizione], time = time_Split_1_10$time, comb_Split_1_10[(penultima_posizione+1):ncol(comb_Split_1_10)])
rownames(Split_1_10)<- c("G0", "G1", "G2", "G3", "N-N", "N-NIG")
colnames(Split_1_10)<- c("ARI","LPML","RMSE", "#clus", "time (sec)", "accept_rate split", "accept_rate merge")

Split_0_5 <- cbind(comb_Split_0_5[1:penultima_posizione], time = time_Split_0_5$time, comb_Split_0_5[(penultima_posizione+1):ncol(comb_Split_0_5)])
rownames(Split_0_5)<- c("G0", "G1", "G2", "G3", "N-N", "N-NIG")
colnames(Split_0_5)<- c("ARI","LPML","RMSE", "#clus", "time (sec)", "accept_rate split", "accept_rate merge")

Gibbs <- round(Gibbs,3)
Split_1_10 <- round(Split_1_10, 3)
Split_0_5 <- round(Split_0_5, 3)
Split_0_10 <- round(Split_0_10, 3)


save.image("regression.RData") #OK  #OK #OK 
##### 75,75,50,50,25,25 -----
set.seed(123)
group_1 <- cbind(rmvnorm(2500, mean=c(-8, 8), sigma= diag(0.5, 2)), 
                 sample(c(0,1), size = 2500, prob = c(0.9, 0.1), replace = TRUE),
                 sample(c(0,1), size = 2500, prob = c(0.9, 0.1), replace = TRUE))
group_1 <- cbind(apply(group_1, 1, function(x) rnorm(1, 1 + x %*% c(5, 2, 1, 0), 0.5)), group_1)

group_2 <- cbind(rmvnorm(2500, mean=c(-4, -4), sigma= diag(0.5, 2)), 
                 sample(c(0,1), size = 2500, prob = c(0.5, 0.5), replace = TRUE),
                 sample(c(0,1), size = 2500, prob = c(0.5, 0.5), replace = TRUE))
group_2 <- cbind(apply(group_2, 1, function(x) rnorm(1, 4 + x %*% c(2, -2, 1, -1), 0.5)), group_2)

group_3 <- cbind(rmvnorm(1666, mean=c(6, 8), sigma= diag(0.5, 2)), 
                 sample(c(0,1), size = 1666, prob = c(0.1, 0.9), replace = TRUE),
                 sample(c(0,1), size = 1666, prob = c(0.1, 0.9), replace = TRUE))
group_3 <- cbind(apply(group_3, 1, function(x) rnorm(1, -1 + x %*% c(-5, -2, -1, 1), 0.5)), group_3)

group_4 <- cbind(rmvnorm(1666, mean=c(6, -6), sigma= diag(0.5, 2)), 
                 sample(c(0,1), size = 1666, prob = c(0.7, 0.3), replace = TRUE),
                 sample(c(0,1), size = 1666, prob = c(0.7, 0.3), replace = TRUE))
group_4 <- cbind(apply(group_4, 1, function(x) rnorm(1, 2 + x %*% c(3, 4, -1, 2), 0.5)), group_4)

group_5 <- cbind(rmvnorm(833, mean=c(7 ,-7), sigma= diag(0.5, 2)), 
                 sample(c(0,1), size = 833, prob = c(0.8, 0.2), replace = TRUE),
                 sample(c(0,1), size = 833, prob = c(0.8, 0.2), replace = TRUE))
group_5 <- cbind(apply(group_5, 1, function(x) rnorm(1, 3 + x %*% c(2, 1, 1, 0), 0.5)), group_5)

group_6 <- cbind(rmvnorm(833, mean=c(12, 4), sigma= diag(0.5, 2)), 
                 sample(c(0,1), size = 833, prob = c(0.6, 0.4), replace = TRUE),
                 sample(c(0,1), size = 833, prob = c(0.6, 0.4), replace = TRUE))
group_6 <- cbind(apply(group_6, 1, function(x) rnorm(1, -2 + x %*% c(-4, -3, 1, 0), 0.5)), group_6)


data <- rbind(group_1, group_2, group_3, group_4, group_5, group_6)
new_column_names <- c("Y", "X1", "x2", "x3", "x4")
colnames(data) <- new_column_names

#scaling
data[,c(2,3)] <- scale(data[,c(2,3)])
clusters <- factor(rep(1:6, c(2500, 2500, 1666,1666,833,833)))
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
sampled_train <- sample(nrow(data_train), 300)  #cambio
data_sample <- data_train[sampled_train, ]
Y_sample <- data_sample[,1]
X_sample <- data_sample[,c(2,3,4,5)]
clusters_sample <- clusters_train[sampled_train]

histo_by_group <- ggplot(data.frame(Y_train, clusters_train), aes(x = Y_train, fill = clusters_train)) +
  geom_histogram(binwidth = 1, position = "identity", alpha = 0.7,color = "black")+
  labs(title = "Plot", x = "Y", y = "frequence") +
  scale_fill_manual(values = c("1" = "blue","2" = "darkblue","3" = "#87CEFA", "5"="darkgray","4" = "#E0FFFF","6" ="#4682B4"), name = "groups")
print(histo_by_group)

col <-c("1" = "lightblue",  
        "2" = "darkblue",  
        "3" = "darkgray",  
        "4" = "lightgray",   
        "5" = "#E0FFFF",
        "6" = "#4682B4",
        "7" = "#87CEFA",
        "8" = "#87CEEB")
##
#x1-Y
scatter_plot <- ggplot(as.data.frame(data_train), aes(x = X1, y = Y, color = clusters_train)) +
  geom_point(alpha = 0.6) +
  labs(title = "plot X1-Y", x = "X1", y = "Y") +
  scale_color_manual(values = c("1" = "blue","2" = "darkblue","3" = "#87CEFA",  "5"="darkgray","4" = "#E0FFFF","6" = "#4682B4"), name="groups")
print(scatter_plot)

#x2-Y1
scatter_plot <- ggplot(as.data.frame(data_train), aes(x = x2, y = Y, color = clusters_train)) +
  geom_point(alpha = 0.6) +
  labs(title = "plot X2-Y", x = "X2", y = "Y") +
  scale_color_manual(values = c("1" = "blue","2" = "darkblue","3" = "#87CEFA",  "5"="darkgray", "4" = "#E0FFFF","6" ="#4682B4"), name="groups")
print(scatter_plot)

#x3-Y1
scatter_plot <- ggplot(as.data.frame(data_train), aes(x = x3, y = Y, color = clusters_train)) +
  geom_point(alpha = 0.6) +
  labs(title = "plot X3-Y", x = "X3", y = "Y") +
  scale_color_manual(values = c("1" = "blue","2" = "darkblue","3" = "#87CEFA",  "5"="darkgray", "4" = "#E0FFFF","6" = "#4682B4"), name="groups")
print(scatter_plot)

#x4-Y1
scatter_plot <- ggplot(as.data.frame(data_train), aes(x = x4, y = Y, color = clusters_train)) +
  geom_point(alpha = 0.6) +
  labs(title = "plot X4-Y", x = "X4", y = "Y") +
  scale_color_manual(values = c("1" = "blue","2" = "darkblue","3" = "#87CEFA",  "5"="darkgray","4" = "#E0FFFF","6" = "#4682B4"), name="groups")
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
k0 <- 8
#m <- colMeans(data_rob[,c(2,3)])
#b0 <- apply(data_rob[,c(2,3)], 2, var)
#binary covariates
a <- 0.1
b <- 0.1
#categorical covariates
categs_1 <- NULL
alpha_1 <- rep(0.1, length(categs_1))

g_params <- list(fun = "g0", lambda = 1, alpha = 1, cov = invcov_cont)
g_params <- list(fun = "g1", lambda = 1, alpha = 1, cov = invcov_cont)
g_params <- list(fun = "g2", lambda = 0.7, alpha = 1, cov = invcov_cont)
g_params <- list(fun = "g3", lambda = 0.03, alpha = 1, cov = invcov_cont)
g_params <- list("ppmx_n", c1, c2, a, b, categs_1, alpha_1)
g_params <- list("ppmx_t", a0, k0, a, b, categs_1, alpha_1)


P0_params <- list(mu0 = mean(Y_sample), lambda0 = 0.01, a0 = 1, b0 = 15)
ngg_params <- list(sigma = 0.2, k = 0.3)
niter <- 3000
nburn <- 2000
thin <- 1
nGibbs <- 0
thinGibbs <-1
var_type <- c(1,1,0,0)

system.time(result_0 <- run_mcmc(data=as.matrix(data_sample), type=var_type, g_params=g_params, 
                                 ngg_params=ngg_params, P0_params=P0_params, niter=niter, 
                                 nburn=nburn, thin=thin, nGibbs = nGibbs, thin_Gibbs = thinGibbs))

freq_chain<-t(result_0$freq_chain)[-1,]
matplot(freq_chain, type = "l", xlab = "Iteration", main = " Gibbs (3000, 0, 1) with G0",ylim = c(0, 1), col = "black")

labels <- as.factor(result_0$best_clus_binder)
table(labels)
histo_by_group <- ggplot(as.data.frame(data_sample), aes(x = Y, fill = labels)) +
  geom_histogram(binwidth = 1, position = "identity", alpha = 0.7, color = "black") +
  labs(title = "Plot", x = "Y", y = "frequency") +
  scale_fill_discrete(name = "groups")
print(histo_by_group)

arandi(clusters_sample, result_0$best_clus_binder)


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
g_params <- list(fun = "g2", lambda = 0.7, alpha = 1, cov = invcov_cont)
time_g2 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g3", lambda = 0.03, alpha = 1, cov = invcov_cont)
time_g3 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list("ppmx_n", c1, c2, a, b, categs_1, alpha_1)
time_ppmx_n <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list("ppmx_t", a0, k0, a, b, categs_1, alpha_1)
time_ppmx_t <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
time_Gibbs <- data.frame(time = c(time_g0, time_g1, time_g2, time_g3,
                                  time_ppmx_n, time_ppmx_t))
rownames(time_Gibbs) <- c("g0", "g1", "g2", "g3", "ppmx_n", "ppmx_t")



nGibbs <- 1
thinGibbs <- 10
itermean <- 5
g_params <- list(fun = "g0", lambda = 1, alpha = 1, cov = invcov_cont)
time_g0 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g1", lambda = 1, alpha = 1, cov = invcov_cont)
time_g1 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g2", lambda = 0.7, alpha = 1, cov = invcov_cont)
time_g2 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g3", lambda = 0.03, alpha = 1, cov = invcov_cont)
time_g3 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list("ppmx_n", c1, c2, a, b, categs_1, alpha_1)
time_ppmx_n <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list("ppmx_t", a0, k0, a, b, categs_1, alpha_1)
time_ppmx_t <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
time_Split_1_10 <- data.frame(time = c(time_g0, time_g1, time_g2, time_g3,
                                       time_ppmx_n, time_ppmx_t))
rownames(time_Split_1_10) <- c("g0", "g1", "g2", "g3", "ppmx_n", "ppmx_t")


nGibbs <- 0
thinGibbs <- 10
itermean <- 5
g_params <- list(fun = "g0", lambda = 1, alpha = 1, cov = invcov_cont)
time_g0 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g1", lambda = 1, alpha = 1, cov = invcov_cont)
time_g1 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g2", lambda = 0.7, alpha = 1, cov = invcov_cont)
time_g2 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g3", lambda = 0.03, alpha = 1, cov = invcov_cont)
time_g3 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list("ppmx_n", c1, c2, a, b, categs_1, alpha_1)
time_ppmx_n <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list("ppmx_t", a0, k0, a, b, categs_1, alpha_1)
time_ppmx_t <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
time_Split_0_10 <- data.frame(time = c(time_g0, time_g1, time_g2, time_g3,
                                       time_ppmx_n, time_ppmx_t))
rownames(time_Split_0_10) <- c("g0", "g1", "g2", "g3", "ppmx_n", "ppmx_t")


nGibbs <- 0
thinGibbs <- 5
itermean <- 5
g_params <- list(fun = "g0", lambda = 1, alpha = 1, cov = invcov_cont)
time_g0 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g1", lambda = 0.7, alpha = 1, cov = invcov_cont)
time_g1 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g2", lambda = 1, alpha = 1, cov = invcov_cont)
time_g2 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g3", lambda = 0.03, alpha = 1, cov = invcov_cont)
time_g3 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list("ppmx_n", c1, c2, a, b, categs_1, alpha_1)
time_ppmx_n <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list("ppmx_t", a0, k0, a, b, categs_1, alpha_1)
time_ppmx_t <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
time_Split_0_5 <- data.frame(time = c(time_g0, time_g1, time_g2, time_g3,
                                      time_ppmx_n, time_ppmx_t))
rownames(time_Split_0_5) <- c("g0", "g1", "g2", "g3", "ppmx_n", "ppmx_t")


Gibbs <- cbind(comb_Gibbs, time_Gibbs)
rownames(Gibbs)<- c("G0", "G1", "G2", "G3", "N-N", "N-NIG")
colnames(Gibbs)<- c("ARI","LPML","RMSE", "#clus", "time (sec)")
penultima_posizione <- ncol(comb_Split_0_10)-2

Split_0_10 <- cbind(comb_Split_0_10[1:penultima_posizione], time = time_Split_0_10$time, comb_Split_0_10[(penultima_posizione+1):ncol(comb_Split_0_10)])
rownames(Split_0_10)<- c("G0", "G1", "G2", "G3", "N-N", "N-NIG")
colnames(Split_0_10)<- c("ARI","LPML","RMSE", "#clus", "time (sec)", "accept_rate split", "accept_rate merge")

Split_1_10 <- cbind(comb_Split_1_10[1:penultima_posizione], time = time_Split_1_10$time, comb_Split_1_10[(penultima_posizione+1):ncol(comb_Split_1_10)])
rownames(Split_1_10)<- c("G0", "G1", "G2", "G3", "N-N", "N-NIG")
colnames(Split_1_10)<- c("ARI","LPML","RMSE", "#clus", "time (sec)", "accept_rate split", "accept_rate merge")

Split_0_5 <- cbind(comb_Split_0_5[1:penultima_posizione], time = time_Split_0_5$time, comb_Split_0_5[(penultima_posizione+1):ncol(comb_Split_0_5)])
rownames(Split_0_5)<- c("G0", "G1", "G2", "G3", "N-N", "N-NIG")
colnames(Split_0_5)<- c("ARI","LPML","RMSE", "#clus", "time (sec)", "accept_rate split", "accept_rate merge")

Gibbs <- round(Gibbs,3)
Split_1_10 <- round(Split_1_10, 3)
Split_0_5 <- round(Split_0_5, 3)
Split_0_10 <- round(Split_0_10, 3)


save.image("regression.RData")
#PREDICTION
grid_y1 <- seq(min(Y_train), max(Y_train), by = 0.3)
grid_y2 <- c(0)
niter <- 2000
nburn <- 1000
thin <- 1
nGibbs <- 0
thinGibbs <-1

g_params <- list(fun = "g1", lambda = 1, alpha = 1, cov = invcov_cont)
g_params <- list(fun = "g2", lambda = 1, alpha = 1, cov = invcov_cont)
g_params <- list(fun = "g3", lambda = 0.03, alpha = 1, cov = invcov_cont)
g_params <- list(fun = "g0", lambda = 1, alpha = 1, cov = invcov_cont)
g_params <- list("ppmx_n", c1, c2, a, b, categs_1, alpha_1)
g_params <- list("ppmx_t", a0, k0, a, b, categs_1, alpha_1)

result_pred <- run_mcmc_pred(as.matrix(data_sample), var_type, g_params, ngg_params,
                             P0_params, Y_test[-c(1,2,3,4)], as.matrix(X_test[-c(1,2,3,4),]), grid_y1, niter, nburn, thin, nGibbs, thinGibbs,
                             TRUE, FALSE, FALSE, TRUE)

ggplot(data.frame(x = grid_y1, y = c(colMeans(result_pred$pred_grid)), 
                  ylow = apply(result_pred$pred_grid, 2, quantile, p = 0.05), 
                  yup = apply(result_pred$pred_grid, 2, quantile, p = 0.95))) +
  theme_bw() +
  geom_line(aes(x = x, y = y)) + 
  xlab("y") +
  ylab("density") +
  ggtitle("Predictive distribution with G0") + 
  geom_vline(aes(xintercept = Y_test[5]), lty = 2, col = 2) +
  geom_ribbon(aes(x = x, ymin = ylow, ymax = yup), alpha = 0.25) +
  ylim(c(0, 0.30))+
  xlim(c(-38,10))


#METRICSS
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
g_params <- list(fun = "g2", lambda = 0.7, alpha = 1, cov = invcov_cont)
metrics_2 <- compute_metrics(as.matrix(data_train), clusters_train, var_type, grid_y1, grid_y2,
                             g_params, ngg_params, P0_params, niter, nburn, thin,itermean, nGibbs, thinGibbs,
                             FALSE,FALSE,TRUE,TRUE)
g_params <- list(fun = "g3", lambda = 0.03, alpha = 1, cov = invcov_cont)
metrics_3 <- compute_metrics(as.matrix(data_train), clusters_train, var_type, grid_y1, grid_y2,
                             g_params, ngg_params, P0_params, niter, nburn, thin,itermean, nGibbs, thinGibbs,
                             FALSE,FALSE,TRUE,TRUE)

itermean <- 10
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

nGibbs <- 1
thinGibbs <- 10
itermean <- 30
g_params <- list(fun = "g0", lambda = 1, alpha = 1, cov = invcov_cont)
metrics_0 <- compute_metrics(as.matrix(data_train), clusters_train, var_type, grid_y1, grid_y2,
                             g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs,
                             FALSE,FALSE,TRUE,TRUE)
g_params <- list(fun = "g1", lambda = 1, alpha = 1, cov = invcov_cont)
metrics_1 <- compute_metrics(as.matrix(data_train), clusters_train, var_type, grid_y1, grid_y2,
                             g_params, ngg_params, P0_params, niter, nburn, thin,itermean, nGibbs, thinGibbs,
                             FALSE,FALSE,TRUE,TRUE)
g_params <- list(fun = "g2", lambda = 0.7, alpha = 1, cov = invcov_cont)
metrics_2 <- compute_metrics(as.matrix(data_train), clusters_train, var_type, grid_y1, grid_y2,
                             g_params, ngg_params, P0_params, niter, nburn, thin,itermean, nGibbs, thinGibbs,
                             FALSE,FALSE,TRUE,TRUE)
g_params <- list(fun = "g3", lambda = 0.03, alpha = 1, cov = invcov_cont)
metrics_3 <- compute_metrics(as.matrix(data_train), clusters_train, var_type, grid_y1, grid_y2,
                             g_params, ngg_params, P0_params, niter, nburn, thin,itermean, nGibbs, thinGibbs,
                             FALSE,FALSE,TRUE,TRUE)
itermean <- 10
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
comb_Split_1_10 <- rbind(df1, df2, df3, df4, df5, df6)  
rownames(comb_Split_1_10) <- c("g0", "g1", "g2", "g3", "ppmx_n", "ppmx_t")

nGibbs <- 0
thinGibbs <- 5
itermean <- 30
g_params <- list(fun = "g0", lambda = 1, alpha = 1, cov = invcov_cont)
metrics_0 <- compute_metrics(as.matrix(data_train), clusters_train, var_type, grid_y1, grid_y2,
                             g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs,
                             FALSE,FALSE,TRUE,TRUE)
g_params <- list(fun = "g1", lambda = 1, alpha = 1, cov = invcov_cont)
metrics_1 <- compute_metrics(as.matrix(data_train), clusters_train, var_type, grid_y1, grid_y2,
                             g_params, ngg_params, P0_params, niter, nburn, thin,itermean, nGibbs, thinGibbs,
                             FALSE,FALSE,TRUE,TRUE)
g_params <- list(fun = "g2", lambda = 0.7, alpha = 1, cov = invcov_cont)
metrics_2 <- compute_metrics(as.matrix(data_train), clusters_train, var_type, grid_y1, grid_y2,
                             g_params, ngg_params, P0_params, niter, nburn, thin,itermean, nGibbs, thinGibbs,
                             FALSE,FALSE,TRUE,TRUE)
g_params <- list(fun = "g3", lambda = 0.03, alpha = 1, cov = invcov_cont)
metrics_3 <- compute_metrics(as.matrix(data_train), clusters_train, var_type, grid_y1, grid_y2,
                             g_params, ngg_params, P0_params, niter, nburn, thin,itermean, nGibbs, thinGibbs,
                             FALSE,FALSE,TRUE,TRUE)
itermean <- 10
g_params <- list("ppmx_n", c1, c2, a, b, categs_1, alpha_1)
metrics_n <- compute_metrics(as.matrix(data_train), clusters_train, var_type, grid_y1, grid_y2,
                             g_params, ngg_params, P0_params, niter, nburn, thin,itermean, nGibbs, thinGibbs,
                             FALSE,FALSE,TRUE,TRUE)
g_params <- list("ppmx_t",a0, k0,a, b, categs_1, alpha_1)
metrics_t <- compute_metrics(as.matrix(data_train), clusters_train, var_type, grid_y1, grid_y2,
                             g_params, ngg_params, P0_params, niter, nburn, thin,itermean, nGibbs, thinGibbs,
                             FALSE,FALSE,TRUE,TRUE)
df1 <- data.frame(metrics_0)
df2 <- data.frame(metrics_1)
df3 <- data.frame(metrics_2)
df4 <- data.frame(metrics_3)
df5 <- data.frame(metrics_n)
df6 <- data.frame(metrics_t)
comb_Split_0_5 <- rbind(df1, df2, df3, df4, df5, df6)  
rownames(comb_Split_0_5) <- c("g0", "g1", "g2", "g3", "ppmx_n", "ppmx_t")

nGibbs <- 0
thinGibbs <- 10
itermean <- 30
g_params <- list(fun = "g0", lambda = 1, alpha = 1, cov = invcov_cont)
metrics_0 <- compute_metrics(as.matrix(data_train), clusters_train, var_type, grid_y1, grid_y2,
                             g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs,
                             FALSE,FALSE,TRUE,TRUE)
g_params <- list(fun = "g1", lambda = 1, alpha = 1, cov = invcov_cont)
metrics_1 <- compute_metrics(as.matrix(data_train), clusters_train, var_type, grid_y1, grid_y2,
                             g_params, ngg_params, P0_params, niter, nburn, thin,itermean, nGibbs, thinGibbs,
                             FALSE,FALSE,TRUE,TRUE)
g_params <- list(fun = "g2", lambda = 0.7, alpha = 1, cov = invcov_cont)
metrics_2 <- compute_metrics(as.matrix(data_train), clusters_train, var_type, grid_y1, grid_y2,
                             g_params, ngg_params, P0_params, niter, nburn, thin,itermean, nGibbs, thinGibbs,
                             FALSE,FALSE,TRUE,TRUE)
g_params <- list(fun = "g3", lambda = 0.03, alpha = 1, cov = invcov_cont)
metrics_3 <- compute_metrics(as.matrix(data_train), clusters_train, var_type, grid_y1, grid_y2,
                             g_params, ngg_params, P0_params, niter, nburn, thin,itermean, nGibbs, thinGibbs,
                             FALSE,FALSE,TRUE,TRUE)
itermean <- 10
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
comb_Split_0_10 <- rbind(df1, df2, df3, df4, df5, df6)  
rownames(comb_Split_0_10) <- c("g0", "g1", "g2", "g3","ppmx_n", "ppmx_t")


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
g_params <- list(fun = "g2", lambda = 0.7, alpha = 1, cov = invcov_cont)
time_g2 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g3", lambda = 0.03, alpha = 1, cov = invcov_cont)
time_g3 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list("ppmx_n", c1, c2, a, b, categs_1, alpha_1)
time_ppmx_n <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list("ppmx_t", a0, k0, a, b, categs_1, alpha_1)
time_ppmx_t <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
time_Gibbs <- data.frame(time = c(time_g0, time_g1, time_g2, time_g3,
                                  time_ppmx_n, time_ppmx_t))
rownames(time_Gibbs) <- c("g0", "g1", "g2", "g3", "ppmx_n", "ppmx_t")



nGibbs <- 1
thinGibbs <- 10
itermean <- 5
g_params <- list(fun = "g0", lambda = 1, alpha = 1, cov = invcov_cont)
time_g0 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g1", lambda = 1, alpha = 1, cov = invcov_cont)
time_g1 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g2", lambda = 0.7, alpha = 1, cov = invcov_cont)
time_g2 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g3", lambda = 0.03, alpha = 1, cov = invcov_cont)
time_g3 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list("ppmx_n", c1, c2, a, b, categs_1, alpha_1)
time_ppmx_n <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list("ppmx_t", a0, k0, a, b, categs_1, alpha_1)
time_ppmx_t <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
time_Split_1_10 <- data.frame(time = c(time_g0, time_g1, time_g2, time_g3,
                                       time_ppmx_n, time_ppmx_t))
rownames(time_Split_1_10) <- c("g0", "g1", "g2", "g3", "ppmx_n", "ppmx_t")


nGibbs <- 0
thinGibbs <- 10
itermean <- 5
g_params <- list(fun = "g0", lambda = 1, alpha = 1, cov = invcov_cont)
time_g0 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g1", lambda = 1, alpha = 1, cov = invcov_cont)
time_g1 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g2", lambda = 0.7, alpha = 1, cov = invcov_cont)
time_g2 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g3", lambda = 0.03, alpha = 1, cov = invcov_cont)
time_g3 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list("ppmx_n", c1, c2, a, b, categs_1, alpha_1)
time_ppmx_n <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list("ppmx_t", a0, k0, a, b, categs_1, alpha_1)
time_ppmx_t <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
time_Split_0_10 <- data.frame(time = c(time_g0, time_g1, time_g2, time_g3,
                                       time_ppmx_n, time_ppmx_t))
rownames(time_Split_0_10) <- c("g0", "g1", "g2", "g3", "ppmx_n", "ppmx_t")


nGibbs <- 0
thinGibbs <- 5
itermean <- 5
g_params <- list(fun = "g0", lambda = 1, alpha = 1, cov = invcov_cont)
time_g0 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g1", lambda = 0.7, alpha = 1, cov = invcov_cont)
time_g1 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g2", lambda = 1, alpha = 1, cov = invcov_cont)
time_g2 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g3", lambda = 0.03, alpha = 1, cov = invcov_cont)
time_g3 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list("ppmx_n", c1, c2, a, b, categs_1, alpha_1)
time_ppmx_n <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list("ppmx_t", a0, k0, a, b, categs_1, alpha_1)
time_ppmx_t <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
time_Split_0_5 <- data.frame(time = c(time_g0, time_g1, time_g2, time_g3,
                                      time_ppmx_n, time_ppmx_t))
rownames(time_Split_0_5) <- c("g0", "g1", "g2", "g3", "ppmx_n", "ppmx_t")


Gibbs <- cbind(comb_Gibbs, time_Gibbs)
rownames(Gibbs)<- c("G0", "G1", "G2", "G3", "N-N", "N-NIG")
colnames(Gibbs)<- c("ARI","LPML","RMSE", "#clus", "time (sec)")
penultima_posizione <- ncol(comb_Split_0_10)-2

Split_0_10 <- cbind(comb_Split_0_10[1:penultima_posizione], time = time_Split_0_10$time, comb_Split_0_10[(penultima_posizione+1):ncol(comb_Split_0_10)])
rownames(Split_0_10)<- c("G0", "G1", "G2", "G3", "N-N", "N-NIG")
colnames(Split_0_10)<- c("ARI","LPML","RMSE", "#clus", "time (sec)", "accept_rate split", "accept_rate merge")

Split_1_10 <- cbind(comb_Split_1_10[1:penultima_posizione], time = time_Split_1_10$time, comb_Split_1_10[(penultima_posizione+1):ncol(comb_Split_1_10)])
rownames(Split_1_10)<- c("G0", "G1", "G2", "G3", "N-N", "N-NIG")
colnames(Split_1_10)<- c("ARI","LPML","RMSE", "#clus", "time (sec)", "accept_rate split", "accept_rate merge")

Split_0_5 <- cbind(comb_Split_0_5[1:penultima_posizione], time = time_Split_0_5$time, comb_Split_0_5[(penultima_posizione+1):ncol(comb_Split_0_5)])
rownames(Split_0_5)<- c("G0", "G1", "G2", "G3", "N-N", "N-NIG")
colnames(Split_0_5)<- c("ARI","LPML","RMSE", "#clus", "time (sec)", "accept_rate split", "accept_rate merge")

Gibbs <- round(Gibbs,3)
Split_1_10 <- round(Split_1_10, 3)
Split_0_5 <- round(Split_0_5, 3)
Split_0_10 <- round(Split_0_10, 3)

 #PARAMETRI OK - pc mammma sta andando #SU PC MAMMA STA ANDANDO #OK
##### 500,500,500 -----
set.seed(123)
#TOT 1500
group_1 <- cbind(rmvnorm(n=3775, mean=c(-3,3),sigma= diag(.5, 2)), 
                 sample(c(0,1), size = 3775, prob = c(0.9, 0.1), replace = T),
                 sample(c(0,1), size = 3775, prob = c(0.9, 0.1), replace = T))
group_1 <- cbind(apply(group_1, 1, function(x) rnorm(1, 1 + x %*% c(5, 2, 1, 0), .5)), group_1)

group_2<- cbind(rmvnorm(3775, c(0,0), diag(.5, 2)), 
                sample(c(0,1), size = 3775, prob = c(0.5, 0.5), replace = T),
                sample(c(0,1), size = 3775, prob = c(0.5, 0.5), replace = T))
group_2 <- cbind(apply(group_2, 1, function(x) rnorm(1, 4 + x %*% c(2, -2, 1, -1), .5)), group_2)

group_3 <- cbind(rmvnorm(3775, c(3,3), diag(.5, 2)), 
                 sample(c(0,1), size = 3775, prob = c(0.1, 0.9), replace = T),
                 sample(c(0,1), size = 3775, prob = c(0.1, 0.9), replace = T))
group_3 <- cbind(apply(group_3, 1, function(x) rnorm(1,  -1 + x %*% c(-5, -2, -1, 1), .5)), group_3)

data <- rbind(group_1, group_2, group_3)
new_column_names <- c("Y", "X1", "x2", "x3", "x4")
colnames(data) <- new_column_names

#scaling
data[,c(2,3)] <- scale(data[,c(2,3)])
clusters <- factor(rep(1:3, c(3775, 3775, 3775)))
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
sampled_train <- sample(nrow(data_train), 1500)  #cambio
data_sample <- data_train[sampled_train, ]
Y_sample <- data_sample[,1]
X_sample <- data_sample[,c(2,3,4,5)]
clusters_sample <- clusters_train[sampled_train]

histo_by_group <- ggplot(data.frame(Y_train, clusters_train), aes(x = Y_train, fill = clusters_train)) +
  geom_histogram(binwidth = 1, position = "identity", alpha = 0.7,color = "black")+
  labs(title = "Plot", x = "Y", y = "frequence") +
  scale_fill_manual(values = c("1" = "blue","2" = "darkblue","3" = "#87CEFA"), name = "groups")
print(histo_by_group)

col <-c("1" = "lightblue",  
        "2" = "darkblue",  
        "3" = "darkgray",  
        "4" = "lightgray",   
        "5" = "#E0FFFF",
        "6" = "#4682B4",
        "7" = "#87CEFA",
        "8" = "#87CEEB")
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
k0 <- 4
#m <- colMeans(data_rob[,c(2,3)])
#b0 <- apply(data_rob[,c(2,3)], 2, var)
#binary covariates
a <- 0.1
b <- 0.1
#categorical covariates
categs_1 <- NULL
alpha_1 <- rep(0.1, length(categs_1))

g_params <- list(fun = "g0", lambda = 1, alpha = 1, cov = invcov_cont)
g_params <- list(fun = "g1", lambda = 0.2, alpha = 1, cov = invcov_cont)
g_params <- list(fun = "g2", lambda = 1, alpha = 1, cov = invcov_cont)
g_params <- list(fun = "g3", lambda = 0.01, alpha = 1, cov = invcov_cont)
g_params <- list("ppmx_n", c1, c2, a, b, categs_1, alpha_1)
g_params <- list("ppmx_t", a0, k0, a, b, categs_1, alpha_1)


P0_params <- list(mu0 = mean(Y_sample), lambda0 = 0.01, a0 = 1, b0 = 15)
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

png("500_gibbs_NN.png", width=416, height = 313)
freq_chain<-t(result_0$freq_chain)[-1,]
freq_chain<-freq_chain[,-3]
matplot(freq_chain, type = "l", xlab = "", ylab = "", main = "Gibbs - NN", ylim = c(0, 1), col = "black", axes = FALSE)
axis(1)
axis(2, at = seq(0.0, 1.0, by = 0.2), labels = format(seq(0.0, 1.0, by = 0.2), nsmall = 1), las = 1)
box()
dev.off()

labels <- as.factor(result_0$best_clus_binder)
table(labels)
histo_by_group <- ggplot(as.data.frame(data_sample), aes(x = Y, fill = labels)) +
  geom_histogram(binwidth = 1, position = "identity", alpha = 0.7, color = "black") +
  labs(title = "Plot", x = "Y", y = "frequency") +
  scale_fill_discrete(name = "groups")
print(histo_by_group)

arandi(clusters_sample, result_0$best_clus_binder)

niter <- 3000
nburn <- 2000
thinGibbs <- 1
nGibbs <- 0
thin <- 1
itermean <- 5
g_params <- list(fun = "g0", lambda = 1, alpha = 1, cov = invcov_cont)
time_g0 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g1", lambda = 0.2, alpha = 1, cov = invcov_cont)
time_g1 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g2", lambda = 1, alpha = 1, cov = invcov_cont)
time_g2 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g3", lambda = 0.01, alpha = 1, cov = invcov_cont)
time_g3 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list("ppmx_n", c1, c2, a, b, categs_1, alpha_1)
time_ppmx_n <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list("ppmx_t", a0, k0, a, b, categs_1, alpha_1)
time_ppmx_t <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
time_Gibbs <- data.frame(time = c(time_g0, time_g1, time_g2, time_g3,
                                  time_ppmx_n, time_ppmx_t))
rownames(time_Gibbs) <- c("g0", "g1", "g2", "g3", "ppmx_n", "ppmx_t")



nGibbs <- 1
thinGibbs <- 10
itermean <- 5
g_params <- list(fun = "g0", lambda = 1, alpha = 1, cov = invcov_cont)
time_g0 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g1", lambda = 0.2, alpha = 1, cov = invcov_cont)
time_g1 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g2", lambda = 1, alpha = 1, cov = invcov_cont)
time_g2 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g3", lambda = 0.01, alpha = 1, cov = invcov_cont)
time_g3 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list("ppmx_n", c1, c2, a, b, categs_1, alpha_1)
time_ppmx_n <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list("ppmx_t", a0, k0, a, b, categs_1, alpha_1)
time_ppmx_t <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
time_Split_1_10 <- data.frame(time = c(time_g0, time_g1, time_g2, time_g3,
                                       time_ppmx_n, time_ppmx_t))
rownames(time_Split_1_10) <- c("g0", "g1", "g2", "g3", "ppmx_n", "ppmx_t")


nGibbs <- 0
thinGibbs <- 10
itermean <- 5
g_params <- list(fun = "g0", lambda = 1, alpha = 1, cov = invcov_cont)
time_g0 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g1", lambda = 0.2, alpha = 1, cov = invcov_cont)
time_g1 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g2", lambda = 1, alpha = 1, cov = invcov_cont)
time_g2 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g3", lambda = 0.01, alpha = 1, cov = invcov_cont)
time_g3 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list("ppmx_n", c1, c2, a, b, categs_1, alpha_1)
time_ppmx_n <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list("ppmx_t", a0, k0, a, b, categs_1, alpha_1)
time_ppmx_t <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
time_Split_0_10 <- data.frame(time = c(time_g0, time_g1, time_g2, time_g3,
                                       time_ppmx_n, time_ppmx_t))
rownames(time_Split_0_10) <- c("g0", "g1", "g2", "g3", "ppmx_n", "ppmx_t")


nGibbs <- 0
thinGibbs <- 5
itermean <- 5
g_params <- list(fun = "g0", lambda = 1, alpha = 1, cov = invcov_cont)
time_g0 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g1", lambda = 0.2, alpha = 1, cov = invcov_cont)
time_g1 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g2", lambda = 1, alpha = 1, cov = invcov_cont)
time_g2 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g3", lambda = 0.01, alpha = 1, cov = invcov_cont)
time_g3 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list("ppmx_n", c1, c2, a, b, categs_1, alpha_1)
time_ppmx_n <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list("ppmx_t", a0, k0, a, b, categs_1, alpha_1)
time_ppmx_t <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
time_Split_0_5 <- data.frame(time = c(time_g0, time_g1, time_g2, time_g3,
                                      time_ppmx_n, time_ppmx_t))
rownames(time_Split_0_5) <- c("g0", "g1", "g2", "g3", "ppmx_n", "ppmx_t")


Gibbs <- cbind(comb_Gibbs, time_Gibbs)
rownames(Gibbs)<- c("G0", "G1", "G2", "G3", "N-N", "N-NIG")
colnames(Gibbs)<- c("ARI","LPML","RMSE", "#clus", "time (sec)")
penultima_posizione <- ncol(comb_Split_0_10)-2

Split_0_10 <- cbind(comb_Split_0_10[1:penultima_posizione], time = time_Split_0_10$time, comb_Split_0_10[(penultima_posizione+1):ncol(comb_Split_0_10)])
rownames(Split_0_10)<- c("G0", "G1", "G2", "G3", "N-N", "N-NIG")
colnames(Split_0_10)<- c("ARI","LPML","RMSE", "#clus", "time (sec)", "accept_rate split", "accept_rate merge")

Split_1_10 <- cbind(comb_Split_1_10[1:penultima_posizione], time = time_Split_1_10$time, comb_Split_1_10[(penultima_posizione+1):ncol(comb_Split_1_10)])
rownames(Split_1_10)<- c("G0", "G1", "G2", "G3", "N-N", "N-NIG")
colnames(Split_1_10)<- c("ARI","LPML","RMSE", "#clus", "time (sec)", "accept_rate split", "accept_rate merge")

Split_0_5 <- cbind(comb_Split_0_5[1:penultima_posizione], time = time_Split_0_5$time, comb_Split_0_5[(penultima_posizione+1):ncol(comb_Split_0_5)])
rownames(Split_0_5)<- c("G0", "G1", "G2", "G3", "N-N", "N-NIG")
colnames(Split_0_5)<- c("ARI","LPML","RMSE", "#clus", "time (sec)", "accept_rate split", "accept_rate merge")

Gibbs <- round(Gibbs,3)
Split_1_10 <- round(Split_1_10, 3)
Split_0_5 <- round(Split_0_5, 3)
Split_0_10 <- round(Split_0_10, 3)


save.image("regression.RData")
#PREDICTION
grid_y1 <- seq(min(Y_train), max(Y_train), by = 0.3)
grid_y2 <- c(0)
niter <- 3000
nburn <- 100
thin <- 1
nGibbs <- 0
thinGibbs <-1

g_params <- list(fun = "g1", lambda = 1, alpha = 1, cov = invcov_cont)
g_params <- list(fun = "g2", lambda = 1, alpha = 1, cov = invcov_cont)
g_params <- list(fun = "g3", lambda = 0.03, alpha = 1, cov = invcov_cont)
g_params <- list(fun = "g0", lambda = 1, alpha = 1, cov = invcov_cont)
g_params <- list("ppmx_n", c1, c2, a, b, categs_1, alpha_1)
g_params <- list("ppmx_t", a0, k0, a, b, categs_1, alpha_1)

system.time(result_pred <- run_mcmc_pred(as.matrix(data_sample), var_type, g_params, ngg_params,
                             P0_params, Y_test[], as.matrix(X_test[,]), grid_y1, niter, nburn, thin, nGibbs, thinGibbs,
                             FALSE, FALSE, TRUE, TRUE))
#rmse2 
#nclus
#arandi(cluster_Sample, result_Red$best_clus_bind)
#LPML_new(as.matrix(data_sample), type, result_ped$clus_chain, P0_params)

ggplot(data.frame(x = grid_y1, y = c(colMeans(result_pred$pred_grid)), 
                  ylow = apply(result_pred$pred_grid, 2, quantile, p = 0.05), 
                  yup = apply(result_pred$pred_grid, 2, quantile, p = 0.95))) +
  theme_bw() +
  geom_line(aes(x = x, y = y)) + 
  xlab("y") +
  ylab("density") +
  ggtitle("Predictive distribution with G0") + 
  geom_vline(aes(xintercept = Y_test[5]), lty = 2, col = 2) +
  geom_ribbon(aes(x = x, ymin = ylow, ymax = yup), alpha = 0.25) +
  ylim(c(0, 0.30))+
  xlim(c(-38,10))


#METRICSS
niter <- 3000
nburn <- 2000
thin <- 1

nGibbs <- 0
thinGibbs <- 1
itermean <- 15
g_params <- list(fun = "g0", lambda = 1, alpha = 1, cov = invcov_cont)
metrics_0 <- compute_metrics(as.matrix(data_train), clusters_train, var_type, grid_y1, grid_y2,
                             g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs,
                             FALSE,FALSE,TRUE,TRUE)
g_params <- list(fun = "g1", lambda = 0.2, alpha = 1, cov = invcov_cont)
metrics_1 <- compute_metrics(as.matrix(data_train), clusters_train, var_type, grid_y1, grid_y2,
                             g_params, ngg_params, P0_params, niter, nburn, thin,itermean, nGibbs, thinGibbs,
                             FALSE,FALSE,TRUE,TRUE)
g_params <- list(fun = "g2", lambda = 1, alpha = 1, cov = invcov_cont)
metrics_2 <- compute_metrics(as.matrix(data_train), clusters_train, var_type, grid_y1, grid_y2,
                             g_params, ngg_params, P0_params, niter, nburn, thin,itermean, nGibbs, thinGibbs,
                             FALSE,FALSE,TRUE,TRUE)
g_params <- list(fun = "g3", lambda = 0.01, alpha = 1, cov = invcov_cont)
metrics_3 <- compute_metrics(as.matrix(data_train), clusters_train, var_type, grid_y1, grid_y2,
                             g_params, ngg_params, P0_params, niter, nburn, thin,itermean, nGibbs, thinGibbs,
                             FALSE,FALSE,TRUE,TRUE)

df1 <- data.frame(metrics_0)
df2 <- data.frame(metrics_1)
df3 <- data.frame(metrics_2)
df4 <- data.frame(metrics_3)

comb_Gibbs <- rbind(df1, df2, df3, df4)  
rownames(comb_Gibbs) <- c("g0", "g1", "g2", "g3")

nGibbs <- 1
thinGibbs <- 10
itermean <- 15
g_params <- list(fun = "g0", lambda = 1, alpha = 1, cov = invcov_cont)
metrics_0 <- compute_metrics(as.matrix(data_train), clusters_train, var_type, grid_y1, grid_y2,
                             g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs,
                             FALSE,FALSE,TRUE,TRUE)
g_params <- list(fun = "g1", lambda = 0.2, alpha = 1, cov = invcov_cont)
metrics_1 <- compute_metrics(as.matrix(data_train), clusters_train, var_type, grid_y1, grid_y2,
                             g_params, ngg_params, P0_params, niter, nburn, thin,itermean, nGibbs, thinGibbs,
                             FALSE,FALSE,TRUE,TRUE)
g_params <- list(fun = "g2", lambda = 1, alpha = 1, cov = invcov_cont)
metrics_2 <- compute_metrics(as.matrix(data_train), clusters_train, var_type, grid_y1, grid_y2,
                             g_params, ngg_params, P0_params, niter, nburn, thin,itermean, nGibbs, thinGibbs,
                             FALSE,FALSE,TRUE,TRUE)
g_params <- list(fun = "g3", lambda = 0.01, alpha = 1, cov = invcov_cont)
metrics_3 <- compute_metrics(as.matrix(data_train), clusters_train, var_type, grid_y1, grid_y2,
                             g_params, ngg_params, P0_params, niter, nburn, thin,itermean, nGibbs, thinGibbs,
                             FALSE,FALSE,TRUE,TRUE)

df1 <- data.frame(metrics_0)
df2 <- data.frame(metrics_1)
df3 <- data.frame(metrics_2)
df4 <- data.frame(metrics_3)

comb_Split_1_10 <- rbind(df1, df2, df3, df4)  
rownames(comb_Split_1_10) <- c("g0", "g1", "g2", "g3")

nGibbs <- 0
thinGibbs <- 5
itermean <- 15
g_params <- list(fun = "g0", lambda = 1, alpha = 1, cov = invcov_cont)
metrics_0 <- compute_metrics(as.matrix(data_train), clusters_train, var_type, grid_y1, grid_y2,
                             g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs,
                             FALSE,FALSE,TRUE,TRUE)
g_params <- list(fun = "g1", lambda = 0.2, alpha = 1, cov = invcov_cont)
metrics_1 <- compute_metrics(as.matrix(data_train), clusters_train, var_type, grid_y1, grid_y2,
                             g_params, ngg_params, P0_params, niter, nburn, thin,itermean, nGibbs, thinGibbs,
                             FALSE,FALSE,TRUE,TRUE)
g_params <- list(fun = "g2", lambda = 1, alpha = 1, cov = invcov_cont)
metrics_2 <- compute_metrics(as.matrix(data_train), clusters_train, var_type, grid_y1, grid_y2,
                             g_params, ngg_params, P0_params, niter, nburn, thin,itermean, nGibbs, thinGibbs,
                             FALSE,FALSE,TRUE,TRUE)
g_params <- list(fun = "g3", lambda = 0.01, alpha = 1, cov = invcov_cont)
metrics_3 <- compute_metrics(as.matrix(data_train), clusters_train, var_type, grid_y1, grid_y2,
                             g_params, ngg_params, P0_params, niter, nburn, thin,itermean, nGibbs, thinGibbs,
                             FALSE,FALSE,TRUE,TRUE)

df1 <- data.frame(metrics_0)
df2 <- data.frame(metrics_1)
df3 <- data.frame(metrics_2)
df4 <- data.frame(metrics_3)

comb_Split_0_5 <- rbind(df1, df2, df3, df4)  
rownames(comb_Split_0_5) <- c("g0", "g1", "g2", "g3")

nGibbs <- 0
thinGibbs <- 10
itermean <- 15
g_params <- list(fun = "g0", lambda = 1, alpha = 1, cov = invcov_cont)
metrics_0 <- compute_metrics(as.matrix(data_train), clusters_train, var_type, grid_y1, grid_y2,
                             g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs,
                             FALSE,FALSE,TRUE,TRUE)
g_params <- list(fun = "g1", lambda = 0.2, alpha = 1, cov = invcov_cont)
metrics_1 <- compute_metrics(as.matrix(data_train), clusters_train, var_type, grid_y1, grid_y2,
                             g_params, ngg_params, P0_params, niter, nburn, thin,itermean, nGibbs, thinGibbs,
                             FALSE,FALSE,TRUE,TRUE)
g_params <- list(fun = "g2", lambda = 1, alpha = 1, cov = invcov_cont)
metrics_2 <- compute_metrics(as.matrix(data_train), clusters_train, var_type, grid_y1, grid_y2,
                             g_params, ngg_params, P0_params, niter, nburn, thin,itermean, nGibbs, thinGibbs,
                             FALSE,FALSE,TRUE,TRUE)
g_params <- list(fun = "g3", lambda = 0.01, alpha = 1, cov = invcov_cont)
metrics_3 <- compute_metrics(as.matrix(data_train), clusters_train, var_type, grid_y1, grid_y2,
                             g_params, ngg_params, P0_params, niter, nburn, thin,itermean, nGibbs, thinGibbs,
                             FALSE,FALSE,TRUE,TRUE)

df1 <- data.frame(metrics_0)
df2 <- data.frame(metrics_1)
df3 <- data.frame(metrics_2)
df4 <- data.frame(metrics_3)

comb_Split_0_10 <- rbind(df1, df2, df3, df4)  
rownames(comb_Split_0_10) <- c("g0", "g1", "g2", "g3")

niter <- 3000
nburn <- 2000
thinGibbs <- 1
nGibbs <- 0
thin <- 1
itermean <- 5
g_params <- list(fun = "g0", lambda = 1, alpha = 1, cov = invcov_cont)
time_g0 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g1", lambda = 0.2, alpha = 1, cov = invcov_cont)
time_g1 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g2", lambda = 1, alpha = 1, cov = invcov_cont)
time_g2 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g3", lambda = 0.01, alpha = 1, cov = invcov_cont)
time_g3 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)

time_Gibbs <- data.frame(time = c(time_g0, time_g1, time_g2, time_g3))
rownames(time_Gibbs) <- c("g0", "g1", "g2", "g3")



nGibbs <- 1
thinGibbs <- 10
itermean <- 5
g_params <- list(fun = "g0", lambda = 1, alpha = 1, cov = invcov_cont)
time_g0 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g1", lambda = 0.2, alpha = 1, cov = invcov_cont)
time_g1 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g2", lambda = 1, alpha = 1, cov = invcov_cont)
time_g2 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g3", lambda = 0.01, alpha = 1, cov = invcov_cont)
time_g3 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)

time_Split_1_10 <- data.frame(time = c(time_g0, time_g1, time_g2, time_g3))
rownames(time_Split_1_10) <- c("g0", "g1", "g2", "g3")


nGibbs <- 0
thinGibbs <- 10
itermean <- 5
g_params <- list(fun = "g0", lambda = 1, alpha = 1, cov = invcov_cont)
time_g0 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g1", lambda = 0.2, alpha = 1, cov = invcov_cont)
time_g1 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g2", lambda = 1, alpha = 1, cov = invcov_cont)
time_g2 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g3", lambda = 0.01, alpha = 1, cov = invcov_cont)
time_g3 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)

time_Split_0_10 <- data.frame(time = c(time_g0, time_g1, time_g2, time_g3))
rownames(time_Split_0_10) <- c("g0", "g1", "g2", "g3")


nGibbs <- 0
thinGibbs <- 5
itermean <- 5
g_params <- list(fun = "g0", lambda = 1, alpha = 1, cov = invcov_cont)
time_g0 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g1", lambda = 0.2, alpha = 1, cov = invcov_cont)
time_g1 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g2", lambda = 1, alpha = 1, cov = invcov_cont)
time_g2 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g3", lambda = 0.01, alpha = 1, cov = invcov_cont)
time_g3 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)

time_Split_0_5 <- data.frame(time = c(time_g0, time_g1, time_g2, time_g3))
rownames(time_Split_0_5) <- c("g0", "g1", "g2", "g3")


Gibbs <- cbind(comb_Gibbs, time_Gibbs)
rownames(Gibbs)<- c("G0", "G1", "G2", "G3")
colnames(Gibbs)<- c("ARI","LPML","RMSE", "#clus", "time (sec)")
penultima_posizione <- ncol(comb_Split_0_10)-2

Split_0_10 <- cbind(comb_Split_0_10[1:penultima_posizione], time = time_Split_0_10$time, comb_Split_0_10[(penultima_posizione+1):ncol(comb_Split_0_10)])
rownames(Split_0_10)<- c("G0", "G1", "G2", "G3")
colnames(Split_0_10)<- c("ARI","LPML","RMSE", "#clus", "time (sec)", "accept_rate split", "accept_rate merge")

Split_1_10 <- cbind(comb_Split_1_10[1:penultima_posizione], time = time_Split_1_10$time, comb_Split_1_10[(penultima_posizione+1):ncol(comb_Split_1_10)])
rownames(Split_1_10)<- c("G0", "G1", "G2", "G3")
colnames(Split_1_10)<- c("ARI","LPML","RMSE", "#clus", "time (sec)", "accept_rate split", "accept_rate merge")

Split_0_5 <- cbind(comb_Split_0_5[1:penultima_posizione], time = time_Split_0_5$time, comb_Split_0_5[(penultima_posizione+1):ncol(comb_Split_0_5)])
rownames(Split_0_5)<- c("G0", "G1", "G2", "G3")
colnames(Split_0_5)<- c("ARI","LPML","RMSE", "#clus", "time (sec)", "accept_rate split", "accept_rate merge")

Gibbs <- round(Gibbs,3)
Split_1_10 <- round(Split_1_10, 3)
Split_0_5 <- round(Split_0_5, 3)
Split_0_10 <- round(Split_0_10, 3)


save.image("regression.RData")
  #VA BENE SOLO G0 #SOLO G0
##### 200,150,100 -----
set.seed(123)
#450 = tot
group_1 <- cbind(rmvnorm(n=4444, mean=c(-3,3),sigma= diag(.5, 2)), 
                 sample(c(0,1), size = 4444, prob = c(0.9, 0.1), replace = T),
                 sample(c(0,1), size = 4444, prob = c(0.9, 0.1), replace = T))
group_1 <- cbind(apply(group_1, 1, function(x) rnorm(1, 1 + x %*% c(5, 2, 1, 0), .5)), group_1)

group_2<- cbind(rmvnorm(3333, c(0,0), diag(.5, 2)), 
                sample(c(0,1), size = 3333, prob = c(0.5, 0.5), replace = T),
                sample(c(0,1), size = 3333, prob = c(0.5, 0.5), replace = T))
group_2 <- cbind(apply(group_2, 1, function(x) rnorm(1, 4 + x %*% c(2, -2, 1, -1), .5)), group_2)

group_3 <- cbind(rmvnorm(2222, c(3,3), diag(.5, 2)), 
                 sample(c(0,1), size = 2222, prob = c(0.1, 0.9), replace = T),
                 sample(c(0,1), size = 2222, prob = c(0.1, 0.9), replace = T))
group_3 <- cbind(apply(group_3, 1, function(x) rnorm(1,  -1 + x %*% c(-5, -2, -1, 1), .5)), group_3)

data <- rbind(group_1, group_2, group_3)
new_column_names <- c("Y", "X1", "x2", "x3", "x4")
colnames(data) <- new_column_names

#scaling
data[,c(2,3)] <- scale(data[,c(2,3)])
clusters <- factor(rep(1:3, c(4444, 3333, 2222)))
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
sampled_train <- sample(nrow(data_train), 450)  #cambio
data_sample <- data_train[sampled_train, ]
Y_sample <- data_sample[,1]
X_sample <- data_sample[,c(2,3,4,5)]
clusters_sample <- clusters_train[sampled_train]

histo_by_group <- ggplot(data.frame(Y_train, clusters_train), aes(x = Y_train, fill = clusters_train)) +
  geom_histogram(binwidth = 1, position = "identity", alpha = 0.7,color = "black")+
  labs(title = "Plot", x = "Y", y = "frequence") +
  scale_fill_manual(values = c("1" = "blue","2" = "darkblue","3" = "#87CEFA"), name = "groups")
print(histo_by_group)

col <-c("1" = "lightblue",  
        "2" = "darkblue",  
        "3" = "darkgray",  
        "4" = "lightgray",   
        "5" = "#E0FFFF",
        "6" = "#4682B4",
        "7" = "#87CEFA",
        "8" = "#87CEEB")
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
a0 <- 8
k0 <- 1
#m <- colMeans(data_rob[,c(2,3)])
#b0 <- apply(data_rob[,c(2,3)], 2, var)
#binary covariates
a <- 0.1
b <- 0.1
#categorical covariates
categs_1 <- NULL
alpha_1 <- rep(0.1, length(categs_1))

g_params <- list(fun = "g0", lambda = 1, alpha = 1, cov = invcov_cont)
g_params <- list(fun = "g1", lambda = 0.5, alpha = 1, cov = invcov_cont)
g_params <- list(fun = "g2", lambda = 1, alpha = 1, cov = invcov_cont)
g_params <- list(fun = "g3", lambda = 0.03, alpha = 1, cov = invcov_cont)
g_params <- list("ppmx_n", c1, c2, a, b, categs_1, alpha_1)
g_params <- list("ppmx_t", a0, k0, a, b, categs_1, alpha_1)


P0_params <- list(mu0 = mean(Y_sample), lambda0 = 0.01, a0 = 1, b0 = 15)
ngg_params <- list(sigma = 0.2, k = 0.3)
niter <- 3000
nburn <- 2000
thin <- 1
nGibbs <- 0
thinGibbs <-10
var_type <- c(1,1,0,0)

system.time(result_0 <- run_mcmc(data=as.matrix(data_sample), type=var_type, g_params=g_params, 
                                 ngg_params=ngg_params, P0_params=P0_params, niter=niter, 
                                 nburn=nburn, thin=thin, nGibbs = nGibbs, thin_Gibbs = thinGibbs))

freq_chain<-t(result_0$freq_chain)[-1,]
matplot(freq_chain, type = "l", xlab = "Iteration", main = " Gibbs (3000, 0, 1) with G0",ylim = c(0, 1), col = "black")

labels <- as.factor(result_0$best_clus_binder)
table(labels)
histo_by_group <- ggplot(as.data.frame(data_sample), aes(x = Y, fill = labels)) +
  geom_histogram(binwidth = 1, position = "identity", alpha = 0.7, color = "black") +
  labs(title = "Plot", x = "Y", y = "frequency") +
  scale_fill_discrete(name = "groups")
print(histo_by_group)

arandi(clusters_sample, result_0$best_clus_binder)


niter <- 3000
nburn <- 2000
thinGibbs <- 1
nGibbs <- 0
thin <- 1
itermean <- 5
g_params <- list(fun = "g0", lambda = 1, alpha = 1, cov = invcov_cont)
time_g0 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g1", lambda = 0.5, alpha = 1, cov = invcov_cont)
time_g1 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g2", lambda = 1, alpha = 1, cov = invcov_cont)
time_g2 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g3", lambda = 0.03, alpha = 1, cov = invcov_cont)
time_g3 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list("ppmx_n", c1, c2, a, b, categs_1, alpha_1)
time_ppmx_n <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list("ppmx_t", a0, k0, a, b, categs_1, alpha_1)
time_ppmx_t <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
time_Gibbs <- data.frame(time = c(time_g0, time_g1, time_g2, time_g3,
                                  time_ppmx_n, time_ppmx_t))
rownames(time_Gibbs) <- c("g0", "g1", "g2", "g3", "ppmx_n", "ppmx_t")



nGibbs <- 1
thinGibbs <- 10
itermean <- 5
g_params <- list(fun = "g0", lambda = 1, alpha = 1, cov = invcov_cont)
time_g0 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g1", lambda = 0.5, alpha = 1, cov = invcov_cont)
time_g1 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g2", lambda = 1, alpha = 1, cov = invcov_cont)
time_g2 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g3", lambda = 0.03, alpha = 1, cov = invcov_cont)
time_g3 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list("ppmx_n", c1, c2, a, b, categs_1, alpha_1)
time_ppmx_n <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list("ppmx_t", a0, k0, a, b, categs_1, alpha_1)
time_ppmx_t <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
time_Split_1_10 <- data.frame(time = c(time_g0, time_g1, time_g2, time_g3,
                                       time_ppmx_n, time_ppmx_t))
rownames(time_Split_1_10) <- c("g0", "g1", "g2", "g3", "ppmx_n", "ppmx_t")


nGibbs <- 0
thinGibbs <- 10
itermean <- 5
g_params <- list(fun = "g0", lambda = 1, alpha = 1, cov = invcov_cont)
time_g0 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g1", lambda = 0.5, alpha = 1, cov = invcov_cont)
time_g1 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g2", lambda = 1, alpha = 1, cov = invcov_cont)
time_g2 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g3", lambda = 0.03, alpha = 1, cov = invcov_cont)
time_g3 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list("ppmx_n", c1, c2, a, b, categs_1, alpha_1)
time_ppmx_n <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list("ppmx_t", a0, k0, a, b, categs_1, alpha_1)
time_ppmx_t <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
time_Split_0_10 <- data.frame(time = c(time_g0, time_g1, time_g2, time_g3,
                                       time_ppmx_n, time_ppmx_t))
rownames(time_Split_0_10) <- c("g0", "g1", "g2", "g3", "ppmx_n", "ppmx_t")


nGibbs <- 0
thinGibbs <- 5
itermean <- 5
g_params <- list(fun = "g0", lambda = 1, alpha = 1, cov = invcov_cont)
time_g0 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g1", lambda = 0.5, alpha = 1, cov = invcov_cont)
time_g1 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g2", lambda = 1, alpha = 1, cov = invcov_cont)
time_g2 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g3", lambda = 0.03, alpha = 1, cov = invcov_cont)
time_g3 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list("ppmx_n", c1, c2, a, b, categs_1, alpha_1)
time_ppmx_n <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list("ppmx_t", a0, k0, a, b, categs_1, alpha_1)
time_ppmx_t <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
time_Split_0_5 <- data.frame(time = c(time_g0, time_g1, time_g2, time_g3,
                                      time_ppmx_n, time_ppmx_t))
rownames(time_Split_0_5) <- c("g0", "g1", "g2", "g3", "ppmx_n", "ppmx_t")


Gibbs <- cbind(comb_Gibbs, time_Gibbs)
rownames(Gibbs)<- c("G0", "G1", "G2", "G3", "N-N", "N-NIG")
colnames(Gibbs)<- c("ARI","LPML","RMSE", "#clus", "time (sec)")
penultima_posizione <- ncol(comb_Split_0_10)-2

Split_0_10 <- cbind(comb_Split_0_10[1:penultima_posizione], time = time_Split_0_10$time, comb_Split_0_10[(penultima_posizione+1):ncol(comb_Split_0_10)])
rownames(Split_0_10)<- c("G0", "G1", "G2", "G3", "N-N", "N-NIG")
colnames(Split_0_10)<- c("ARI","LPML","RMSE", "#clus", "time (sec)", "accept_rate split", "accept_rate merge")

Split_1_10 <- cbind(comb_Split_1_10[1:penultima_posizione], time = time_Split_1_10$time, comb_Split_1_10[(penultima_posizione+1):ncol(comb_Split_1_10)])
rownames(Split_1_10)<- c("G0", "G1", "G2", "G3", "N-N", "N-NIG")
colnames(Split_1_10)<- c("ARI","LPML","RMSE", "#clus", "time (sec)", "accept_rate split", "accept_rate merge")

Split_0_5 <- cbind(comb_Split_0_5[1:penultima_posizione], time = time_Split_0_5$time, comb_Split_0_5[(penultima_posizione+1):ncol(comb_Split_0_5)])
rownames(Split_0_5)<- c("G0", "G1", "G2", "G3", "N-N", "N-NIG")
colnames(Split_0_5)<- c("ARI","LPML","RMSE", "#clus", "time (sec)", "accept_rate split", "accept_rate merge")

Gibbs <- round(Gibbs,3)
Split_1_10 <- round(Split_1_10, 3)
Split_0_5 <- round(Split_0_5, 3)
Split_0_10 <- round(Split_0_10, 3)


save.image("regression.RData")
#PREDICTION
grid_y1 <- seq(min(Y_train), max(Y_train), by = 0.3)
grid_y2 <- c(0)
niter <- 2000
nburn <- 1000
thin <- 1
nGibbs <- 0
thinGibbs <-1

g_params <- list(fun = "g1", lambda = 1, alpha = 1, cov = invcov_cont)
g_params <- list(fun = "g2", lambda = 1, alpha = 1, cov = invcov_cont)
g_params <- list(fun = "g3", lambda = 0.03, alpha = 1, cov = invcov_cont)
g_params <- list(fun = "g0", lambda = 1, alpha = 1, cov = invcov_cont)
g_params <- list("ppmx_n", c1, c2, a, b, categs_1, alpha_1)
g_params <- list("ppmx_t", a0, k0, a, b, categs_1, alpha_1)

result_pred <- run_mcmc_pred(as.matrix(data_sample), var_type, g_params, ngg_params,
                             P0_params, Y_test[-c(1,2,3,4)], as.matrix(X_test[-c(1,2,3,4),]), grid_y1, niter, nburn, thin, nGibbs, thinGibbs,
                             TRUE, FALSE, FALSE, TRUE)

ggplot(data.frame(x = grid_y1, y = c(colMeans(result_pred$pred_grid)), 
                  ylow = apply(result_pred$pred_grid, 2, quantile, p = 0.05), 
                  yup = apply(result_pred$pred_grid, 2, quantile, p = 0.95))) +
  theme_bw() +
  geom_line(aes(x = x, y = y)) + 
  xlab("y") +
  ylab("density") +
  ggtitle("Predictive distribution with G0") + 
  geom_vline(aes(xintercept = Y_test[5]), lty = 2, col = 2) +
  geom_ribbon(aes(x = x, ymin = ylow, ymax = yup), alpha = 0.25) +
  ylim(c(0, 0.30))+
  xlim(c(-38,10))


#METRICSS
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
g_params <- list(fun = "g1", lambda = 0.5, alpha = 1, cov = invcov_cont)
metrics_1 <- compute_metrics(as.matrix(data_train), clusters_train, var_type, grid_y1, grid_y2,
                             g_params, ngg_params, P0_params, niter, nburn, thin,itermean, nGibbs, thinGibbs,
                             FALSE,FALSE,TRUE,TRUE)
g_params <- list(fun = "g2", lambda = 1, alpha = 1, cov = invcov_cont)
metrics_2 <- compute_metrics(as.matrix(data_train), clusters_train, var_type, grid_y1, grid_y2,
                             g_params, ngg_params, P0_params, niter, nburn, thin,itermean, nGibbs, thinGibbs,
                             FALSE,FALSE,TRUE,TRUE)
g_params <- list(fun = "g3", lambda = 0.03, alpha = 1, cov = invcov_cont)
metrics_3 <- compute_metrics(as.matrix(data_train), clusters_train, var_type, grid_y1, grid_y2,
                             g_params, ngg_params, P0_params, niter, nburn, thin,itermean, nGibbs, thinGibbs,
                             FALSE,FALSE,TRUE,TRUE)
itermean <- 10
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

nGibbs <- 1
thinGibbs <- 10
itermean <- 30
g_params <- list(fun = "g0", lambda = 1, alpha = 1, cov = invcov_cont)
metrics_0 <- compute_metrics(as.matrix(data_train), clusters_train, var_type, grid_y1, grid_y2,
                             g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs,
                             FALSE,FALSE,TRUE,TRUE)
g_params <- list(fun = "g1", lambda = 0.5, alpha = 1, cov = invcov_cont)
metrics_1 <- compute_metrics(as.matrix(data_train), clusters_train, var_type, grid_y1, grid_y2,
                             g_params, ngg_params, P0_params, niter, nburn, thin,itermean, nGibbs, thinGibbs,
                             FALSE,FALSE,TRUE,TRUE)
g_params <- list(fun = "g2", lambda = 1, alpha = 1, cov = invcov_cont)
metrics_2 <- compute_metrics(as.matrix(data_train), clusters_train, var_type, grid_y1, grid_y2,
                             g_params, ngg_params, P0_params, niter, nburn, thin,itermean, nGibbs, thinGibbs,
                             FALSE,FALSE,TRUE,TRUE)
g_params <- list(fun = "g3", lambda = 0.03, alpha = 1, cov = invcov_cont)
metrics_3 <- compute_metrics(as.matrix(data_train), clusters_train, var_type, grid_y1, grid_y2,
                             g_params, ngg_params, P0_params, niter, nburn, thin,itermean, nGibbs, thinGibbs,
                             FALSE,FALSE,TRUE,TRUE)
itermean <- 10
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
comb_Split_1_10 <- rbind(df1, df2, df3, df4, df5, df6)  
rownames(comb_Split_1_10) <- c("g0", "g1", "g2", "g3", "ppmx_n", "ppmx_t")

nGibbs <- 0
thinGibbs <- 5
itermean <- 30
g_params <- list(fun = "g0", lambda = 1, alpha = 1, cov = invcov_cont)
metrics_0 <- compute_metrics(as.matrix(data_train), clusters_train, var_type, grid_y1, grid_y2,
                             g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs,
                             FALSE,FALSE,TRUE,TRUE)
g_params <- list(fun = "g1", lambda = 0.5, alpha = 1, cov = invcov_cont)
metrics_1 <- compute_metrics(as.matrix(data_train), clusters_train, var_type, grid_y1, grid_y2,
                             g_params, ngg_params, P0_params, niter, nburn, thin,itermean, nGibbs, thinGibbs,
                             FALSE,FALSE,TRUE,TRUE)
g_params <- list(fun = "g2", lambda = 1, alpha = 1, cov = invcov_cont)
metrics_2 <- compute_metrics(as.matrix(data_train), clusters_train, var_type, grid_y1, grid_y2,
                             g_params, ngg_params, P0_params, niter, nburn, thin,itermean, nGibbs, thinGibbs,
                             FALSE,FALSE,TRUE,TRUE)
g_params <- list(fun = "g3", lambda = 0.03, alpha = 1, cov = invcov_cont)
metrics_3 <- compute_metrics(as.matrix(data_train), clusters_train, var_type, grid_y1, grid_y2,
                             g_params, ngg_params, P0_params, niter, nburn, thin,itermean, nGibbs, thinGibbs,
                             FALSE,FALSE,TRUE,TRUE)
itermean <- 10
g_params <- list("ppmx_n", c1, c2, a, b, categs_1, alpha_1)
metrics_n <- compute_metrics(as.matrix(data_train), clusters_train, var_type, grid_y1, grid_y2,
                             g_params, ngg_params, P0_params, niter, nburn, thin,itermean, nGibbs, thinGibbs,
                             FALSE,FALSE,TRUE,TRUE)
g_params <- list("ppmx_t",a0, k0,a, b, categs_1, alpha_1)
metrics_t <- compute_metrics(as.matrix(data_train), clusters_train, var_type, grid_y1, grid_y2,
                             g_params, ngg_params, P0_params, niter, nburn, thin,itermean, nGibbs, thinGibbs,
                             FALSE,FALSE,TRUE,TRUE)
df1 <- data.frame(metrics_0)
df2 <- data.frame(metrics_1)
df3 <- data.frame(metrics_2)
df4 <- data.frame(metrics_3)
df5 <- data.frame(metrics_n)
df6 <- data.frame(metrics_t)
comb_Split_0_5 <- rbind(df1, df2, df3, df4, df5, df6)  
rownames(comb_Split_0_5) <- c("g0", "g1", "g2", "g3", "ppmx_n", "ppmx_t")

nGibbs <- 0
thinGibbs <- 10
itermean <- 30
g_params <- list(fun = "g0", lambda = 1, alpha = 1, cov = invcov_cont)
metrics_0 <- compute_metrics(as.matrix(data_train), clusters_train, var_type, grid_y1, grid_y2,
                             g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs,
                             FALSE,FALSE,TRUE,TRUE)
g_params <- list(fun = "g1", lambda = 0.5, alpha = 1, cov = invcov_cont)
metrics_1 <- compute_metrics(as.matrix(data_train), clusters_train, var_type, grid_y1, grid_y2,
                             g_params, ngg_params, P0_params, niter, nburn, thin,itermean, nGibbs, thinGibbs,
                             FALSE,FALSE,TRUE,TRUE)
g_params <- list(fun = "g2", lambda = 1, alpha = 1, cov = invcov_cont)
metrics_2 <- compute_metrics(as.matrix(data_train), clusters_train, var_type, grid_y1, grid_y2,
                             g_params, ngg_params, P0_params, niter, nburn, thin,itermean, nGibbs, thinGibbs,
                             FALSE,FALSE,TRUE,TRUE)
g_params <- list(fun = "g3", lambda = 0.03, alpha = 1, cov = invcov_cont)
metrics_3 <- compute_metrics(as.matrix(data_train), clusters_train, var_type, grid_y1, grid_y2,
                             g_params, ngg_params, P0_params, niter, nburn, thin,itermean, nGibbs, thinGibbs,
                             FALSE,FALSE,TRUE,TRUE)
itermean <- 10
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
comb_Split_0_10 <- rbind(df1, df2, df3, df4, df5, df6)  
rownames(comb_Split_0_10) <- c("g0", "g1", "g2", "g3","ppmx_n", "ppmx_t")

niter <- 3000
nburn <- 2000
thinGibbs <- 1
nGibbs <- 0
thin <- 1
itermean <- 5
g_params <- list(fun = "g0", lambda = 1, alpha = 1, cov = invcov_cont)
time_g0 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g1", lambda = 0.5, alpha = 1, cov = invcov_cont)
time_g1 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g2", lambda = 1, alpha = 1, cov = invcov_cont)
time_g2 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g3", lambda = 0.03, alpha = 1, cov = invcov_cont)
time_g3 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list("ppmx_n", c1, c2, a, b, categs_1, alpha_1)
time_ppmx_n <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list("ppmx_t", a0, k0, a, b, categs_1, alpha_1)
time_ppmx_t <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
time_Gibbs <- data.frame(time = c(time_g0, time_g1, time_g2, time_g3,
                                  time_ppmx_n, time_ppmx_t))
rownames(time_Gibbs) <- c("g0", "g1", "g2", "g3", "ppmx_n", "ppmx_t")



nGibbs <- 1
thinGibbs <- 10
itermean <- 5
g_params <- list(fun = "g0", lambda = 1, alpha = 1, cov = invcov_cont)
time_g0 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g1", lambda = 0.5, alpha = 1, cov = invcov_cont)
time_g1 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g2", lambda = 1, alpha = 1, cov = invcov_cont)
time_g2 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g3", lambda = 0.03, alpha = 1, cov = invcov_cont)
time_g3 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list("ppmx_n", c1, c2, a, b, categs_1, alpha_1)
time_ppmx_n <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list("ppmx_t", a0, k0, a, b, categs_1, alpha_1)
time_ppmx_t <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
time_Split_1_10 <- data.frame(time = c(time_g0, time_g1, time_g2, time_g3,
                                       time_ppmx_n, time_ppmx_t))
rownames(time_Split_1_10) <- c("g0", "g1", "g2", "g3", "ppmx_n", "ppmx_t")


nGibbs <- 0
thinGibbs <- 10
itermean <- 5
g_params <- list(fun = "g0", lambda = 1, alpha = 1, cov = invcov_cont)
time_g0 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g1", lambda = 0.5, alpha = 1, cov = invcov_cont)
time_g1 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g2", lambda = 1, alpha = 1, cov = invcov_cont)
time_g2 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g3", lambda = 0.03, alpha = 1, cov = invcov_cont)
time_g3 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list("ppmx_n", c1, c2, a, b, categs_1, alpha_1)
time_ppmx_n <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list("ppmx_t", a0, k0, a, b, categs_1, alpha_1)
time_ppmx_t <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
time_Split_0_10 <- data.frame(time = c(time_g0, time_g1, time_g2, time_g3,
                                       time_ppmx_n, time_ppmx_t))
rownames(time_Split_0_10) <- c("g0", "g1", "g2", "g3", "ppmx_n", "ppmx_t")


nGibbs <- 0
thinGibbs <- 5
itermean <- 5
g_params <- list(fun = "g0", lambda = 1, alpha = 1, cov = invcov_cont)
time_g0 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g1", lambda = 0.5, alpha = 1, cov = invcov_cont)
time_g1 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g2", lambda = 1, alpha = 1, cov = invcov_cont)
time_g2 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g3", lambda = 0.03, alpha = 1, cov = invcov_cont)
time_g3 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list("ppmx_n", c1, c2, a, b, categs_1, alpha_1)
time_ppmx_n <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list("ppmx_t", a0, k0, a, b, categs_1, alpha_1)
time_ppmx_t <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
time_Split_0_5 <- data.frame(time = c(time_g0, time_g1, time_g2, time_g3,
                                      time_ppmx_n, time_ppmx_t))
rownames(time_Split_0_5) <- c("g0", "g1", "g2", "g3", "ppmx_n", "ppmx_t")


Gibbs <- cbind(comb_Gibbs, time_Gibbs)
rownames(Gibbs)<- c("G0", "G1", "G2", "G3", "N-N", "N-NIG")
colnames(Gibbs)<- c("ARI","LPML","RMSE", "#clus", "time (sec)")
penultima_posizione <- ncol(comb_Split_0_10)-2

Split_0_10 <- cbind(comb_Split_0_10[1:penultima_posizione], time = time_Split_0_10$time, comb_Split_0_10[(penultima_posizione+1):ncol(comb_Split_0_10)])
rownames(Split_0_10)<- c("G0", "G1", "G2", "G3", "N-N", "N-NIG")
colnames(Split_0_10)<- c("ARI","LPML","RMSE", "#clus", "time (sec)", "accept_rate split", "accept_rate merge")

Split_1_10 <- cbind(comb_Split_1_10[1:penultima_posizione], time = time_Split_1_10$time, comb_Split_1_10[(penultima_posizione+1):ncol(comb_Split_1_10)])
rownames(Split_1_10)<- c("G0", "G1", "G2", "G3", "N-N", "N-NIG")
colnames(Split_1_10)<- c("ARI","LPML","RMSE", "#clus", "time (sec)", "accept_rate split", "accept_rate merge")

Split_0_5 <- cbind(comb_Split_0_5[1:penultima_posizione], time = time_Split_0_5$time, comb_Split_0_5[(penultima_posizione+1):ncol(comb_Split_0_5)])
rownames(Split_0_5)<- c("G0", "G1", "G2", "G3", "N-N", "N-NIG")
colnames(Split_0_5)<- c("ARI","LPML","RMSE", "#clus", "time (sec)", "accept_rate split", "accept_rate merge")

Gibbs <- round(Gibbs,3)
Split_1_10 <- round(Split_1_10, 3)
Split_0_5 <- round(Split_0_5, 3)
Split_0_10 <- round(Split_0_10, 3)


save.image("450.RData")
 #OK 
##### 50,50,50,50,50,50,50,50,50,50 -----
set.seed(123)
#500 = tot
# Generate 10 clusters with closer centers

n <- 1000
group_1 <- cbind(rmvnorm(n, mean=c(-8, 8), sigma= diag(0.5, 2)), 
                 sample(c(0,1), size = n, prob = c(0.9, 0.1), replace = TRUE),
                 sample(c(0,1), size = n, prob = c(0.9, 0.1), replace = TRUE))
group_1 <- cbind(apply(group_1, 1, function(x) rnorm(1, 1 + x %*% c(5, 2, 1, 0), 0.5)), group_1)

group_2 <- cbind(rmvnorm(n, mean=c(-4, -4), sigma= diag(0.5, 2)), 
                 sample(c(0,1), size = n, prob = c(0.5, 0.5), replace = TRUE),
                 sample(c(0,1), size = n, prob = c(0.5, 0.5), replace = TRUE))
group_2 <- cbind(apply(group_2, 1, function(x) rnorm(1, 4 + x %*% c(2, -2, 1, -1), 0.5)), group_2)

group_3 <- cbind(rmvnorm(n, mean=c(6, 8), sigma= diag(0.5, 2)), 
                 sample(c(0,1), size = n, prob = c(0.1, 0.9), replace = TRUE),
                 sample(c(0,1), size = n, prob = c(0.1, 0.9), replace = TRUE))
group_3 <- cbind(apply(group_3, 1, function(x) rnorm(1, -1 + x %*% c(-5, -2, -1, 1), 0.5)), group_3)

group_4 <- cbind(rmvnorm(n, mean=c(6, -6), sigma= diag(0.5, 2)), 
                 sample(c(0,1), size = n, prob = c(0.7, 0.3), replace = TRUE),
                 sample(c(0,1), size = n, prob = c(0.7, 0.3), replace = TRUE))
group_4 <- cbind(apply(group_4, 1, function(x) rnorm(1, 2 + x %*% c(3, 4, -1, 2), 0.5)), group_4)

group_5 <- cbind(rmvnorm(n, mean=c(7 ,-7), sigma= diag(0.5, 2)), 
                 sample(c(0,1), size = n, prob = c(0.8, 0.2), replace = TRUE),
                 sample(c(0,1), size = n, prob = c(0.8, 0.2), replace = TRUE))
group_5 <- cbind(apply(group_5, 1, function(x) rnorm(1, 3 + x %*% c(2, 1, 1, 0), 0.5)), group_5)

group_6 <- cbind(rmvnorm(n, mean=c(12, 4), sigma= diag(0.5, 2)), 
                 sample(c(0,1), size = n, prob = c(0.6, 0.4), replace = TRUE),
                 sample(c(0,1), size = n, prob = c(0.6, 0.4), replace = TRUE))
group_6 <- cbind(apply(group_6, 1, function(x) rnorm(1, -2 + x %*% c(-4, -3, 1, 0), 0.5)), group_6)

group_7 <- cbind(rmvnorm(1000, mean=c(-20, 6), sigma= diag(.5, 2)), 
                 sample(c(0,1), size = 1000, prob = c(0.85, 0.15), replace = TRUE),
                 sample(c(0,1), size = 1000, prob = c(0.85, 0.15), replace = TRUE))
group_7 <- cbind(apply(group_7, 1, function(x) rnorm(1, 3 + x %*% c(1, 1, 1, 0), .5)), group_7)

group_8 <- cbind(rmvnorm(1000, mean=c(-10, 10), sigma= diag(.5, 2)), 
                 sample(c(0,1), size = 1000, prob = c(0.75, 0.25), replace = TRUE),
                 sample(c(0,1), size = 1000, prob = c(0.75, 0.25), replace = TRUE))
group_8 <- cbind(apply(group_8, 1, function(x) rnorm(1, 2 + x %*% c(2, -2, 1, -1), .5)), group_8)

group_9 <- cbind(rmvnorm(1000, mean=c(-8, 8), sigma= diag(.5, 2)), 
                 sample(c(0,1), size = 1000, prob = c(0.6, 0.4), replace = TRUE),
                 sample(c(0,1), size = 1000, prob = c(0.6, 0.4), replace = TRUE))
group_9 <- cbind(apply(group_9, 1, function(x) rnorm(1, 1 + x %*% c(-2, 2, -1, 1), .5)), group_9)

group_10 <- cbind(rmvnorm(1000, mean=c(6, -6), sigma= diag(.5, 2)), 
                  sample(c(0,1), size = 1000, prob = c(0.9, 0.1), replace = TRUE),
                  sample(c(0,1), size = 1000, prob = c(0.9, 0.1), replace = TRUE))
group_10 <- cbind(apply(group_10, 1, function(x) rnorm(1, 5 + x %*% c(3, 1, 1, -1), .5)), group_10)

# Combine all 10 clusters
data <- rbind(group_1, group_2, group_3, group_4, group_5, group_6, group_7, group_8, group_9, group_10)

new_column_names <- c("Y", "X1", "x2", "x3", "x4")
colnames(data) <- new_column_names

#scaling
data[,c(2,3)] <- scale(data[,c(2,3)])
clusters <- factor(rep(1:10, c(1000, 1000, 1000,1000,1000,1000,1000,1000,1000,1000)))
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
sampled_train <- sample(nrow(data_train), 500)  #cambio
data_sample <- data_train[sampled_train, ]
Y_sample <- data_sample[,1]
X_sample <- data_sample[,c(2,3,4,5)]
clusters_sample <- clusters_train[sampled_train]

histo_by_group <- ggplot(data.frame(Y_train, clusters_train), aes(x = Y_train, fill = clusters_train)) +
  geom_histogram(binwidth = 1, position = "identity", alpha = 0.7,color = "black")+
  labs(title = "Plot", x = "Y", y = "frequence") +
  scale_fill_manual(values = c("1" = "blue","2" = "darkblue","3" = "#87CEFA", "9"="darkgray","6" = "#E0FFFF","10" = "black",  "5" = "#4682B4" ,"4" = "#87CEEB","8" = "lightgray","7"="white"), name = "groups")
print(histo_by_group)

col <-c("1" = "lightblue",  
        "2" = "darkblue",  
        "3" = "darkgray",  
        "4" = "lightgray",   
        "5" = "#E0FFFF",
        "6" = "#4682B4",
        "7" = "#87CEFA",
        "8" = "#87CEEB")
##
#x1-Y
scatter_plot <- ggplot(as.data.frame(data_train), aes(x = X1, y = Y, color = clusters_train)) +
  geom_point(alpha = 0.6) +
  labs(title = "plot X1-Y", x = "X1", y = "Y") +
  scale_color_manual(values = c("1" = "blue","2" = "darkblue","3" = "#87CEFA", "9"="darkgray","6" = "#E0FFFF","10" = "black",  "5" = "#4682B4" ,"4" = "#87CEEB","8" = "lightgray","7"="white"), name="groups")
print(scatter_plot)

#x2-Y1
scatter_plot <- ggplot(as.data.frame(data_train), aes(x = x2, y = Y, color = clusters_train)) +
  geom_point(alpha = 0.6) +
  labs(title = "plot X2-Y", x = "X2", y = "Y") +
  scale_color_manual(values = c("1" = "blue","2" = "darkblue","3" = "#87CEFA", "9"="darkgray","6" = "#E0FFFF","10" = "black",  "5" = "#4682B4" ,"4" = "#87CEEB","8" = "lightgray","7"="white"), name="groups")
print(scatter_plot)

#x3-Y1
scatter_plot <- ggplot(as.data.frame(data_train), aes(x = x3, y = Y, color = clusters_train)) +
  geom_point(alpha = 0.6) +
  labs(title = "plot X3-Y", x = "X3", y = "Y") +
  scale_color_manual(values = c("1" = "blue","2" = "darkblue","3" = "#87CEFA", "9"="darkgray","6" = "#E0FFFF","10" = "black",  "5" = "#4682B4" ,"4" = "#87CEEB","8" = "lightgray","7"="white"), name="groups")
print(scatter_plot)

#x4-Y1
scatter_plot <- ggplot(as.data.frame(data_train), aes(x = x4, y = Y, color = clusters_train)) +
  geom_point(alpha = 0.6) +
  labs(title = "plot X4-Y", x = "X4", y = "Y") +
  scale_color_manual(values = c("1" = "blue","2" = "darkblue","3" = "#87CEFA", "9"="darkgray","6" = "#E0FFFF","10" = "black",  "5" = "#4682B4" ,"4" = "#87CEEB","8" = "lightgray","7"="white"), name="groups")
print(scatter_plot)


#G1/G2/G3
invcov_cont <- solve(cov(data_train[,c(2,3)]))

#PPMX
#continuous covariates
#sigma2 known (prior on mu) (ppmx_n)
c1 <- 1
c2 <- 8
#m0 <- colMeans(data_rob[,c(2,3)])
#S0 <- apply(data_rob[,c(2,3)], 2, var)
#sigma2 unknown (prior on mu and sigma2) (ppmx_t)
a0 <- 1
k0 <- 8
#m <- colMeans(data_rob[,c(2,3)])
#b0 <- apply(data_rob[,c(2,3)], 2, var)
#binary covariates
a <- 0.1
b <- 0.1
#categorical covariates
categs_1 <- NULL
alpha_1 <- rep(0.1, length(categs_1))

g_params <- list(fun = "g0", lambda = 1, alpha = 1, cov = invcov_cont)
g_params <- list(fun = "g1", lambda = 1, alpha = 1, cov = invcov_cont)
g_params <- list(fun = "g2", lambda = 1, alpha = 1, cov = invcov_cont)
g_params <- list(fun = "g3", lambda = 0.05, alpha = 1, cov = invcov_cont)
g_params <- list("ppmx_n", c1, c2, a, b, categs_1, alpha_1)
g_params <- list("ppmx_t", a0, k0, a, b, categs_1, alpha_1)


P0_params <- list(mu0 = mean(Y_sample), lambda0 = 0.01, a0 = 1, b0 = 15)
ngg_params <- list(sigma = 0.2, k = 0.3)
niter <- 3000
nburn <- 2000
thin <- 1
nGibbs <- 0
thinGibbs <-10
var_type <- c(1,1,0,0)

system.time(result_0 <- run_mcmc(data=as.matrix(data_sample), type=var_type, g_params=g_params, 
                                 ngg_params=ngg_params, P0_params=P0_params, niter=niter, 
                                 nburn=nburn, thin=thin, nGibbs = nGibbs, thin_Gibbs = thinGibbs))

freq_chain<-t(result_0$freq_chain)[-1,]
matplot(freq_chain, type = "l", xlab = "Iteration", main = " Gibbs (3000, 0, 1) with G0",ylim = c(0, 1), col = "black")

labels <- as.factor(result_0$best_clus_binder)
table(labels)
histo_by_group <- ggplot(as.data.frame(data_sample), aes(x = Y, fill = labels)) +
  geom_histogram(binwidth = 1, position = "identity", alpha = 0.7, color = "black") +
  labs(title = "Plot", x = "Y", y = "frequency") +
  scale_fill_discrete(name = "groups")
print(histo_by_group)

arandi(clusters_sample, result_0$best_clus_binder)


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
g_params <- list(fun = "g3", lambda = 0.05, alpha = 1, cov = invcov_cont)
time_g3 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list("ppmx_n", c1, c2, a, b, categs_1, alpha_1)
time_ppmx_n <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list("ppmx_t", a0, k0, a, b, categs_1, alpha_1)
time_ppmx_t <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
time_Gibbs <- data.frame(time = c(time_g0, time_g1, time_g2, time_g3,
                                  time_ppmx_n, time_ppmx_t))
rownames(time_Gibbs) <- c("g0", "g1", "g2", "g3", "ppmx_n", "ppmx_t")



nGibbs <- 1
thinGibbs <- 10
itermean <- 5
g_params <- list(fun = "g0", lambda = 1, alpha = 1, cov = invcov_cont)
time_g0 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g1", lambda = 1, alpha = 1, cov = invcov_cont)
time_g1 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g2", lambda = 1, alpha = 1, cov = invcov_cont)
time_g2 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g3", lambda = 0.05, alpha = 1, cov = invcov_cont)
time_g3 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list("ppmx_n", c1, c2, a, b, categs_1, alpha_1)
time_ppmx_n <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list("ppmx_t", a0, k0, a, b, categs_1, alpha_1)
time_ppmx_t <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
time_Split_1_10 <- data.frame(time = c(time_g0, time_g1, time_g2, time_g3,
                                       time_ppmx_n, time_ppmx_t))
rownames(time_Split_1_10) <- c("g0", "g1", "g2", "g3", "ppmx_n", "ppmx_t")


nGibbs <- 0
thinGibbs <- 10
itermean <- 5
g_params <- list(fun = "g0", lambda = 1, alpha = 1, cov = invcov_cont)
time_g0 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g1", lambda = 1, alpha = 1, cov = invcov_cont)
time_g1 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g2", lambda = 1, alpha = 1, cov = invcov_cont)
time_g2 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g3", lambda = 0.05, alpha = 1, cov = invcov_cont)
time_g3 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list("ppmx_n", c1, c2, a, b, categs_1, alpha_1)
time_ppmx_n <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list("ppmx_t", a0, k0, a, b, categs_1, alpha_1)
time_ppmx_t <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
time_Split_0_10 <- data.frame(time = c(time_g0, time_g1, time_g2, time_g3,
                                       time_ppmx_n, time_ppmx_t))
rownames(time_Split_0_10) <- c("g0", "g1", "g2", "g3", "ppmx_n", "ppmx_t")


nGibbs <- 0
thinGibbs <- 5
itermean <- 5
g_params <- list(fun = "g0", lambda = 1, alpha = 1, cov = invcov_cont)
time_g0 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g1", lambda = 1, alpha = 1, cov = invcov_cont)
time_g1 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g2", lambda = 1, alpha = 1, cov = invcov_cont)
time_g2 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g3", lambda = 0.05, alpha = 1, cov = invcov_cont)
time_g3 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list("ppmx_n", c1, c2, a, b, categs_1, alpha_1)
time_ppmx_n <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list("ppmx_t", a0, k0, a, b, categs_1, alpha_1)
time_ppmx_t <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
time_Split_0_5 <- data.frame(time = c(time_g0, time_g1, time_g2, time_g3,
                                      time_ppmx_n, time_ppmx_t))
rownames(time_Split_0_5) <- c("g0", "g1", "g2", "g3", "ppmx_n", "ppmx_t")


Gibbs <- cbind(comb_Gibbs, time_Gibbs)
rownames(Gibbs)<- c("G0", "G1", "G2", "G3", "N-N", "N-NIG")
colnames(Gibbs)<- c("ARI","LPML","RMSE", "#clus", "time (sec)")
penultima_posizione <- ncol(comb_Split_0_10)-2

Split_0_10 <- cbind(comb_Split_0_10[1:penultima_posizione], time = time_Split_0_10$time, comb_Split_0_10[(penultima_posizione+1):ncol(comb_Split_0_10)])
rownames(Split_0_10)<- c("G0", "G1", "G2", "G3", "N-N", "N-NIG")
colnames(Split_0_10)<- c("ARI","LPML","RMSE", "#clus", "time (sec)", "accept_rate split", "accept_rate merge")

Split_1_10 <- cbind(comb_Split_1_10[1:penultima_posizione], time = time_Split_1_10$time, comb_Split_1_10[(penultima_posizione+1):ncol(comb_Split_1_10)])
rownames(Split_1_10)<- c("G0", "G1", "G2", "G3", "N-N", "N-NIG")
colnames(Split_1_10)<- c("ARI","LPML","RMSE", "#clus", "time (sec)", "accept_rate split", "accept_rate merge")

Split_0_5 <- cbind(comb_Split_0_5[1:penultima_posizione], time = time_Split_0_5$time, comb_Split_0_5[(penultima_posizione+1):ncol(comb_Split_0_5)])
rownames(Split_0_5)<- c("G0", "G1", "G2", "G3", "N-N", "N-NIG")
colnames(Split_0_5)<- c("ARI","LPML","RMSE", "#clus", "time (sec)", "accept_rate split", "accept_rate merge")

Gibbs <- round(Gibbs,3)
Split_1_10 <- round(Split_1_10, 3)
Split_0_5 <- round(Split_0_5, 3)
Split_0_10 <- round(Split_0_10, 3)


save.image("regression.RData")
#PREDICTION
grid_y1 <- seq(min(Y_train), max(Y_train), by = 0.3)
grid_y2 <- c(0)
niter <- 2000
nburn <- 1000
thin <- 1
nGibbs <- 0
thinGibbs <-1

g_params <- list(fun = "g1", lambda = 1, alpha = 1, cov = invcov_cont)
g_params <- list(fun = "g2", lambda = 1, alpha = 1, cov = invcov_cont)
g_params <- list(fun = "g3", lambda = 0.03, alpha = 1, cov = invcov_cont)
g_params <- list(fun = "g0", lambda = 1, alpha = 1, cov = invcov_cont)
g_params <- list("ppmx_n", c1, c2, a, b, categs_1, alpha_1)
g_params <- list("ppmx_t", a0, k0, a, b, categs_1, alpha_1)

result_pred <- run_mcmc_pred(as.matrix(data_sample), var_type, g_params, ngg_params,
                             P0_params, Y_test[-c(1,2,3,4)], as.matrix(X_test[-c(1,2,3,4),]), grid_y1, niter, nburn, thin, nGibbs, thinGibbs,
                             TRUE, FALSE, FALSE, TRUE)

ggplot(data.frame(x = grid_y1, y = c(colMeans(result_pred$pred_grid)), 
                  ylow = apply(result_pred$pred_grid, 2, quantile, p = 0.05), 
                  yup = apply(result_pred$pred_grid, 2, quantile, p = 0.95))) +
  theme_bw() +
  geom_line(aes(x = x, y = y)) + 
  xlab("y") +
  ylab("density") +
  ggtitle("Predictive distribution with G0") + 
  geom_vline(aes(xintercept = Y_test[5]), lty = 2, col = 2) +
  geom_ribbon(aes(x = x, ymin = ylow, ymax = yup), alpha = 0.25) +
  ylim(c(0, 0.30))+
  xlim(c(-38,10))


#METRICSS
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
g_params <- list(fun = "g3", lambda = 0.05, alpha = 1, cov = invcov_cont)
metrics_3 <- compute_metrics(as.matrix(data_train), clusters_train, var_type, grid_y1, grid_y2,
                             g_params, ngg_params, P0_params, niter, nburn, thin,itermean, nGibbs, thinGibbs,
                             FALSE,FALSE,TRUE,TRUE)
itermean <- 10
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

nGibbs <- 1
thinGibbs <- 10
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
g_params <- list(fun = "g3", lambda = 0.05, alpha = 1, cov = invcov_cont)
metrics_3 <- compute_metrics(as.matrix(data_train), clusters_train, var_type, grid_y1, grid_y2,
                             g_params, ngg_params, P0_params, niter, nburn, thin,itermean, nGibbs, thinGibbs,
                             FALSE,FALSE,TRUE,TRUE)
itermean <- 10
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
comb_Split_1_10 <- rbind(df1, df2, df3, df4, df5, df6)  
rownames(comb_Split_1_10) <- c("g0", "g1", "g2", "g3", "ppmx_n", "ppmx_t")

nGibbs <- 0
thinGibbs <- 5
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
g_params <- list(fun = "g3", lambda = 0.05, alpha = 1, cov = invcov_cont)
metrics_3 <- compute_metrics(as.matrix(data_train), clusters_train, var_type, grid_y1, grid_y2,
                             g_params, ngg_params, P0_params, niter, nburn, thin,itermean, nGibbs, thinGibbs,
                             FALSE,FALSE,TRUE,TRUE)
itermean <- 10
g_params <- list("ppmx_n", c1, c2, a, b, categs_1, alpha_1)
metrics_n <- compute_metrics(as.matrix(data_train), clusters_train, var_type, grid_y1, grid_y2,
                             g_params, ngg_params, P0_params, niter, nburn, thin,itermean, nGibbs, thinGibbs,
                             FALSE,FALSE,TRUE,TRUE)
g_params <- list("ppmx_t",a0, k0,a, b, categs_1, alpha_1)
metrics_t <- compute_metrics(as.matrix(data_train), clusters_train, var_type, grid_y1, grid_y2,
                             g_params, ngg_params, P0_params, niter, nburn, thin,itermean, nGibbs, thinGibbs,
                             FALSE,FALSE,TRUE,TRUE)
df1 <- data.frame(metrics_0)
df2 <- data.frame(metrics_1)
df3 <- data.frame(metrics_2)
df4 <- data.frame(metrics_3)
df5 <- data.frame(metrics_n)
df6 <- data.frame(metrics_t)
comb_Split_0_5 <- rbind(df1, df2, df3, df4, df5, df6)  
rownames(comb_Split_0_5) <- c("g0", "g1", "g2", "g3", "ppmx_n", "ppmx_t")

nGibbs <- 0
thinGibbs <- 10
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
g_params <- list(fun = "g3", lambda = 0.05, alpha = 1, cov = invcov_cont)
metrics_3 <- compute_metrics(as.matrix(data_train), clusters_train, var_type, grid_y1, grid_y2,
                             g_params, ngg_params, P0_params, niter, nburn, thin,itermean, nGibbs, thinGibbs,
                             FALSE,FALSE,TRUE,TRUE)
itermean <- 10
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
comb_Split_0_10 <- rbind(df1, df2, df3, df4, df5, df6)  
rownames(comb_Split_0_10) <- c("g0", "g1", "g2", "g3","ppmx_n", "ppmx_t")

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
g_params <- list(fun = "g3", lambda = 0.05, alpha = 1, cov = invcov_cont)
time_g3 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list("ppmx_n", c1, c2, a, b, categs_1, alpha_1)
time_ppmx_n <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list("ppmx_t", a0, k0, a, b, categs_1, alpha_1)
time_ppmx_t <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
time_Gibbs <- data.frame(time = c(time_g0, time_g1, time_g2, time_g3,
                                  time_ppmx_n, time_ppmx_t))
rownames(time_Gibbs) <- c("g0", "g1", "g2", "g3", "ppmx_n", "ppmx_t")



nGibbs <- 1
thinGibbs <- 10
itermean <- 5
g_params <- list(fun = "g0", lambda = 1, alpha = 1, cov = invcov_cont)
time_g0 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g1", lambda = 1, alpha = 1, cov = invcov_cont)
time_g1 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g2", lambda = 1, alpha = 1, cov = invcov_cont)
time_g2 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g3", lambda = 0.05, alpha = 1, cov = invcov_cont)
time_g3 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list("ppmx_n", c1, c2, a, b, categs_1, alpha_1)
time_ppmx_n <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list("ppmx_t", a0, k0, a, b, categs_1, alpha_1)
time_ppmx_t <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
time_Split_1_10 <- data.frame(time = c(time_g0, time_g1, time_g2, time_g3,
                                       time_ppmx_n, time_ppmx_t))
rownames(time_Split_1_10) <- c("g0", "g1", "g2", "g3", "ppmx_n", "ppmx_t")


nGibbs <- 0
thinGibbs <- 10
itermean <- 5
g_params <- list(fun = "g0", lambda = 1, alpha = 1, cov = invcov_cont)
time_g0 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g1", lambda = 1, alpha = 1, cov = invcov_cont)
time_g1 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g2", lambda = 1, alpha = 1, cov = invcov_cont)
time_g2 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g3", lambda = 0.05, alpha = 1, cov = invcov_cont)
time_g3 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list("ppmx_n", c1, c2, a, b, categs_1, alpha_1)
time_ppmx_n <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list("ppmx_t", a0, k0, a, b, categs_1, alpha_1)
time_ppmx_t <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
time_Split_0_10 <- data.frame(time = c(time_g0, time_g1, time_g2, time_g3,
                                       time_ppmx_n, time_ppmx_t))
rownames(time_Split_0_10) <- c("g0", "g1", "g2", "g3", "ppmx_n", "ppmx_t")


nGibbs <- 0
thinGibbs <- 5
itermean <- 5
g_params <- list(fun = "g0", lambda = 1, alpha = 1, cov = invcov_cont)
time_g0 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g1", lambda = 1, alpha = 1, cov = invcov_cont)
time_g1 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g2", lambda = 1, alpha = 1, cov = invcov_cont)
time_g2 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g3", lambda = 0.05, alpha = 1, cov = invcov_cont)
time_g3 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list("ppmx_n", c1, c2, a, b, categs_1, alpha_1)
time_ppmx_n <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list("ppmx_t", a0, k0, a, b, categs_1, alpha_1)
time_ppmx_t <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
time_Split_0_5 <- data.frame(time = c(time_g0, time_g1, time_g2, time_g3,
                                      time_ppmx_n, time_ppmx_t))
rownames(time_Split_0_5) <- c("g0", "g1", "g2", "g3", "ppmx_n", "ppmx_t")


Gibbs <- cbind(comb_Gibbs, time_Gibbs)
rownames(Gibbs)<- c("G0", "G1", "G2", "G3", "N-N", "N-NIG")
colnames(Gibbs)<- c("ARI","LPML","RMSE", "#clus", "time (sec)")
penultima_posizione <- ncol(comb_Split_0_10)-2

Split_0_10 <- cbind(comb_Split_0_10[1:penultima_posizione], time = time_Split_0_10$time, comb_Split_0_10[(penultima_posizione+1):ncol(comb_Split_0_10)])
rownames(Split_0_10)<- c("G0", "G1", "G2", "G3", "N-N", "N-NIG")
colnames(Split_0_10)<- c("ARI","LPML","RMSE", "#clus", "time (sec)", "accept_rate split", "accept_rate merge")

Split_1_10 <- cbind(comb_Split_1_10[1:penultima_posizione], time = time_Split_1_10$time, comb_Split_1_10[(penultima_posizione+1):ncol(comb_Split_1_10)])
rownames(Split_1_10)<- c("G0", "G1", "G2", "G3", "N-N", "N-NIG")
colnames(Split_1_10)<- c("ARI","LPML","RMSE", "#clus", "time (sec)", "accept_rate split", "accept_rate merge")

Split_0_5 <- cbind(comb_Split_0_5[1:penultima_posizione], time = time_Split_0_5$time, comb_Split_0_5[(penultima_posizione+1):ncol(comb_Split_0_5)])
rownames(Split_0_5)<- c("G0", "G1", "G2", "G3", "N-N", "N-NIG")
colnames(Split_0_5)<- c("ARI","LPML","RMSE", "#clus", "time (sec)", "accept_rate split", "accept_rate merge")

Gibbs <- round(Gibbs,3)
Split_1_10 <- round(Split_1_10, 3)
Split_0_5 <- round(Split_0_5, 3)
Split_0_10 <- round(Split_0_10, 3)
  #PARAMETRI OK  #PARAMETRI OK #OK
##### 300,300,300,300,300,300,300,300 ----
set.seed(123)

n <- 1225
group_1 <- cbind(rmvnorm(n, mean=c(-8, 8), sigma= diag(0.5, 2)), 
                 sample(c(0,1), size = n, prob = c(0.9, 0.1), replace = TRUE),
                 sample(c(0,1), size = n, prob = c(0.9, 0.1), replace = TRUE))
group_1 <- cbind(apply(group_1, 1, function(x) rnorm(1, 1 + x %*% c(5, 2, 1, 0), 0.5)), group_1)

group_2 <- cbind(rmvnorm(n, mean=c(-4, -4), sigma= diag(0.5, 2)), 
                 sample(c(0,1), size = n, prob = c(0.5, 0.5), replace = TRUE),
                 sample(c(0,1), size = n, prob = c(0.5, 0.5), replace = TRUE))
group_2 <- cbind(apply(group_2, 1, function(x) rnorm(1, 4 + x %*% c(2, -2, 1, -1), 0.5)), group_2)

group_3 <- cbind(rmvnorm(n, mean=c(6, 8), sigma= diag(0.5, 2)), 
                 sample(c(0,1), size = n, prob = c(0.1, 0.9), replace = TRUE),
                 sample(c(0,1), size = n, prob = c(0.1, 0.9), replace = TRUE))
group_3 <- cbind(apply(group_3, 1, function(x) rnorm(1, -1 + x %*% c(-5, -2, -1, 1), 0.5)), group_3)

group_4 <- cbind(rmvnorm(n, mean=c(6, -6), sigma= diag(0.5, 2)), 
                 sample(c(0,1), size = n, prob = c(0.7, 0.3), replace = TRUE),
                 sample(c(0,1), size = n, prob = c(0.7, 0.3), replace = TRUE))
group_4 <- cbind(apply(group_4, 1, function(x) rnorm(1, 2 + x %*% c(3, 4, -1, 2), 0.5)), group_4)

group_5 <- cbind(rmvnorm(n, mean=c(7 ,-7), sigma= diag(0.5, 2)), 
                 sample(c(0,1), size = n, prob = c(0.8, 0.2), replace = TRUE),
                 sample(c(0,1), size = n, prob = c(0.8, 0.2), replace = TRUE))
group_5 <- cbind(apply(group_5, 1, function(x) rnorm(1, 3 + x %*% c(2, 1, 1, 0), 0.5)), group_5)

group_6 <- cbind(rmvnorm(n, mean=c(12, 4), sigma= diag(0.5, 2)), 
                 sample(c(0,1), size = n, prob = c(0.6, 0.4), replace = TRUE),
                 sample(c(0,1), size = n, prob = c(0.6, 0.4), replace = TRUE))
group_6 <- cbind(apply(group_6, 1, function(x) rnorm(1, -2 + x %*% c(-4, -3, 1, 0), 0.5)), group_6)

group_7 <- cbind(rmvnorm(1225, mean=c(8, 8), sigma= diag(.5, 2)), 
                 sample(c(0,1), size = 1225, prob = c(0.85, 0.15), replace = TRUE),
                 sample(c(0,1), size = 1225, prob = c(0.85, 0.15), replace = TRUE))
group_7 <- cbind(apply(group_7, 1, function(x) rnorm(1, 3 + x %*% c(1, 1, 1, 0), .5)), group_7)

group_8 <- cbind(rmvnorm(1225, mean=c(-10, 10), sigma= diag(.5, 2)), 
                 sample(c(0,1), size = 1225, prob = c(0.75, 0.25), replace = TRUE),
                 sample(c(0,1), size = 1225, prob = c(0.75, 0.25), replace = TRUE))
group_8 <- cbind(apply(group_8, 1, function(x) rnorm(1, 2 + x %*% c(2, -2, 1, -1), .5)), group_8)


# Combine all 10 clusters
data <- rbind(group_1, group_2, group_3, group_4, group_5, group_6, group_7, group_8)

new_column_names <- c("Y", "X1", "x2", "x3", "x4")
colnames(data) <- new_column_names

#scaling
data[,c(2,3)] <- scale(data[,c(2,3)])
clusters <- factor(rep(1:8, c(1225, 1225, 1225,1225,1225,1225,1225,1225)))
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
sampled_train <- sample(nrow(data_train), 2400)  #cambio
data_sample <- data_train[sampled_train, ]
Y_sample <- data_sample[,1]
X_sample <- data_sample[,c(2,3,4,5)]
clusters_sample <- clusters_train[sampled_train]

histo_by_group <- ggplot(data.frame(Y_train, clusters_train), aes(x = Y_train, fill = clusters_train)) +
  geom_histogram(binwidth = 1, position = "identity", alpha = 0.7,color = "black")+
  labs(title = "Plot", x = "Y", y = "frequence") +
  scale_fill_manual(values = c("1" = "blue","2" = "darkblue","3" = "lightgray", "4"="darkgray","5" = "#E0FFFF","6" = "#87CEFA","7" = "#4682B4", "8" = "black"), name = "groups")
print(histo_by_group)

col <-c("1" = "lightblue",  
        "2" = "darkblue",  
        "3" = "darkgray",  
        "4" = "lightgray",   
        "5" = "#E0FFFF",
        "6" = "#4682B4",
        "7" = "#87CEFA",
        "8" = "#87CEEB")
##
#x1-Y
scatter_plot <- ggplot(as.data.frame(data_train), aes(x = X1, y = Y, color = clusters_train)) +
  geom_point(alpha = 0.6) +
  labs(title = "plot X1-Y", x = "X1", y = "Y") +
  scale_color_manual(values = c("1" = "blue","2" = "darkblue","3" = "lightgray", "4"="darkgray","5" = "#E0FFFF","6" = "#87CEFA","7" = "#4682B4", "8" = "black"), name="groups")
print(scatter_plot)

#x2-Y1
scatter_plot <- ggplot(as.data.frame(data_train), aes(x = x2, y = Y, color = clusters_train)) +
  geom_point(alpha = 0.6) +
  labs(title = "plot X2-Y", x = "X2", y = "Y") +
  scale_color_manual(values = c("1" = "blue","2" = "darkblue","3" = "lightgray", "4"="darkgray","5" = "#E0FFFF","6" = "#87CEFA","7" = "#4682B4", "8" = "black"), name="groups")
print(scatter_plot)

#x3-Y1
scatter_plot <- ggplot(as.data.frame(data_train), aes(x = x3, y = Y, color = clusters_train)) +
  geom_point(alpha = 0.6) +
  labs(title = "plot X3-Y", x = "X3", y = "Y") +
  scale_color_manual(values = c("1" = "blue","2" = "darkblue","3" = "lightgray", "4"="darkgray","5" = "#E0FFFF","6" = "#87CEFA","7" = "#4682B4", "8" = "black"), name="groups")
print(scatter_plot)

#x4-Y1
scatter_plot <- ggplot(as.data.frame(data_train), aes(x = x4, y = Y, color = clusters_train)) +
  geom_point(alpha = 0.6) +
  labs(title = "plot X4-Y", x = "X4", y = "Y") +
  scale_color_manual(values = c("1" = "blue","2" = "darkblue","3" = "lightgray", "4"="darkgray","5" = "#E0FFFF","6" = "#87CEFA","7" = "#4682B4", "8" = "black"), name="groups")
print(scatter_plot)


#G1/G2/G3
invcov_cont <- solve(cov(data_train[,c(2,3)]))

#PPMX
#continuous covariates
#sigma2 known (prior on mu) (ppmx_n)
c1 <- 1
c2 <- 8
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

g_params <- list(fun = "g0", lambda = 1, alpha = 1, cov = invcov_cont)
g_params <- list(fun = "g1", lambda = 0.7, alpha = 1, cov = invcov_cont)
g_params <- list(fun = "g2", lambda = 1, alpha = 1, cov = invcov_cont)
g_params <- list(fun = "g3", lambda = 0.03, alpha = 1, cov = invcov_cont)
g_params <- list("ppmx_n", c1, c2, a, b, categs_1, alpha_1)
g_params <- list("ppmx_t", a0, k0, a, b, categs_1, alpha_1)


P0_params <- list(mu0 = mean(Y_sample), lambda0 = 0.01, a0 = 1, b0 = 15)
ngg_params <- list(sigma = 0.2, k = 0.3)
niter <- 3000
nburn <- 2000
thin <- 1
nGibbs <- 0
thinGibbs <-10
var_type <- c(1,1,0,0)

system.time(result_0 <- run_mcmc(data=as.matrix(data_sample), type=var_type, g_params=g_params, 
                                 ngg_params=ngg_params, P0_params=P0_params, niter=niter, 
                                 nburn=nburn, thin=thin, nGibbs = nGibbs, thin_Gibbs = thinGibbs))

freq_chain<-t(result_0$freq_chain)[-1,]
matplot(freq_chain, type = "l", xlab = "Iteration", main = " Gibbs (3000, 0, 1) with G0",ylim = c(0, 1), col = "black")

labels <- as.factor(result_0$best_clus_binder)
table(labels)
histo_by_group <- ggplot(as.data.frame(data_sample), aes(x = Y, fill = labels)) +
  geom_histogram(binwidth = 1, position = "identity", alpha = 0.7, color = "black") +
  labs(title = "Plot", x = "Y", y = "frequency") +
  scale_fill_discrete(name = "groups")
print(histo_by_group)

arandi(clusters_sample, result_0$best_clus_binder)


niter <- 3000
nburn <- 2000
thinGibbs <- 1
nGibbs <- 0
thin <- 1
itermean <- 5
g_params <- list(fun = "g0", lambda = 1, alpha = 1, cov = invcov_cont)
time_g0 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g1", lambda = 0.5, alpha = 1, cov = invcov_cont)
time_g1 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g2", lambda = 1, alpha = 1, cov = invcov_cont)
time_g2 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g3", lambda = 0.03, alpha = 1, cov = invcov_cont)
time_g3 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list("ppmx_n", c1, c2, a, b, categs_1, alpha_1)
time_ppmx_n <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list("ppmx_t", a0, k0, a, b, categs_1, alpha_1)
time_ppmx_t <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
time_Gibbs <- data.frame(time = c(time_g0, time_g1, time_g2, time_g3,
                                  time_ppmx_n, time_ppmx_t))
rownames(time_Gibbs) <- c("g0", "g1", "g2", "g3", "ppmx_n", "ppmx_t")



nGibbs <- 1
thinGibbs <- 10
itermean <- 5
g_params <- list(fun = "g0", lambda = 1, alpha = 1, cov = invcov_cont)
time_g0 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g1", lambda = 0.5, alpha = 1, cov = invcov_cont)
time_g1 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g2", lambda = 1, alpha = 1, cov = invcov_cont)
time_g2 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g3", lambda = 0.03, alpha = 1, cov = invcov_cont)
time_g3 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list("ppmx_n", c1, c2, a, b, categs_1, alpha_1)
time_ppmx_n <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list("ppmx_t", a0, k0, a, b, categs_1, alpha_1)
time_ppmx_t <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
time_Split_1_10 <- data.frame(time = c(time_g0, time_g1, time_g2, time_g3,
                                       time_ppmx_n, time_ppmx_t))
rownames(time_Split_1_10) <- c("g0", "g1", "g2", "g3", "ppmx_n", "ppmx_t")


nGibbs <- 0
thinGibbs <- 10
itermean <- 5
g_params <- list(fun = "g0", lambda = 1, alpha = 1, cov = invcov_cont)
time_g0 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g1", lambda = 0.5, alpha = 1, cov = invcov_cont)
time_g1 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g2", lambda = 1, alpha = 1, cov = invcov_cont)
time_g2 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g3", lambda = 0.03, alpha = 1, cov = invcov_cont)
time_g3 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list("ppmx_n", c1, c2, a, b, categs_1, alpha_1)
time_ppmx_n <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list("ppmx_t", a0, k0, a, b, categs_1, alpha_1)
time_ppmx_t <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
time_Split_0_10 <- data.frame(time = c(time_g0, time_g1, time_g2, time_g3,
                                       time_ppmx_n, time_ppmx_t))
rownames(time_Split_0_10) <- c("g0", "g1", "g2", "g3", "ppmx_n", "ppmx_t")


nGibbs <- 0
thinGibbs <- 5
itermean <- 5
g_params <- list(fun = "g0", lambda = 1, alpha = 1, cov = invcov_cont)
time_g0 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g1", lambda = 0.5, alpha = 1, cov = invcov_cont)
time_g1 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g2", lambda = 1, alpha = 1, cov = invcov_cont)
time_g2 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list(fun = "g3", lambda = 0.03, alpha = 1, cov = invcov_cont)
time_g3 <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list("ppmx_n", c1, c2, a, b, categs_1, alpha_1)
time_ppmx_n <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
g_params <- list("ppmx_t", a0, k0, a, b, categs_1, alpha_1)
time_ppmx_t <- time_run(as.matrix(data_train), var_type, g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs)
time_Split_0_5 <- data.frame(time = c(time_g0, time_g1, time_g2, time_g3,
                                      time_ppmx_n, time_ppmx_t))
rownames(time_Split_0_5) <- c("g0", "g1", "g2", "g3", "ppmx_n", "ppmx_t")


Gibbs <- cbind(comb_Gibbs, time_Gibbs)
rownames(Gibbs)<- c("G0", "G1", "G2", "G3", "N-N", "N-NIG")
colnames(Gibbs)<- c("ARI","LPML","RMSE", "#clus", "time (sec)")
penultima_posizione <- ncol(comb_Split_0_10)-2

Split_0_10 <- cbind(comb_Split_0_10[1:penultima_posizione], time = time_Split_0_10$time, comb_Split_0_10[(penultima_posizione+1):ncol(comb_Split_0_10)])
rownames(Split_0_10)<- c("G0", "G1", "G2", "G3", "N-N", "N-NIG")
colnames(Split_0_10)<- c("ARI","LPML","RMSE", "#clus", "time (sec)", "accept_rate split", "accept_rate merge")

Split_1_10 <- cbind(comb_Split_1_10[1:penultima_posizione], time = time_Split_1_10$time, comb_Split_1_10[(penultima_posizione+1):ncol(comb_Split_1_10)])
rownames(Split_1_10)<- c("G0", "G1", "G2", "G3", "N-N", "N-NIG")
colnames(Split_1_10)<- c("ARI","LPML","RMSE", "#clus", "time (sec)", "accept_rate split", "accept_rate merge")

Split_0_5 <- cbind(comb_Split_0_5[1:penultima_posizione], time = time_Split_0_5$time, comb_Split_0_5[(penultima_posizione+1):ncol(comb_Split_0_5)])
rownames(Split_0_5)<- c("G0", "G1", "G2", "G3", "N-N", "N-NIG")
colnames(Split_0_5)<- c("ARI","LPML","RMSE", "#clus", "time (sec)", "accept_rate split", "accept_rate merge")

Gibbs <- round(Gibbs,3)
Split_1_10 <- round(Split_1_10, 3)
Split_0_5 <- round(Split_0_5, 3)
Split_0_10 <- round(Split_0_10, 3)


save.image("regression.RData")
#PREDICTION
grid_y1 <- seq(min(Y_train), max(Y_train), by = 0.3)
grid_y2 <- c(0)
niter <- 3000
nburn <- 1000
thin <- 1
nGibbs <- 0
thinGibbs <-10

g_params <- list(fun = "g1", lambda = 1, alpha = 1, cov = invcov_cont)
g_params <- list(fun = "g2", lambda = 1, alpha = 1, cov = invcov_cont)
g_params <- list(fun = "g3", lambda = 0.03, alpha = 1, cov = invcov_cont)
g_params <- list(fun = "g0", lambda = 1, alpha = 1, cov = invcov_cont)
g_params <- list("ppmx_n", c1, c2, a, b, categs_1, alpha_1)
g_params <- list("ppmx_t", a0, k0, a, b, categs_1, alpha_1)

system.time(result_pred <- run_mcmc_pred(as.matrix(data_sample), var_type, g_params, ngg_params,
                                         P0_params, Y_test[], as.matrix(X_test[,]), grid_y1, niter, nburn, thin, nGibbs, thinGibbs,
                                         FALSE, FALSE, TRUE, TRUE))
#rmse2 
#nclus
arandi(clusters_sample, result_pred$best_clus_binder)
LPML_new(as.matrix(data_sample), var_type, result_pred$clus_chain, P0_params)
result_pred$rmse2
result_pred$n_clust_mean
result_pred$accept_merge
result_pred$accept_split

png("300_split_NN.png", width=416, height = 313)
freq_chain<-t(result_pred$freq_chain)[-1,]
matplot(freq_chain, type = "l", xlab = "", ylab = "", main = "Split-Merge(0,10) - NN", ylim = c(0, 1), col = "black", axes = FALSE)
axis(1)
axis(2, at = seq(0.0, 1.0, by = 0.2), labels = format(seq(0.0, 1.0, by = 0.2), nsmall = 1), las = 1)
box()
dev.off()

ggplot(data.frame(x = grid_y1, y = c(colMeans(result_pred$pred_grid)), 
                  ylow = apply(result_pred$pred_grid, 2, quantile, p = 0.05), 
                  yup = apply(result_pred$pred_grid, 2, quantile, p = 0.95))) +
  theme_bw() +
  geom_line(aes(x = x, y = y)) + 
  xlab("y") +
  ylab("density") +
  ggtitle("Predictive distribution with NN") + 
  geom_vline(aes(xintercept = Y_test[5]), lty = 2, col = 2) +
  geom_ribbon(aes(x = x, ymin = ylow, ymax = yup), alpha = 0.25) +
  ylim(c(0, 0.30))+
  xlim(c(-38,10))


#METRICSS
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
g_params <- list(fun = "g1", lambda = 0.5, alpha = 1, cov = invcov_cont)
metrics_1 <- compute_metrics(as.matrix(data_train), clusters_train, var_type, grid_y1, grid_y2,
                             g_params, ngg_params, P0_params, niter, nburn, thin,itermean, nGibbs, thinGibbs,
                             FALSE,FALSE,TRUE,TRUE)
g_params <- list(fun = "g2", lambda = 1, alpha = 1, cov = invcov_cont)
metrics_2 <- compute_metrics(as.matrix(data_train), clusters_train, var_type, grid_y1, grid_y2,
                             g_params, ngg_params, P0_params, niter, nburn, thin,itermean, nGibbs, thinGibbs,
                             FALSE,FALSE,TRUE,TRUE)
g_params <- list(fun = "g3", lambda = 0.03, alpha = 1, cov = invcov_cont)
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

nGibbs <- 1
thinGibbs <- 10
itermean <- 30
g_params <- list(fun = "g0", lambda = 1, alpha = 1, cov = invcov_cont)
metrics_0 <- compute_metrics(as.matrix(data_train), clusters_train, var_type, grid_y1, grid_y2,
                             g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs,
                             FALSE,FALSE,TRUE,TRUE)
g_params <- list(fun = "g1", lambda = 0.5, alpha = 1, cov = invcov_cont)
metrics_1 <- compute_metrics(as.matrix(data_train), clusters_train, var_type, grid_y1, grid_y2,
                             g_params, ngg_params, P0_params, niter, nburn, thin,itermean, nGibbs, thinGibbs,
                             FALSE,FALSE,TRUE,TRUE)
g_params <- list(fun = "g2", lambda = 1, alpha = 1, cov = invcov_cont)
metrics_2 <- compute_metrics(as.matrix(data_train), clusters_train, var_type, grid_y1, grid_y2,
                             g_params, ngg_params, P0_params, niter, nburn, thin,itermean, nGibbs, thinGibbs,
                             FALSE,FALSE,TRUE,TRUE)
g_params <- list(fun = "g3", lambda = 0.03, alpha = 1, cov = invcov_cont)
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
comb_Split_1_10 <- rbind(df1, df2, df3, df4, df5, df6)  
rownames(comb_Split_1_10) <- c("g0", "g1", "g2", "g3", "ppmx_n", "ppmx_t")

nGibbs <- 0
thinGibbs <- 5
itermean <- 30
g_params <- list(fun = "g0", lambda = 1, alpha = 1, cov = invcov_cont)
metrics_0 <- compute_metrics(as.matrix(data_train), clusters_train, var_type, grid_y1, grid_y2,
                             g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs,
                             FALSE,FALSE,TRUE,TRUE)
g_params <- list(fun = "g1", lambda = 0.5, alpha = 1, cov = invcov_cont)
metrics_1 <- compute_metrics(as.matrix(data_train), clusters_train, var_type, grid_y1, grid_y2,
                             g_params, ngg_params, P0_params, niter, nburn, thin,itermean, nGibbs, thinGibbs,
                             FALSE,FALSE,TRUE,TRUE)
g_params <- list(fun = "g2", lambda = 1, alpha = 1, cov = invcov_cont)
metrics_2 <- compute_metrics(as.matrix(data_train), clusters_train, var_type, grid_y1, grid_y2,
                             g_params, ngg_params, P0_params, niter, nburn, thin,itermean, nGibbs, thinGibbs,
                             FALSE,FALSE,TRUE,TRUE)
g_params <- list(fun = "g3", lambda = 0.03, alpha = 1, cov = invcov_cont)
metrics_3 <- compute_metrics(as.matrix(data_train), clusters_train, var_type, grid_y1, grid_y2,
                             g_params, ngg_params, P0_params, niter, nburn, thin,itermean, nGibbs, thinGibbs,
                             FALSE,FALSE,TRUE,TRUE)
g_params <- list("ppmx_n", c1, c2, a, b, categs_1, alpha_1)
metrics_n <- compute_metrics(as.matrix(data_train), clusters_train, var_type, grid_y1, grid_y2,
                             g_params, ngg_params, P0_params, niter, nburn, thin,itermean, nGibbs, thinGibbs,
                             FALSE,FALSE,TRUE,TRUE)
g_params <- list("ppmx_t",a0, k0,a, b, categs_1, alpha_1)
metrics_t <- compute_metrics(as.matrix(data_train), clusters_train, var_type, grid_y1, grid_y2,
                             g_params, ngg_params, P0_params, niter, nburn, thin,itermean, nGibbs, thinGibbs,
                             FALSE,FALSE,TRUE,TRUE)
df1 <- data.frame(metrics_0)
df2 <- data.frame(metrics_1)
df3 <- data.frame(metrics_2)
df4 <- data.frame(metrics_3)
df5 <- data.frame(metrics_n)
df6 <- data.frame(metrics_t)
comb_Split_0_5 <- rbind(df1, df2, df3, df4, df5, df6)  
rownames(comb_Split_0_5) <- c("g0", "g1", "g2", "g3", "ppmx_n", "ppmx_t")

nGibbs <- 0
thinGibbs <- 10
itermean <- 30
g_params <- list(fun = "g0", lambda = 1, alpha = 1, cov = invcov_cont)
metrics_0 <- compute_metrics(as.matrix(data_train), clusters_train, var_type, grid_y1, grid_y2,
                             g_params, ngg_params, P0_params, niter, nburn, thin, itermean, nGibbs, thinGibbs,
                             FALSE,FALSE,TRUE,TRUE)
g_params <- list(fun = "g1", lambda = 0.5, alpha = 1, cov = invcov_cont)
metrics_1 <- compute_metrics(as.matrix(data_train), clusters_train, var_type, grid_y1, grid_y2,
                             g_params, ngg_params, P0_params, niter, nburn, thin,itermean, nGibbs, thinGibbs,
                             FALSE,FALSE,TRUE,TRUE)
g_params <- list(fun = "g2", lambda = 1, alpha = 1, cov = invcov_cont)
metrics_2 <- compute_metrics(as.matrix(data_train), clusters_train, var_type, grid_y1, grid_y2,
                             g_params, ngg_params, P0_params, niter, nburn, thin,itermean, nGibbs, thinGibbs,
                             FALSE,FALSE,TRUE,TRUE)
g_params <- list(fun = "g3", lambda = 0.03, alpha = 1, cov = invcov_cont)
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
comb_Split_0_10 <- rbind(df1, df2, df3, df4, df5, df6)  
rownames(comb_Split_0_10) <- c("g0", "g1", "g2", "g3","ppmx_n", "ppmx_t") #DA FINIRE PARAMETRI 



