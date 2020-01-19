library(abctools)

#setwd("\\home\\gsnkel001\\master_dissertation\\")

setwd("C:\\My Work Documents\\Dissertation\\pythonCodeVersions\\")

SS_options = t(read.csv("SS_options_test.csv"))
param_options = t(read.csv("param_options_test.csv"))
p_true = t(read.csv("p_true.csv"))
SS_true_true = t(read.csv("SS_true_options_test.csv"))

param_true = as.data.frame(c(0.025, 0.025, 0.15, 100, 10, 0.001))


colnames(param_options) = c("delta", "mu", "alpha", "lamda0", "C_lambda", "delta_S")
rownames(param_true) = c("delta", "mu", "alpha", "lamda0", "C_lambda", "delta_S")


SS_options_sqr = cbind(SS_options, SS_options^2)
write.csv(SS_options_sqr,"SS_options_sqr.csv")
colnames(SS_options_sqr)= c("mean", "std", "Skew", "Kurt", "Hurst", "KS", "diff_percentile", 
                        "mean2", "std2", "Skew2", "Kurt2", "Hurst2", "KS2", "diff_percentile2")
head(SS_options_sqr)

# Generic function for selecting summary statistics in ABC inference
# without scaling
select_summary_stats = selectsumm(SS_true_true, param_options, SS_options, 
                                  #obspar=t(param_true), 
                                  ssmethod = mincrit,
                                  do.err = TRUE, verbose = TRUE)

select_summary_stats$best

save(select_summary_stats, file="AS_summary_stats_small.RData")

a = load("AS_summary_stats_small.RData")
# with scaling
#select_summary_stats$err
select_summary_stats$best


# Summary statistic construction by semi-automatic ABC
saABC(param_options, SS_options, plot = TRUE)$BICs
a = saABC(param_options, cbind(SS_options, SS_options^2), plot=TRUE) ##Variance parameter estimated better
a$B
#saBC_summary$BICs
dim(SS_options)
dim(SS_true_true)

a$B
SS_options[,6]

#dim(SS_options)

a$B0

g
head(param_options)

y = as.data.frame(param_options)$alpha
y
df = as.data.frame(cbind(y, SS_options_sqr))
head(df)

glm0=glm(y~.,data = df)
summary(glm0)
summary(aov(glm0))
head(SS_options_sqr)

resid=residuals(glm0)
fitted=fitted(glm0) # = pi estimates
par(mfrow=c(1,1))
hist(fitted,breaks=30)
par(mfrow=c(1,2))
plot(glm0)
plot(glm0,)
hist(resid, breaks=30)

plot(fitted~y)
