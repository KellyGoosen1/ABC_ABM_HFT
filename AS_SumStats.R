library(abctools)

#setwd("\\home\\gsnkel001\\master_dissertation\\")

#setwd("C:\\My Work Documents\\Dissertation\\pythonCodeVersions")

SS_options = t(read.csv("SS_options.csv"))
param_options = t(read.csv("param_options.csv"))
p_true = t(read.csv("p_true.csv"))
SS_true_true = t(read.csv("SS_true_options.csv"))

param_true = as.data.frame(c(0.025, 0.025, 0.15, 100, 10, 0.001))

colnames(param_options) = c("delta", "mu", "alpha", "lamda0", "C_lambda", "delta_S")
rownames(param_true) = c("delta", "mu", "alpha", "lamda0", "C_lambda", "delta_S")


# Generic function for selecting summary statistics in ABC inference
# without scaling
select_summary_stats = selectsumm(SS_true_true, param_options, SS_options, 
                                  obspar=t(param_true), ssmethod = mincrit,
                                  do.err = TRUE, verbose = FALSE)

save(select_summary_stats, file="AS_summary_stats.RData")

# with scaling
#select_summary_stats$err


# Summary statistic construction by semi-automatic ABC
#saABC(param_options, SS_options, plot = TRUE)$BICs
#saABC(param_options, cbind(SS_options, SS_options^2), plot=TRUE)$BICs ##Variance parameter estimated better

#saBC_summary$BICs



#dim(SS_options)





