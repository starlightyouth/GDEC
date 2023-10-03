library("survival")
library("ipred")
library("survminer")
library("ggplot2")
library("tsne")
library("factoextra")
library("cluster")

kx2 <- read.csv("E:/demo/datadata/stad_log.csv", row.names = 1)
label <- read.csv("E:/paad stad/stad1 0.0054.csv",header = FALSE)
label <- label[, ncol(label)]
n_z <- ncol(kx2)
time <- kx2[1,]/365
status <- kx2[2,]
time[time == 0] <- 0.001
time <- as.numeric(time)
x <- read.csv("E:/paad stad/stad1 0.0054.csv")
x <- x[, -ncol(x)]
status <- as.numeric(status)
y <- Surv(time, status)
subc <- as.matrix(label)
subc <- subc[, 1]
q2 <- survdiff(y ~ subc)
p.val <- 1 - pchisq(q2$chisq, length(q2$n) - 1)
p.val
kx2 <- as.data.frame(kx2)
fit <- survfit(y ~ subc, data = kx2)
ggsurvplot(fit, surv.median.line = "none", pval = TRUE, legend.title = "cluster label", legend = "none", pval.size = 22,
           font.tickslab = c(40, "bold"), font.legend = c(30, "bold"), font.x = c(40, "bold.italic"), font.y = c(40, "bold.italic"),
           
           )




