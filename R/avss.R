install.packages("factoextra")
library(factoextra)
library(cluster)
setwd("E:/demo/datadata/")  
data <- read.csv("read.csv", skip = 3, row.names = 1,header = FALSE)
data=t(data)
setwd("E:/dataresult/read/")
label <- read.csv("x_tsne.csv",header = FALSE)
label <- label[, ncol(label)]
x <- as.matrix(data)
ss=silhouette(label,dist(x))
avss=mean(ss[,3])

