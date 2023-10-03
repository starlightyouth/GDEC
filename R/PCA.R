library("survival")
library("ipred")
library('survminer')
library('ggplot2')
library("tsne")
library(factoextra)
library(cluster)
x_tsne <- read.csv("E:/esca/esca 0.16.csv",header = FALSE)
value <- x_tsne[, -ncol(x_tsne)]
label <- x_tsne[, ncol(x_tsne)]
subc <- as.matrix(label)
subc <- subc[, 1]
x <- as.matrix(value)
subc = t(subc)
subc=as.numeric(subc)
PCA=princomp(x) 
summary(PCA)
fviz_pca_ind(PCA,habillage = subc,geom = "point",addEllipses = T,repel = T,ellipse.level=0.8,legend.position = "none")


