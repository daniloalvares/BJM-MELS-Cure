rm(list=ls())

# Required sources
library(tidyverse)
library(rstanarm)
library(rstan)
library(loo)
library(statmod)
library(ggplot2)
library(survival)
library(survminer)

options(mc.cores = parallel::detectCores())

# Longitudinal data
data <- read.csv("bHCG_D.txt", sep="")

# Data for the Stan model
IDL <- data$Sujeto
y <- data$Y
N <- length(y)
n <- length(unique(IDL))
IDoneobs <- as.numeric(which(table(IDL) == 1))
group <- data$Grupo
time <- data$Tiempo

# Start and stop rows for each woman
start <- stop <- rep(NA,n)
aux <- 0
for(i in 1:n){
  pos <- which(IDL == unique(IDL)[i])
  start[i] <- aux + 1
  stop[i] <- aux + length(pos)
  aux <- stop[i]
}

# Normal group data
data0 <- data %>% filter(Grupo == 0)
ID0 <- data0$Sujeto[!duplicated(data0$Sujeto)]
n0 <- length(ID0)
# Creating right-censured times
tCens <- aggregate(data0$Tiempo, by=list(data0$Sujeto), FUN=tail, n=1)[,2]

# Abnormal group data
data1 <- data %>% filter(Grupo == 1)
ID1 <- data1$Sujeto[!duplicated(data1$Sujeto)]
n1 <- length(ID1)
# Creating interval times
tLeft <- aggregate(data1$Tiempo, by=list(data1$Sujeto), FUN=tail, n=1)[,2]
tRight <- tLeft + 10

# Gauss-Legendre quadrature (15 points)
glq <- gauss.quad(15, kind = "legendre")
xk <- glq$nodes   # nodes
wk <- glq$weights # weights
K <- length(xk)   # K-points

# JOINT MODEL 1
fitJM1 <- stan(file   = "JM1.stan", 
               data   = list(N=N, n=n, n0=n0, n1=n1, IDL=IDL, ID0=ID0, ID1=ID1,
                           y=y, time=time, tCens=tCens, tLeft=tLeft, tRight=tRight,
                           start=start, stop=stop, K=K, xk=xk, wk=wk),
               warmup = 3000,                 
               iter   = 6000,
               chains = 3,
               seed   = 1,
               cores  = getOption("mc.cores",3)) 

print(fitJM1)


# JOINT MODEL 2
fitJM2 <- stan(file   = "JM2.stan",
               data   = list(N=N, n=n, n0=n0, n1=n1, noneobs=length(IDoneobs), IDL=IDL, ID0=ID0, ID1=ID1,
                             IDoneobs=IDoneobs, y=y, time=time, tCens=tCens, tLeft=tLeft, tRight=tRight,
                             start=start, stop=stop, K=K, xk=xk, wk=wk),
               warmup = 3000,                 
               iter   = 6000,
               chains = 3,
               seed   = 1,
               cores  = getOption("mc.cores",3)) 

print(fitJM2)