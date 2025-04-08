# Function for maximum a posteriori (MAP)
MAP_fc <- function(x){
  
  lim.inf <- min(x)-1
  lim.sup <- max(x)+1
  s <- density(x,from=lim.inf,to=lim.sup,bw=0.2)
  n <- length(s$y)
  v1 <- s$y[1:(n-2)]
  v2 <- s$y[2:(n-1)]
  v3 <- s$y[3:n]
  ix <- 1+which((v1<v2)&(v2>v3))
  out <- s$x[which(s$y==max(s$y))]
  
  return(out)
  
}

# Individual weighted residuals (IWRES) function
iwres_fc <- function(fit, y, time, ID, model="JM1"){
  
  N <- length(y)
  
  theta1 <- extract(fit, "theta")$theta[,1]
  theta2 <- extract(fit, "theta")$theta[,2]
  theta3 <- extract(fit, "theta")$theta[,3]
  
  bi1 <- extract(fit, "bi")$bi[,,1]
  bi2 <- extract(fit, "bi")$bi[,,2]
  bi3 <- extract(fit, "bi")$bi[,,3]
  
  a1 <- apply(exp(theta1 + bi1), 2, MAP_fc)
  a2 <- apply(exp(theta2 + bi2), 2, MAP_fc)
  a3 <- apply(exp(theta3 + bi3), 2, MAP_fc)
  
  mu <- rep(NA, N)
  for(j in 1:N){ mu[j] = a1[ID[j]]/(1+exp(-(time[j]-a2[ID[j]])/a3[ID[j]])) }
  
  if(model == "JM1"){
    sigma_e <- MAP_fc(sqrt(extract(fit, "sigma2_e")$sigma2_e))
    iwres <- (y - mu)/sigma_e
  }else{
    theta4 <- extract(fit, "theta")$theta[,4]
    bi4 <- extract(fit, "bi")$bi[,,4]
    sigma_e <- apply(exp(theta4 + bi4), 2, MAP_fc)
    iwres <- (y - mu)/sigma_e[ID]
  }
  
  dta <- data.frame(time=time, iwres=iwres)
  ggplot(data = dta, aes(x = time, y = iwres)) +
    geom_point(aes(x = time, y = iwres), size = 1, color = "black") +
    theme_bw() + geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
    theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
          panel.background = element_blank()) + xlab("Time of pregnancy (in days)") + ylab("IWRES") +
      scale_x_continuous(limits=c(0, 95), breaks = seq(0, 90, 10), expand = c(0, 0.05)) + 
      scale_y_continuous(limits=c(-4, 2.5))
  
  # ggsave("iwres.png", units="in", width=4.5, height=2.5, dpi=300)
  
}


# Cox-Snell residuals
cox_snell_fc <- function(fit, tCens=tCens, tLeft=tLeft, tRight=tRight, ID0=ID0, ID1=ID1, model="JM1"){
  
  # Gauss-Legendre quadrature (15 points)
  glq <- gauss.quad(15, kind = "legendre")
  xk <- glq$nodes   # nodes
  wk <- glq$weights # weights
  K <- length(xk)   # K-points
  
  theta1 <- extract(fit, "theta")$theta[,1]
  theta2 <- extract(fit, "theta")$theta[,2]
  theta3 <- extract(fit, "theta")$theta[,3]
  
  bi1 <- extract(fit, "bi")$bi[,,1]
  bi2 <- extract(fit, "bi")$bi[,,2]
  bi3 <- extract(fit, "bi")$bi[,,3]
  
  a1 <- apply(exp(theta1 + bi1), 2, MAP_fc)
  a2 <- apply(exp(theta2 + bi2), 2, MAP_fc)
  a3 <- apply(exp(theta3 + bi3), 2, MAP_fc)

  lambda <- MAP_fc(extract(fit, "lambda")$lambda)
  phi <- MAP_fc(extract(fit, "phi")$phi)
  
  r_CS <- rep(NA,n0+n1)
  if(model == "JM1"){
    alpha <- MAP_fc(extract(fit, "alpha")$alpha)
    
    # Normal group
    for(i in 1:n0){
      # Hazard function at integration points
      hCens <- rep(NA,K)
      for(k in 1:K){
        hCens[k] <- phi * (tCens[i]/2*(xk[k]+1))^(phi-1) * 
          exp( lambda + alpha * a1[ID0[i]]/(1+exp(-((tCens[i] / 2 * (xk[k] + 1))-a2[ID0[i]])/a3[ID0[i]])) )
      }
      # Cox-Snell residuals
      r_CS[ID0[i]] <- tCens[i] / 2 * sum(wk * hCens)
    }
    
    # Abnormal group
    for(i in 1:n1){
      # Left and right hazard functions at integration points
      hLeft <- rep(NA,K)
      hRight <- rep(NA,K)
      for(k in 1:K){
        hLeft[k] <- phi * (tLeft[i]/2*(xk[k]+1))^(phi-1) * 
          exp( lambda + alpha * a1[ID1[i]]/(1+exp(-((tLeft[i] / 2 * (xk[k] + 1))-a2[ID1[i]])/a3[ID1[i]])) )
        hRight[k] <- phi * (tRight[i]/2*(xk[k]+1))^(phi-1) * 
          exp( lambda + alpha * a1[ID1[i]]/(1+exp(-((tRight[i] / 2 * (xk[k] + 1))-a2[ID1[i]])/a3[ID1[i]])) )
      }
      
      # Left and right survival functions with Gauss-Legendre quadrature
      sLeft <- exp( -tLeft[i] / 2 * sum(wk * hLeft) )
      sRight <- exp( -tRight[i] / 2 * sum(wk * hRight) )
      
      # Cox-Snell residuals (DOI: 10.1111/j.0006-341x.2000.00473.x)
      r_CS[ID1[i]] <- ( sLeft * (1 - log(sLeft)) - sRight * (1 - log(sRight)) ) / (sLeft - sRight)
    }
  }else{
    theta4 <- extract(fit, "theta")$theta[,4]
    bi4 <- extract(fit, "bi")$bi[,,4]
    sigma2_e <- apply(exp(theta4 + bi4)^2, 2, MAP_fc)
    alpha1 <- MAP_fc(extract(fit, "alpha")$alpha[,1])
    alpha2 <- MAP_fc(extract(fit, "alpha")$alpha[,2])
    
    # Normal group
    for(i in 1:n0){
      # Hazard function at integration points
      hCens <- rep(NA,K)
      for(k in 1:K){
        hCens[k] <- phi * (tCens[i]/2*(xk[k]+1))^(phi-1) * 
          exp( lambda + alpha1 * a1[ID0[i]]/(1+exp(-((tCens[i] / 2 * (xk[k] + 1))-a2[ID0[i]])/a3[ID0[i]])) + alpha2 * sigma2_e[ID0[i]] )
      }
      # Cox-Snell residual with Gauss-Legendre quadrature
      r_CS[ID0[i]] <- tCens[i] / 2 * sum(wk * hCens)
    }
    
    # Abnormal group
    for(i in 1:n1){
      # Left and right hazard functions at integration points
      hLeft <- rep(NA,K)
      hRight <- rep(NA,K)
      for(k in 1:K){
        hLeft[k] = phi * (tLeft[i]/2*(xk[k]+1))^(phi-1) * 
          exp( lambda + alpha1 * a1[ID1[i]]/(1+exp(-((tLeft[i] / 2 * (xk[k] + 1))-a2[ID1[i]])/a3[ID1[i]])) + alpha2 * sigma2_e[ID1[i]] )
        hRight[k] = phi * (tRight[i]/2*(xk[k]+1))^(phi-1) * 
          exp( lambda + alpha1 * a1[ID1[i]]/(1+exp(-((tRight[i] / 2 * (xk[k] + 1))-a2[ID1[i]])/a3[ID1[i]])) + alpha2 * sigma2_e[ID1[i]] )
      }
      
      # Left and right survival functions with Gauss-Legendre quadrature
      sLeft = exp( -tLeft[i] / 2 * sum(wk * hLeft) )
      sRight = exp( -tRight[i] / 2 * sum(wk * hRight) )
      
      # Cox-Snell residuals (DOI: 10.1111/j.0006-341x.2000.00473.x)
      r_CS[ID1[i]] = ( sLeft * (1 - log(sLeft)) - sRight * (1 - log(sRight)) ) / (sLeft - sRight)
    }
  }
  
  cc <- rep(0, n0+n1)
  cc[ID1] <- 1
  dta <- data.frame(time = r_CS, status = cc)
  km <- survfit(Surv(time, status) ~ 1, data = dta)
  tt <- seq(0, 2.1, len = 1000)
  dta_exp <- data.frame(time = tt, exp1 = exp(-tt))
  
  survplot <- ggsurvplot(km,
                         censor = FALSE,
                         legend = "none",
                         palette = "black",
                         linetype = "dashed",
                         xlab = "Cox-Snell residuals",
                         xlim = c(0, 2.1),
                         break.x.by = 0.5,
                         size = 0.3,
                         conf.int = TRUE,
                         ggtheme = theme_bw() + theme(panel.grid.major = element_blank(),
                                                      panel.grid.minor = element_blank()),
                         data = dta)$plot
  
  survplot + geom_line(data = dta_exp, aes(x = time, y = exp1), col = "red") +
    scale_x_continuous(limits=c(0, 2.1), expand = c(0, 0.05))
  
  # ggsave("coxsnell.png", units="in", width=4.5, height=2.5, dpi=300)
  
}


iwres_fc(fitJM1, y=y, time=time, ID=IDL, model="JM1")
iwres_fc(fitJM2, y=y, time=time, ID=IDL, model="JM2")

cox_snell_fc(fitJM1, tCens=tCens, tLeft=tLeft, tRight=tRight, ID0=ID0, ID1=ID1, model="JM1")
cox_snell_fc(fitJM2, tCens=tCens, tLeft=tLeft, tRight=tRight, ID0=ID0, ID1=ID1, model="JM2")
