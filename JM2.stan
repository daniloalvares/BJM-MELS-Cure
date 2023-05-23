// Joint model (3.4)-(3.5)
// - Mixed-effects location scale (MELS) for longitudinal data
// - Three-parameter logistic model for longitudinal data
// - Mixture cure model for survival data
// - Interval-censoring for survival data
// - Shared current value and error variance

// THREE-PARAMETER LOGISTIC SPECIFICATION
functions{               
  vector nonlinear_predictor(int[] IDL, vector time, vector theta, matrix bi){
      int N = num_elements(time);         
      vector[N] a1 = exp(theta[1] + bi[IDL,1]);
      vector[N] a2 = exp(theta[2] + bi[IDL,2]);
      vector[N] a3 = exp(theta[3] + bi[IDL,3]);
      vector[N] out;
       
      for(j in 1:N){ out[j] = a1[j]/(1+exp(-(time[j]-a2[j])/a3[j])); }

      return out;
  }
}

// Legend:
// 0: censored/immune women (normal group)
// 1: uncensored/susceptible women (abnormal group)

data{
  int N;
  int n;
  int n0;
  int n1;
  int noneobs;
  int<lower=1,upper=n> IDL[N];
  int<lower=1,upper=n> ID0[n0];
  int<lower=1,upper=n> ID1[n1];
  int<lower=1,upper=n> IDoneobs[noneobs];
  vector[N] y;
  vector[N] time;
  vector[n0] tCens;
  vector[n1] tLeft;
  vector[n1] tRight;
  int<lower=1,upper=N> start[n];
  int<lower=1,upper=N> stop[n];
  int K;
  vector[K] xk;
  vector[K] wk;
}

parameters{
  vector[4] theta;
  real beta;
  real lambda;
  vector[2] alpha;
  real<lower=0> phi;
  cov_matrix[4] Sigma;
  matrix[n,4] bi;
}

transformed parameters{
  matrix[n,3] a;
  a[,1] = exp(theta[1] + bi[,1]);
  a[,2] = exp(theta[2] + bi[,2]);
  a[,3] = exp(theta[3] + bi[,3]);
  vector[n] bi4 = bi[,4];
  bi4[IDoneobs] = rep_vector(0,noneobs);
  vector[n] sigma_ei = exp(theta[4] + bi4);
  vector[n] sigma2_ei = rows_dot_product(sigma_ei,sigma_ei);
  real eta = 1/(1 + exp(-beta));
}

model{
  vector[N] nonlinpred;
  matrix[n0,K] hCens;
  vector[n0] sCens;
  matrix[n1,K] hLeft;
  matrix[n1,K] hRight;
  vector[n1] sLeft;
  vector[n1] sRight;

  // LOG-LIKELIHOOD FOR LONGITUDINAL SUBMODEL  
  // Nonlinear predictor
  nonlinpred = nonlinear_predictor(IDL, time, theta, bi);
  // Longitudinal Normal log-likelihood
  target += normal_lpdf(y | nonlinpred, sigma_ei[IDL]);

  // LOG-LIKELIHOOD FOR SURVIVAL SUBMODEL
  // Immune women (normal group)
  for(i in 1:n0){
      // Hazard function at integration points
      for(k in 1:K){
          hCens[i,k] = phi * pow(tCens[i]/2*(xk[k]+1), phi-1) * exp( lambda +
            alpha[1] * a[ID0[i],1]/(1+exp(-((tCens[i] / 2 * (xk[k] + 1))-a[ID0[i],2])/a[ID0[i],3])) +
            alpha[2] * sigma2_ei[ID0[i]] );
      }

      // Survival function with Gauss-Legendre quadrature
      sCens[i] = exp( -tCens[i] / 2 * dot_product(wk, hCens[i,]) );

      target += log( eta + (1-eta) * sCens[i] );
  }

  // Susceptible women (abnormal group)
  for(i in 1:n1){
      // Left and right hazard functions at integration points
      for(k in 1:K){
          hLeft[i,k] = phi * pow(tLeft[i]/2*(xk[k]+1), phi-1) * exp( lambda +
            alpha[1] * a[ID1[i],1]/(1+exp(-((tLeft[i] / 2 * (xk[k] + 1))-a[ID1[i],2])/a[ID1[i],3])) +
            alpha[2] * sigma2_ei[ID1[i]] );

          hRight[i,k] = phi * pow(tRight[i]/2*(xk[k]+1), phi-1) * exp( lambda +
            alpha[1] * a[ID1[i],1]/(1+exp(-((tRight[i] / 2 * (xk[k] + 1))-a[ID1[i],2])/a[ID1[i],3])) +
            alpha[2] * sigma2_ei[ID1[i]] );
      }

      // Left and right survival functions with Gauss-Legendre quadrature
      sLeft[i] = exp( -tLeft[i] / 2 * dot_product(wk, hLeft[i,]) );
      sRight[i] = exp( -tRight[i] / 2 * dot_product(wk, hRight[i,]) );

      target += log(1-eta) + log( sLeft[i] - sRight[i] );
  }
  
  // LOG-PRIORS
  // Longitudinal fixed effects
  target += normal_lpdf(theta | 0, 10);

  // Parameter for fraction cure 
  target += normal_lpdf(beta | 0, 10);

  // Association parameter
  target += normal_lpdf(alpha | 0, 10);

  // Log-scale parameter (Weibull hazard)
  target += normal_lpdf(lambda | 0, 10);

  // Shape parameter (Weibull hazard)
  target += cauchy_lpdf(phi | 0, 1);

  // Random-effects variance-covariance matrix
  Sigma ~ inv_wishart(5, diag_matrix(rep_vector(1,4)));

  // Random-effects
  for(i in 1:n){ target += multi_normal_lpdf(bi[i,1:4] | rep_vector(0,4), Sigma); }

}

// LOG-LIKELIHOOD FOR LEAVE-ONE-OUT CROSS-VALIDATION (LOO-CV)
generated quantities{
  vector[n] log_lik;
  vector[N] nonlinpred = nonlinear_predictor(IDL, time, theta, bi);
  vector[N] longit;
  matrix[n0,K] hCens;
  vector[n0] sCens;
  matrix[n1,K] hLeft;
  matrix[n1,K] hRight;
  vector[n1] sLeft;
  vector[n1] sRight;

  for(j in 1:N){ longit[j] = normal_lpdf(y[j] | nonlinpred[j], sigma_ei[IDL[j]]); }
  for(i in 1:n){ log_lik[i] = sum(longit[start[i]:stop[i]]); }

  // Immune women (normal group)
  for(i in 1:n0){
      // Hazard function at integration points
      for(k in 1:K){
          hCens[i,k] = phi * pow(tCens[i]/2*(xk[k]+1), phi-1) * exp( lambda +
            alpha[1] * a[ID0[i],1]/(1+exp(-((tCens[i] / 2 * (xk[k] + 1))-a[ID0[i],2])/a[ID0[i],3])) +
            alpha[2] * sigma2_ei[ID0[i]] );
      }

      // Survival function with Gauss-Legendre quadrature
      sCens[i] = exp( -tCens[i] / 2 * dot_product(wk, hCens[i,]) );

      log_lik[ID0[i]] = log_lik[ID0[i]] + log( eta + (1-eta) * sCens[i] );
  }

  // Susceptible women (abnormal group)
  for(i in 1:n1){
      // Left and right hazard functions at integration points
      for(k in 1:K){
          hLeft[i,k] = phi * pow(tLeft[i]/2*(xk[k]+1), phi-1) * exp( lambda +
            alpha[1] * a[ID1[i],1]/(1+exp(-((tLeft[i] / 2 * (xk[k] + 1))-a[ID1[i],2])/a[ID1[i],3])) +
            alpha[2] * sigma2_ei[ID1[i]] );

          hRight[i,k] = phi * pow(tRight[i]/2*(xk[k]+1), phi-1) * exp( lambda +
            alpha[1] * a[ID1[i],1]/(1+exp(-((tRight[i] / 2 * (xk[k] + 1))-a[ID1[i],2])/a[ID1[i],3])) +
            alpha[2] * sigma2_ei[ID1[i]] );
      }

      // Left and right survival functions with Gauss-Legendre quadrature
      sLeft[i] = exp( -tLeft[i] / 2 * dot_product(wk, hLeft[i,]) );
      sRight[i] = exp( -tRight[i] / 2 * dot_product(wk, hRight[i,]) );

      log_lik[ID1[i]] = log_lik[ID1[i]] + log(1-eta) + log( sLeft[i] - sRight[i] );
  }

}
