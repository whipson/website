data {
  int N_obs; // number of observations
  int N_pts; // number of participants
  int K; // number of predictors + intercept
  int pid[N_obs]; // participant id vector
  matrix[N_obs, K] x; // matrix of predictors
  real y[N_obs]; // y vector
}

parameters {
  matrix[K, N_pts] z_p; // matrix of intercepts and slope
  vector<lower=0>[K] sigma_p; // sd for intercept and slope
  vector[K] beta; // intercept and slope hyper-priors
  cholesky_factor_corr[K] L_p; // Cholesky correlation matrix
  real<lower=0> sigma; // population sigma
}

transformed parameters {
  matrix[K, N_pts] z; // non-centered version of beta_p
  z = diag_pre_multiply(sigma_p, L_p) * z_p; 
}

model {
  vector[N_obs] mu;
  
  // priors
  beta ~ normal(0, 1);
  sigma ~ exponential(1);
  sigma_p ~ exponential(1);
  L_p ~ lkj_corr_cholesky(2);
  to_vector(z_p) ~ normal(0, 1);
  
  // likelihood
  for(i in 1:N_obs) {
    mu[i] = beta[1] + z[1, pid[i]] + (beta[2] + z[2, pid[i]]) * x[i, 2];
  }
  y ~ normal(mu, sigma);
}

generated quantities {
  matrix[2, 2] Omega;
  Omega = multiply_lower_tri_self_transpose(L_p);
}
