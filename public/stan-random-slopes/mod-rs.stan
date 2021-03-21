data {
  int N_obs; // number of observations
  int N_pts; // number of participants
  int N_days; // number of days per participant
  int pid[N_obs]; // participant id vector
  real x[N_obs]; // x vector
  real y[N_obs]; // y vector
}

parameters {
  real alpha; // average intercept
  real beta; // average slope
  real<lower=0> sigma_e; // average error
  vector<lower=0>[2] sigma_pt; // participant intercept and slope sd
  cholesky_factor_corr[2] L_pt; // lkj matrix for pt intercept and slope
  matrix[2, N_pts] z_pt; // standardized correlation matrix
}

transformed parameters {
  matrix[2, N_pts] corr;
  corr = diag_pre_multiply(sigma_pt, L_pt) * z_pt;
}

model {
  vector[N_obs] mu;
  
  // priors
  L_pt ~ lkj_corr_cholesky(1.5);
  to_vector(z_pt) ~ normal(0, 1);
  sigma_e ~ exponential(1);
  sigma_pt ~ exponential(1);
  alpha ~ normal(0, 1);
  beta ~ normal(0, 0.5);
  
  // likelihood
  for(i in 1:N_obs) {
    mu[i] = alpha + corr[1, pid[i]] + (beta + corr[2, pid[i]]) * x[i];
  }
  y ~ normal(mu, sigma_e);
}

generated quantities {
  matrix[2, 2] Omega;
  Omega = L_pt * L_pt';
}
