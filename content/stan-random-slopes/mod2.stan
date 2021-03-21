data {
  int N_obs; // number of observations
  int N_pts; // number of participants
  int K; // number of predictors + intercept
  int pid[N_obs]; // participant id vector
  matrix[N_obs, K] x; // matrix of predictors
  real y[N_obs]; // y vector
}

parameters {
  vector[K] beta_p[N_pts]; // ind participant intercept and slope coefficients by group
  vector<lower=0>[K] sigma_p; // sd for intercept and slope
  vector[K] beta; // intercept and slope hyper-priors
  corr_matrix[K] Omega; // correlation matrix
  real<lower=0> sigma; // population sigma
}

model {
  vector[N_obs] mu;
  
  // priors
  beta ~ normal(0, 1);
  Omega ~ lkj_corr(2);
  sigma_p ~ exponential(1);
  sigma ~ exponential(1);
  beta_p ~ multi_normal(beta, quad_form_diag(Omega, sigma_p));
  
  // likelihood
  for(i in 1:N_obs) {
    mu[i] = x[i] * (beta_p[pid[i]]); // * is matrix multiplication in this context
  }
  
  y ~ normal(mu, sigma);
}
