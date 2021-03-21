data {
  int N_obs; // number of observations
  int N_pts; // number of participants
  int K; // number of predictors + intercept
  int pid[N_obs]; // participant id vector
  matrix[N_obs, K] x; // matrix of predictors
  real y[N_obs]; // y vector
}

parameters {
  vector[K] beta; // fixed intercept and slope
  vector<lower=0>[K] sigma_k; // sd for intercept and predictors
  vector[K] beta_k[N_pts]; // ind participant intercept and slope coefficients by group
  corr_matrix[K] Omega; // correlation matrix
  real<lower=0> sigma; // error
}

model {
  vector[N_obs] mu;
  
  // priors
  beta ~ normal(0, 1);
  Omega ~ lkj_corr(2);
  sigma_k ~ exponential(1);
  sigma ~ exponential(1);
  beta_k ~ multi_normal(beta, quad_form_diag(Omega, sigma_k));
  
  // likelihood
  for(i in 1:N_obs) {
    mu[i] = x[i] * (beta_k[pid[i]]);
  }
  
  y ~ normal(mu, sigma);
}

