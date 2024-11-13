#include <RcppArmadillo.h>
#include <RcppDist.h> 
#include <progress.hpp>
#include <progress_bar.hpp>
#include <Rcpp.h>
#include <limits>
#include <chrono>
#define INF std::numeric_limits<double>::infinity()
using namespace Rcpp;
using namespace arma;
static double const log2pi = std::log(2.0 * M_PI);
//[[Rcpp::depends(RcppArmadillo, RcppDist, RcppProgress)]]

//update u with MH
void update_u(double &u_0, double &accept_rate , List &ngg_params, const ivec &clus){
  double v_old = log(u_0);
  double v_new = R::rnorm(v_old,1);
  double u_new = exp(v_new);
  ivec t_unique = arma::unique(clus);
  double k = t_unique.n_elem;
  double n = clus.n_elem; 
  double sigma = as<double>(ngg_params[0]);
  double kappa = as<double>(ngg_params[1]);
  double log_pi_new = (n-1)*log(u_new)-kappa/sigma*(pow(u_new+1,sigma)-1)+(sigma*k-n)*log(u_new+1);
  double log_pi_old = (n-1)*log(u_0)-kappa/sigma*(pow(u_0+1,sigma)-1)+(sigma*k-n)*log(u_0+1);
  double prop_old = R::dnorm4(v_old,v_new,1,true);
  double prop_new = R::dnorm4(v_new,v_old,1,true);
  double alpha = log_pi_new + prop_old + log(1/u_0) - log_pi_old - prop_new - log(1/u_new);
  double ratio_log = std::min(0.0,alpha);
  if(log(randu()) <= ratio_log){
    u_0 = u_new;
    accept_rate +=1;
  }
}

double LogSumExp(double a, double b){
  if (a == -INF && b == -INF) return -INF;
	if (a > b) return a + log(1 + exp(b - a));
	else return b + log(1 + exp(a - b));
}

int sample_log(vec &logprobs){
  vec probs(logprobs.n_elem);
  probs = exp(logprobs);
  probs = probs / accu(probs);
  vec cum_probs = cumsum(probs);
  double u = randu();
  for(uword k = 0; k < probs.n_elem; k++) {
    if(u <= cum_probs[k]) {
      return k;
    }
  }
  return -1;
}

double dt_ls(const double &x, const double &df, const double &mu, const double &sigma){
  double z = (x - mu)/sigma;
  double out = lgamma((df + 1) / 2) - log(sqrt(M_PI * df)) - log(sigma) - lgamma(df / 2) - (df + 1) * std::log1p(z * z / df) / 2;
  return(out);
}

double d_norm(const double &x, const double &mu, const double &sd, bool logd = false){
  double out = -(0.5 * std::log(2.0 * M_PI) + log(sd)) - 0.5 * ((x - mu) / sd)*((x - mu) / sd);
  if (logd == false) {
    out = exp(out);
  }
  return out;
}

void inplace_tri_mat_mult(rowvec &x, mat const &trimat){
  uword const n = trimat.n_cols;
  for(unsigned j = n; j-- > 0;){
    double tmp(0.);
    for(unsigned i = 0; i <= j; ++i)
      tmp += trimat.at(i, j) * x[i];
    x[j] = tmp;
  }
}

rowvec dmvnrm_arma_fast(const mat &x, const rowvec &mean, const mat &sigma, const bool logd = false) { 
  using arma::uword;
  uword const n = x.n_rows, 
            xdim = x.n_cols;
  arma::rowvec out(n);
  arma::mat const rooti = arma::inv(trimatu(arma::chol(sigma)));
  double const rootisum = arma::sum(log(rooti.diag())), 
              constants = -(double)xdim/2.0 * log2pi, 
            other_terms = rootisum + constants;
  
  arma::rowvec z;
  for (uword i = 0; i < n; i++) {
      z = (x.row(i) - mean);
      inplace_tri_mat_mult(z, rooti);
      out(i) = other_terms - 0.5 * arma::dot(z, z);     
  }  
    
  if (logd)
    return out;
  return exp(out);
}

rowvec dmvt_fast(const mat &x, const rowvec &mean, const mat &sigma, const double &df, const bool logd = false) {
  uword const n = x.n_rows, 
            p = x.n_cols;
  arma::rowvec out(n);
  mat rooti = arma::inv(arma::trimatu(arma::chol(sigma)));
  double rootisum = arma::sum(log(rooti.diag()));
  double constants = lgamma((df + p) / 2.0) - lgamma(df / 2.0) - 0.5 * p * log(df * M_PI);
  
  arma::rowvec z;
  for (uword i = 0; i < n; i++) {
      z = (x.row(i) - mean);
      double quadform = arma::as_scalar(z * rooti.t() * rooti * z.t());
      out(i) = constants + rootisum - 0.5 * (df + p) * log(1.0 + quadform / df);   
  }  

  if (logd)
    return out;
  return exp(out);
}

//for cont. covariates we consider as similarity function the marginal under a 
//normal sampling model N(mu,sigma2) with conjugate prior NIG(m,k0,v,S0) = N(mu; m, sigma2/k0)xIG(sigma2; a0, b0)
//for bin. covariates the similarity function becomes a Beta–Binomial probability (BetaBin(n,a,b))
//without the Binomial coefficient
  
double log_g_ppmx(const mat &x, const vec &type, List &g_params){
  std::string fun = g_params[0];
  rowvec density(x.n_cols);
  uvec indices_bin = find(type == 0);
  uvec indices_cat = find(type == -1);
  uvec indices_cont = find(type == 1);
  mat cont = x.cols(indices_cont);
  mat bin = x.cols(indices_bin);
  mat cat = x.cols(indices_cat);
  int n = x.n_rows;
  
  if (indices_cont.n_elem > 0){
    vec identity = ones<vec>(n);
    mat I = arma::eye(n,n);
    mat ii = identity*identity.t();
    if (fun == "ppmx_n"){
      double c1 = g_params[1]; 
      double c2 = g_params[2]; 
      double V = c1;
      double B = c2;
      mat eval(cont.n_cols, n);
      for (uword i = 0; i < cont.n_cols; i++){
        eval.row(i) = cont.col(i).t();
      }
      density(indices_cont) = dmvnrm_arma_fast(eval, 0*identity.t(), V*I+B*ii, true);
    }
    
    if (fun == "ppmx_t"){
      double a0 = g_params[1]; 
      double k0 = g_params[2];   
      mat eval(cont.n_cols, n);
      for (uword i=0; i<cont.n_cols; i++){
        eval.row(i) = cont.col(i).t();
      }
      density(indices_cont) = dmvt_fast(eval, 0*identity.t(), (1.0/a0)*(I + ii/k0), 2*a0, true);
    }
    
  }
  if (indices_bin.n_elem > 0){
    double a = g_params[3];
    double b = g_params[4];
    double C = lgamma(a+b)-lgamma(a)-lgamma(b)-lgamma(a+b+x.n_rows);
    for (uword i = 0; i < bin.n_cols; ++i){
      density(indices_bin(i)) = C+lgamma(a+arma::sum(bin.col(i)))+lgamma(b+x.n_rows-arma::sum(bin.col(i)));
    }
  }
  if (indices_cat.n_elem > 0){
    for (uword i = 0; i < cat.n_cols; ++i){
      vec categs = g_params[5+2*i];
      vec alpha = g_params[6+2*i];
      double alpha_0 = sum(alpha);
      double C = lgamma(alpha_0)-lgamma(alpha_0+n);
      vec freqs(categs.n_elem);
      for (uword j = 0; j < freqs.n_elem; j++){
        freqs(j)= accu(cat.col(i) == categs(j));
      }
      density(indices_cat(i)) = C+sum(lgamma(alpha+freqs))-sum(lgamma(alpha));
    }
  }
  double tot=arma::sum(density);
  return tot;
}

rowvec compute_mode(const mat &x){
  rowvec mode(x.n_cols);
  for (uword i = 0; i < x.n_cols; i++){
    vec unq_elem = arma::unique(x.col(i));
    vec elem_count(unq_elem.n_elem);
    for (size_t j = 0 ; j < elem_count.n_elem; j++){
     elem_count(j)= accu(x.col(i) == unq_elem(j));
    }
    mode(i) = unq_elem(index_max(elem_count));
  }
  return mode;
}

double log_g_123_new(const mat &x, const rowvec &x0, const vec &type, List &g_params){
  std::string fun = g_params[0];

  if(x.n_rows == 1 || fun == "g0" ){
    return 0.0;
  }
  else{
    int n = x.n_rows;
    mat x_x0 = join_vert(x, x0);
    rowvec centroid_old = arma::mean(x,0);
    rowvec centroid_new = (centroid_old * n + x0) / (n + 1);
    uvec indices_bin = find(type == 0);
    uvec indices_cat = find(type == -1);
    uvec indices_cont = find(type == 1);
  
    centroid_old.elem(indices_bin) = round(centroid_old.elem(indices_bin));
    centroid_new.elem(indices_bin) = round(centroid_new.elem(indices_bin));
    
    double dist_old = 0.0;
    double dist_new = 0.0;
  
    //compute,for each obs, Mahalanobis distance for cont. covariates and Hamming distance for bin. covariates
    
    if (indices_cont.n_elem > 0){
      mat cont_old = x.cols(indices_cont);
      mat cont_new = x_x0.cols(indices_cont);
      rowvec cont_centroid_old = centroid_old.cols(indices_cont);
      rowvec cont_centroid_new = centroid_new.cols(indices_cont);
      mat x_cont_old = cont_old.each_row() - cont_centroid_old;
      mat x_cont_new = cont_new.each_row() - cont_centroid_new;
      mat invcov= g_params[3]; 
      vec dist_cont_old = sqrt(arma::sum((x_cont_old * invcov) % x_cont_old, 1));
      vec dist_cont_new = sqrt(arma::sum((x_cont_new * invcov) % x_cont_new, 1)); 
      double cont_frac_old = (double)(cont_old.n_cols)/x.n_cols;
      double cont_frac_new = (double)(cont_new.n_cols)/x_x0.n_cols;
      dist_old = dist_old + accu(cont_frac_old * dist_cont_old);
      dist_new = dist_new + accu(cont_frac_new * dist_cont_new);
    }
    if (indices_bin.n_elem > 0){
      mat bin_old = x.cols(indices_bin);
      mat bin_new = x_x0.cols(indices_bin);
      rowvec bin_centroid_old = centroid_old.cols(indices_bin);
      rowvec bin_centroid_new = centroid_new.cols(indices_bin);
      vec dist_bin_old = sum(abs(bin_old.each_row() - bin_centroid_old), 1) / bin_old.n_cols;
      vec dist_bin_new = sum(abs(bin_new.each_row() - bin_centroid_new), 1) / bin_new.n_cols;
      double bin_frac_old = (double)(bin_old.n_cols)/x.n_cols;
      double bin_frac_new = (double)(bin_new.n_cols)/x_x0.n_cols;
      dist_old = dist_old + accu(bin_frac_old * dist_bin_old);
      dist_new = dist_new + accu(bin_frac_new * dist_bin_new);
    }
    if (indices_cat.n_elem > 0){
      mat cat_old = x.cols(indices_cat);
      mat cat_new = x_x0.cols(indices_cat);
      rowvec cat_centroid_old = compute_mode(cat_old);
      rowvec cat_centroid_new = compute_mode(cat_new);
      vec dist_cat_old(x.n_rows), dist_cat_new(x_x0.n_rows);
      for (uword i = 0; i < cat_old.n_rows; i++){
        dist_cat_old(i) = sum(cat_old.row(i) != cat_centroid_old) / cat_old.n_cols;
        dist_cat_new(i) = sum(cat_new.row(i) != cat_centroid_new) / cat_new.n_cols;
      }
      dist_cat_new(dist_cat_new.n_elem-1) = sum(cat_new.row(dist_cat_new.n_elem-1) != cat_centroid_new) / cat_new.n_cols;
      double cat_frac_old = (double)(cat_old.n_cols)/x.n_cols;
      double cat_frac_new = (double)(cat_new.n_cols)/x_x0.n_cols;
      dist_old = dist_old + accu(cat_frac_old * dist_cat_old);
      dist_new = dist_new + accu(cat_frac_new * dist_cat_new);
    }

    double g(0);
    double lambda = g_params[1]; 
    double alpha = g_params[2] ;
    if (fun == "g1"){
        g = pow(-dist_new*lambda,alpha) - pow(-dist_old*lambda,alpha);
    }
    if (fun == "g2"){
        g = -alpha*log(1+dist_new*lambda) - (-alpha*log(1+dist_old*lambda));
    }
    if (fun == "g3"){
        g = -dist_new*lambda*log(1+dist_new*lambda) - (-dist_old*lambda*log(1+dist_old*lambda));
    }
    return g;
  }
}

double log_g_123(const mat &x, const vec &type, List &g_params){
  std::string fun = g_params[0];

  if(x.n_rows == 1 || fun == "g0" ){
    return 0.0;
  }
  else{
    rowvec centroid = arma::mean(x,0);
    uvec indices_bin = find(type == 0);
    uvec indices_cat = find(type == -1);
    uvec indices_cont = find(type == 1);
    mat cont = x.cols(indices_cont);
    mat bin = x.cols(indices_bin);
    mat cat = x.cols(indices_cat);
    centroid.elem(indices_bin) = round(centroid.elem(indices_bin));
    
    rowvec cont_centroid(cont.n_cols);
    rowvec bin_centroid(bin.n_cols);
    rowvec cat_centroid(cat.n_cols);
    vec dist_bin(bin.n_rows);
    vec dist_cat(cat.n_rows);
    vec dist_cont(cont.n_rows);
    //compute,for each obs, Mahalanobis distance for cont. covariates and Hamming distance for bin. covariates
    
    if (indices_cont.n_elem > 0){
      cont_centroid = centroid.cols(indices_cont);
      mat x_cont;
      x_cont.copy_size(cont);
      x_cont = cont.each_row() - cont_centroid;
      mat invcov= g_params[3]; 
      dist_cont = sqrt(arma::sum((x_cont * invcov) % x_cont, 1)); 
    }
    if (indices_bin.n_elem > 0){
      bin_centroid = centroid.cols(indices_bin);
      dist_bin = sum(abs(bin.each_row() - bin_centroid), 1) / bin.n_cols;
    }
    if (indices_cat.n_elem > 0){
      cat_centroid = compute_mode(cat);
      for (uword i = 0; i < cat.n_rows; i++){
        dist_cat(i) = sum(cat.row(i) != cat_centroid) / cat.n_cols;
      }
    }

    //sum all the distances of the cluster
    double cont_frac = (double)(cont.n_cols)/x.n_cols;
    double bin_frac = (double)(bin.n_cols)/x.n_cols;
    double cat_frac = (double)(cat.n_cols)/x.n_cols;

    double dist = accu(cont_frac*dist_cont + bin_frac*dist_bin + cat_frac*dist_cat);
    double g(0);
    double lambda = g_params[1]; 
    double alpha = g_params[2] ;
    if (fun == "g1"){
        g = pow(-dist*lambda,alpha);
    }
    if (fun == "g2"){
        g = -alpha*log(1+dist*lambda);
    }
    if (fun == "g3"){
        g = -dist*lambda*log(1+dist*lambda);
    }
    return g;
  }
}

double log_g(const mat &x, const vec &type, List &g_params){
  std::string fun = g_params[0];
  if (fun == "g0" || fun == "g1" || fun == "g2" || fun == "g3"){
    return log_g_123(x, type, g_params);
  }
  if (fun == "ppmx_n" || fun == "ppmx_t"){
    return log_g_ppmx(x, type, g_params);
  }
  else {
    Rcpp::Rcout<<"Invalid g"<<std::endl;
    return 0;
  }
}

double log_prior(List &P0_params, double y_star = 0, bool gen = false){
  double mu0 = P0_params[0];
  double lambda0 = P0_params[1];
  double a0 = P0_params[2];
  double b0 = P0_params[3];
  if(gen){
    return r_lst(2*a0, mu0, sqrt(b0*(lambda0+1)/(a0*lambda0)));
  }else{
    return dt_ls(y_star,  2*a0, mu0, sqrt(b0*(lambda0+1)/(a0*lambda0)));
  }
}

double log_post(const vec &y, List &P0_params, double y_star = 0, bool gen = false){
  double mu0 = P0_params[0];
  double lambda0 = P0_params[1];
  double a0 = P0_params[2];
  double b0 = P0_params[3];
  double n = y.n_elem;  
  double y_mean = arma::mean(y);
  double mu_p = (n*y_mean+lambda0*mu0)/(n+lambda0);
  double lambda_p = lambda0+n;
  double a_p = a0 + 0.5*n;
  double b_p = b0 + 0.5*(arma::accu(pow(y-y_mean*arma::ones(y.n_elem,1),2)))+0.5*(n*lambda0*pow(mu0-y_mean,2))/(n+lambda0);
  if(gen){
    return r_lst(2*a_p, mu_p, sqrt(b_p*(lambda_p+1)/(a_p*lambda_p)));
  }else{
    return dt_ls(y_star, 2*a_p, mu_p, sqrt(b_p*(lambda_p+1)/(a_p*lambda_p)));
  }
}

double log_post_beta_bern(const vec &y, List &P0_params, double y_star = 0, bool gen = false){
  double a0 = P0_params[0];
  double b0 = P0_params[1];
  double n = y.n_elem; 
  double a_p = sum(y) + a0;
  double b_p = n - sum(y) + b0;
  double p_1 = a_p/(a_p + b_p);
  double p_0 = b_p/(a_p + b_p);
  if(gen){
    double res = (randu() <= p_1)? 1:0;
    return res;
  }else{
    double res = (y_star)? log(p_1):log(p_0);
    return res;
  }
}

double log_prior_beta_bern(List &P0_params, double y_star = 0, bool gen = false){
  double a0 = P0_params[0];
  double b0 = P0_params[1];
  double p_1 = a0/(a0 + b0);
  double p_0 = b0/(a0 + b0);
  if(gen){
    double res = (randu() <= p_1)? 1:0;
    return res;
  }else{
    double res = (y_star)? log(p_1):log(p_0);
    return res;
  }
}

double eval_log_post_beta_multibern(const mat &y, List &P0_params, rowvec y_star){
  mat B = P0_params[0];
  double n = y.n_rows; 
  double res(0);
  for ( uword i = 0; i < y.n_cols; i++){
    res += (log(arma::sum(y.col(i) == y_star(i)) + B(y_star(i), i)) - log(n + B(0,i) + B(1,i)));
  }
  return res;
}

rowvec gen_post_beta_multibern(const mat &y, List &P0_params){
  mat B = P0_params[0];
  double n = y.n_rows; 
  rowvec res(B.n_cols);
  for ( uword i = 0; i < y.n_cols; i++){
    double a_p = sum(y.col(i)) + B(1,i);
    double b_p = n - sum(y.col(i)) + B(0,i);
    double p_1 = a_p/(a_p + b_p);
    //double p_0 = b_p/(a_p + b_p);
    res(i) = (randu() <= p_1)? 1:0;
  }
  return res;
}

double eval_log_prior_beta_multibern(List &P0_params, rowvec y_star){
  mat B = P0_params[0];
  double res(0);
  for ( uword i = 0; i < B.n_cols; i++){
    res += (log(B(y_star(i), i)) - log(B(0,i) + B(1,i)));
  }
  return res;
}

rowvec gen_prior_beta_multibern(List &P0_params){
  mat B = P0_params[0];
  rowvec res(B.n_cols);
  for ( uword i = 0; i < B.n_cols; i++){
    double p_1 = B(1,i)/(B(1,i) + B(0,1));
    //double p_0 = B(0,i)/(B(1,i) + B(0,1));
    res(i) = (randu() <= p_1)? 1:0;
  }
  return res;
}

double log_prior_pois_gamma(List &P0_params, double y_star = 0, bool gen = false){
  double alpha = P0_params[0];  //strictly positive, need not be integer.
  double beta = P0_params[1];
  if(gen){
    double res = R::rnbinom(alpha, beta/(beta+1));
    return res;
  }else{
    double res = R::dnbinom(y_star, alpha, beta/(beta+1), TRUE);
    return res;
  }
}

double log_post_pois_gamma(const vec &y, List &P0_params, double y_star = 0, bool gen = false){
  double alpha = P0_params[0];
  double beta = P0_params[1];
  double n = y.n_elem; 
  double alpha_p = sum(y) + alpha;
  double beta_p = n + beta;
  if(gen){
    double res = R::rnbinom(alpha_p, beta_p/(beta_p+1));
    return res;
  }else{
    double res = R::dnbinom(y_star, alpha_p, beta_p/(beta_p+1), TRUE);
    return res;
  }
}

double log_prior_reg(List &P0_params, double y_star = 0, rowvec x_star = 0, bool gen = false ){
  vec beta0 = P0_params[0];
  mat B0 = P0_params[1];
  double a0 = P0_params[2];
  double b0 = P0_params[3];
  rowvec x_star_intrcpt = join_horiz(rowvec{1},x_star); 
  if (gen){
    return r_lst(2*a0, as_scalar(x_star_intrcpt*beta0), sqrt((b0/a0)*(1 + as_scalar(x_star_intrcpt*B0*x_star_intrcpt.t()))));
  }else{
    double dens = dt_ls(y_star,  2*a0, as_scalar(x_star_intrcpt*beta0), sqrt((b0/a0)*(1 + as_scalar(x_star_intrcpt*B0*x_star_intrcpt.t()))));
    return dens;
  }
}

double log_post_reg(const vec &y, const mat x, List &P0_params, double y_star = 0, rowvec x_star = 0, bool gen = false ){
  vec beta0 = P0_params[0];
  mat B0 = P0_params[1];
  double a0 = P0_params[2];
  double b0 = P0_params[3];
  vec intrcp = arma::ones<vec>(x.n_rows); // Create a column vector of ones
  mat x_intrcpt = join_horiz(intrcp, x); 
  rowvec x_star_intrcpt = join_horiz(rowvec{1},x_star);
  double n = y.n_elem;  
  mat B_n = inv(inv(B0) + x_intrcpt.t()*x_intrcpt);
  vec beta_n = B_n*(x_intrcpt.t()*y + inv(B0)*beta0);
  double a_n = 2*a0 + n;
  double b_n = 2*b0 + as_scalar(beta0.t()*inv(B0)*beta0) + as_scalar(y.t()*y) - as_scalar(beta_n.t()*inv(B_n)*beta_n);
  if (gen){
    return r_lst(a_n, as_scalar(x_star_intrcpt*beta_n), sqrt((b_n/a_n)*(1 + as_scalar(x_star_intrcpt*B_n*x_star_intrcpt.t()))));
  }else{
    double dens = dt_ls(y_star,  a_n, as_scalar(x_star_intrcpt*beta_n), sqrt((b_n/a_n)*(1 + as_scalar(x_star_intrcpt*B_n*x_star_intrcpt.t()))));
    return dens;
  }
}

double eval_log_prior_multi(List &P0_params, rowvec y_star){
  rowvec mu0 = P0_params[0];
  double k0 = P0_params[1];
  mat S0 = P0_params[2];
  double v0 = P0_params[3];
  double p = y_star.n_elem;
  return dmvt_fast(y_star, mu0, (k0+1)/(k0*(v0-p+1))*S0, v0-p+1, true)(0);
}

double eval_log_post_multi(const mat &y, List &P0_params, rowvec y_star){
  double n = y.n_rows;
  rowvec mu0 = P0_params[0];
  double k0 = P0_params[1];
  mat S0 = P0_params[2];
  double v0 = P0_params[3];
  double p = y_star.n_elem;
  rowvec mean = arma::mean(y, 0);
  rowvec mun = (k0*mu0 + n*mean)/(k0 + n);
  double kn = k0 + n;
  mat Sn = S0 + (y.each_row()-mean).t()*(y.each_row()-mean) + (k0*n)/(k0 + n) * (mean-mu0).t()*(mean-mu0);
  double vn = v0 + n;
  return dmvt_fast(y_star, mun, (kn+1)/(kn*(vn-p+1))*Sn, vn-p+1, true)(0);
}

rowvec gen_prior_multi(List &P0_params){
  rowvec mu0 = P0_params[0];
  double k0 = P0_params[1];
  mat S0 = P0_params[2];
  double v0 = P0_params[3];
  double p = mu0.n_elem;

  double u = arma::chi2rnd(v0-p+1);
  rowvec y = arma::mvnrnd(zeros(p), (k0+1)/(k0*(v0-p+1))*S0).t();
  return sqrt((v0-p+1)/u)*y + mu0;
}

rowvec gen_post_multi(const mat &Y, List &P0_params){
  double n = Y.n_rows;
  rowvec mu0 = P0_params[0];
  double k0 = P0_params[1];
  mat S0 = P0_params[2];
  double v0 = P0_params[3];
  double p = mu0.n_elem;

  rowvec mean = arma::mean(Y, 0);
  rowvec mun = (k0*mu0 + n*mean)/(k0 + n);
  double kn = k0 + n;
  mat Sn = S0 + (Y.each_row()-mean).t()*(Y.each_row()-mean) + (k0*n)/(k0 + n) * (mean-mu0).t()*(mean-mu0);
  double vn = v0 + n;

  double u = arma::chi2rnd(vn-p+1);
  rowvec y = arma::mvnrnd(zeros(p), (kn+1)/(kn*(vn-p+1))*Sn).t();
  return sqrt((vn-p+1)/u)*y + mun;
}

double innergibbs_rev_new(const mat &YX, const uvec &S_idx, const uvec &S_with_ij_idx, uvec &S_i_with_i_idx, uvec &S_j_with_j_idx, const ivec &target_labels_local, const vec &type, List &g_params, List &P0_params, double &sigma){
  mat X = YX.cols(1, YX.n_cols - 1);
  vec Y = YX.col(0);
  //mat X = YX.cols(2, YX.n_cols - 1);
  //mat Y = YX.cols(0, 1);
  //mat X = YX.cols(6, YX.n_cols - 1);
  //mat Y = YX.cols(0,5);
  int n = S_idx.n_elem;
  double logprob = 0.0;
  double log_g_j = log_g(X.rows(S_j_with_j_idx), type, g_params);
  double log_g_i = log_g(X.rows(S_i_with_i_idx), type, g_params);
  double log_g_j_x0 = 0.0, log_g_i_x0 = 0.0, term_i = 0.0, term_j = 0.0;
  for( int i = 0; i < n; ++i ){
    uword cidx = S_idx(i);
    double n_i = S_i_with_i_idx.n_elem;
    double n_j = S_j_with_j_idx.n_elem;
    bool ki = arma::any(S_i_with_i_idx == cidx);
    if(ki){
      S_i_with_i_idx.shed_row(arma::conv_to<uword>::from(arma::find(S_i_with_i_idx==cidx)));
      log_g_i_x0 = log_g_i;
      log_g_i = log_g(X.rows(S_i_with_i_idx), type, g_params);
      mat x_j = X.rows(S_j_with_j_idx);
      //mat x_i = X.rows(S_i_with_i_idx);
      mat x_j_x0 = join_vert(x_j,X.row(cidx));
      log_g_j_x0 = log_g(x_j_x0, type, g_params);
      term_i = log(n_i-1-sigma)+log_post(Y(S_i_with_i_idx), P0_params, Y(cidx))+log_g_i_x0-log_g_i;
      term_j = log(n_j-sigma)+log_post(Y(S_j_with_j_idx), P0_params, Y(cidx))+log_g_j_x0-log_g_j;
      //term_i = log(n_i-1-sigma)+log_post_reg(Y(S_i_with_i_idx), x_i, P0_params, Y(cidx), X.row(cidx))+log_g_i_x0-log_g_i;
      //term_j = log(n_j-sigma)+log_post_reg(Y(S_j_with_j_idx), x_j, P0_params, Y(cidx), X.row(cidx))+log_g_j_x0-log_g_j;
      //term_i = log(n_i-1-sigma)+log_post_beta_bern(Y(S_i_with_i_idx), P0_params, Y(cidx))+log_g_i_x0-log_g_i;
      //term_j = log(n_j-sigma)+log_post_beta_bern(Y(S_j_with_j_idx), P0_params, Y(cidx))+log_g_j_x0-log_g_j;
      //term_i = log(n_i-1-sigma)+eval_log_post_beta_multibern(Y.rows(S_i_with_i_idx), P0_params, Y.row(cidx))+log_g_i_x0-log_g_i;
      //term_j = log(n_j-sigma)+eval_log_post_beta_multibern(Y.rows(S_j_with_j_idx), P0_params, Y.row(cidx))+log_g_j_x0-log_g_j;
      //term_i = log(n_i-1-sigma)+eval_log_post_multi(Y.rows(S_i_with_i_idx), P0_params, Y.row(cidx))+log_g_i_x0-log_g_i;
      //term_j = log(n_j-sigma)+eval_log_post_multi(Y.rows(S_j_with_j_idx), P0_params, Y.row(cidx))+log_g_j_x0-log_g_j;
    }else{
      S_j_with_j_idx.shed_row(arma::conv_to<uword>::from(arma::find(S_j_with_j_idx==cidx)));
      log_g_j_x0 = log_g_j;
      mat x_i = X.rows(S_i_with_i_idx);
      //mat x_j = X.rows(S_j_with_j_idx);//
      mat x_i_x0 = join_vert(x_i,X.row(cidx));
      log_g_i_x0 = log_g(x_i_x0, type, g_params);
      term_i = log(n_i-sigma)+log_post(Y(S_i_with_i_idx), P0_params, Y(cidx))+log_g_i_x0-log_g_i;
      term_j = log(n_j-1-sigma)+log_post(Y(S_j_with_j_idx), P0_params, Y(cidx))+log_g_j_x0-log_g_j;
      //term_i = log(n_i-sigma)+log_post_reg(Y(S_i_with_i_idx), x_i, P0_params, Y(cidx), X.row(cidx))+log_g_i_x0-log_g_i;
      //term_j = log(n_j-1-sigma)+log_post_reg(Y(S_j_with_j_idx), x_j, P0_params, Y(cidx), X.row(cidx))+log_g_j_x0-log_g_j;
      //term_i = log(n_i-sigma)+log_post_beta_bern(Y(S_i_with_i_idx), P0_params, Y(cidx))+log_g_i_x0-log_g_i;
      //term_j = log(n_j-1-sigma)+log_post_beta_bern(Y(S_j_with_j_idx), P0_params, Y(cidx))+log_g_j_x0-log_g_j;
      //term_i = log(n_i-sigma)+eval_log_post_beta_multibern(Y.rows(S_i_with_i_idx), P0_params, Y.row(cidx))+log_g_i_x0-log_g_i;
      //term_j = log(n_j-1-sigma)+eval_log_post_beta_multibern(Y.rows(S_j_with_j_idx), P0_params, Y.row(cidx))+log_g_j_x0-log_g_j;
      //term_i = log(n_i-sigma)+eval_log_post_multi(Y.rows(S_i_with_i_idx), P0_params, Y.row(cidx))+log_g_i_x0-log_g_i;
      //term_j = log(n_j-1-sigma)+eval_log_post_multi(Y.rows(S_j_with_j_idx), P0_params, Y.row(cidx))+log_g_j_x0-log_g_j;
    }
    
    double nc = LogSumExp(term_i, term_j);
    //double nc = log(exp(term_i)+ exp(term_j));
    
    if(target_labels_local(cidx) == 1){
      S_i_with_i_idx = arma::join_cols(S_i_with_i_idx, uvec({cidx}));
      log_g_i = log_g_i_x0;
      logprob += term_i - nc;
    } else {
      S_j_with_j_idx = arma::join_cols(S_j_with_j_idx, uvec({cidx}));
      log_g_j = log_g_j_x0;
      logprob += term_j - nc;
    }
  }
  return logprob;
}

double innerGibbs_new(const mat &YX, const uvec &S_idx, const uvec &S_with_ij_idx, uvec &S_i_with_i_idx, uvec &S_j_with_j_idx, const int &t, const vec &type, List &g_params, List &P0_params, double &sigma){
  mat X = YX.cols(1, YX.n_cols - 1);
  vec Y = YX.col(0);
  //mat X = YX.cols(2, YX.n_cols - 1);
  //mat Y = YX.cols(0, 1);
  //mat X = YX.cols(6, YX.n_cols - 1);
  //mat Y = YX.cols(0,5);
  int n = S_idx.n_elem;
  double logprob= 0.0;
  double log_g_j = log_g(X.rows(S_j_with_j_idx), type, g_params);
  double log_g_i = log_g(X.rows(S_i_with_i_idx), type, g_params);
  double log_g_j_x0 = 0.0, log_g_i_x0 = 0.0, term_i = 0.0, term_j = 0.0;
  for( int iter = 0 ; iter < t; ++iter ){
    uvec indxperm = arma::shuffle(arma::linspace<arma::uvec>(0, n-1, n));
    for( int i = 0; i < n; ++i ){
      uword cidx = S_idx(indxperm(i));
      double n_i = S_i_with_i_idx.n_elem;
      double n_j = S_j_with_j_idx.n_elem;
      bool ki = arma::any(S_i_with_i_idx == cidx);
      if(ki){
        S_i_with_i_idx.shed_row(arma::conv_to<uword>::from(arma::find(S_i_with_i_idx==cidx)));
        log_g_i_x0 = log_g_i;
        log_g_i = log_g(X.rows(S_i_with_i_idx), type, g_params);
        mat x_j = X.rows(S_j_with_j_idx);
        //mat x_i = X.rows(S_i_with_i_idx);
        mat x_j_x0 = join_vert(x_j,X.row(cidx));
        log_g_j_x0 = log_g(x_j_x0, type, g_params);
        term_i = log(n_i-1-sigma)+log_post(Y(S_i_with_i_idx), P0_params, Y(cidx))+log_g_i_x0-log_g_i;
        term_j = log(n_j-sigma)+log_post(Y(S_j_with_j_idx), P0_params, Y(cidx))+log_g_j_x0-log_g_j;
        //term_i = log(n_i-1-sigma)+log_post_reg(Y(S_i_with_i_idx), x_i, P0_params, Y(cidx), X.row(cidx))+log_g_i_x0-log_g_i;
        //term_j = log(n_j-sigma)+log_post_reg(Y(S_j_with_j_idx), x_j, P0_params, Y(cidx), X.row(cidx))+log_g_j_x0-log_g_j;
        //term_i = log(n_i-1-sigma)+log_post_beta_bern(Y(S_i_with_i_idx), P0_params, Y(cidx))+log_g_i_x0-log_g_i;
        //term_j = log(n_j-sigma)+log_post_beta_bern(Y(S_j_with_j_idx), P0_params, Y(cidx))+log_g_j_x0-log_g_j;
        //term_i = log(n_i-1-sigma)+eval_log_post_beta_multibern(Y.rows(S_i_with_i_idx), P0_params, Y.row(cidx))+log_g_i_x0-log_g_i;
        //term_j = log(n_j-sigma)+eval_log_post_beta_multibern(Y.rows(S_j_with_j_idx), P0_params, Y.row(cidx))+log_g_j_x0-log_g_j;
        //term_i = log(n_i-1-sigma)+eval_log_post_multi(Y.rows(S_i_with_i_idx), P0_params, Y.row(cidx))+log_g_i_x0-log_g_i;
        //term_j = log(n_j-sigma)+eval_log_post_multi(Y.rows(S_j_with_j_idx), P0_params, Y.row(cidx))+log_g_j_x0-log_g_j;
        
      }else{
        S_j_with_j_idx.shed_row(arma::conv_to<uword>::from(arma::find(S_j_with_j_idx==cidx)));
        log_g_j_x0 = log_g_j;
        mat x_i = X.rows(S_i_with_i_idx);
        //mat x_j = X.rows(S_j_with_j_idx);
        mat x_i_x0 = join_vert(x_i,X.row(cidx));
        log_g_i_x0 = log_g(x_i_x0, type, g_params);
        term_i = log(n_i-sigma)+log_post(Y(S_i_with_i_idx), P0_params, Y(cidx))+log_g_i_x0-log_g_i;
        term_j = log(n_j-1-sigma)+log_post(Y(S_j_with_j_idx), P0_params, Y(cidx))+log_g_j_x0-log_g_j;
        //term_i = log(n_i-sigma)+log_post_reg(Y(S_i_with_i_idx), x_i, P0_params, Y(cidx), X.row(cidx))+log_g_i_x0-log_g_i;
        //term_j = log(n_j-1-sigma)+log_post_reg(Y(S_j_with_j_idx), x_j, P0_params, Y(cidx), X.row(cidx))+log_g_j_x0-log_g_j;
        //term_i = log(n_i-sigma)+log_post_beta_bern(Y(S_i_with_i_idx), P0_params, Y(cidx))+log_g_i_x0-log_g_i;
        //term_j = log(n_j-1-sigma)+log_post_beta_bern(Y(S_j_with_j_idx), P0_params, Y(cidx))+log_g_j_x0-log_g_j;
        //term_i = log(n_i-sigma)+eval_log_post_beta_multibern(Y.rows(S_i_with_i_idx), P0_params, Y.row(cidx))+log_g_i_x0-log_g_i;
        //term_j = log(n_j-1-sigma)+eval_log_post_beta_multibern(Y.rows(S_j_with_j_idx), P0_params, Y.row(cidx))+log_g_j_x0-log_g_j;
        //term_i = log(n_i-sigma)+eval_log_post_multi(Y.rows(S_i_with_i_idx), P0_params, Y.row(cidx))+log_g_i_x0-log_g_i;
        //term_j = log(n_j-1-sigma)+eval_log_post_multi(Y.rows(S_j_with_j_idx), P0_params, Y.row(cidx))+log_g_j_x0-log_g_j;
      }
       
      double nc = LogSumExp(term_i, term_j);
      //double nc = log(exp(term_i)+ exp(term_j));
      if (randu()< exp(term_i - nc)){
        S_i_with_i_idx = arma::join_cols(S_i_with_i_idx, uvec({cidx}));
        log_g_i = log_g_i_x0;
        logprob += (term_i - nc);
      } else {
        S_j_with_j_idx = arma::join_cols(S_j_with_j_idx, uvec({cidx}));
        log_g_j = log_g_j_x0;
        logprob += (term_j - nc);
      }
    }
  }
  return logprob;
}

void sample_clus_allocs_Sp_Mrg(const mat &YX, ivec &clus_alloc, double &u, const vec &type, List &g_params, List &ngg_params, List &P0_params, double &accept_rate_split, double &accept_rate_merge, double &count_split, double &count_merge, double nGibbs = 0){
  mat X = YX.cols(1, YX.n_cols - 1);
  vec Y = YX.col(0);
  //mat X = YX.cols(2, YX.n_cols - 1);
  //mat Y = YX.cols(0, 1);
  //mat X = YX.cols(6, YX.n_cols - 1);
  //mat Y = YX.cols(0,5);
  double sigma = ngg_params[0];
  double kappa = ngg_params[1];
  int n = YX.n_rows;
  int maxlabelval = arma::max(clus_alloc) + 1;
  arma::uvec RandIdx = arma::shuffle(arma::linspace<arma::uvec>(0, n-1, n));
  uword i_idx = RandIdx(0);
  uword j_idx = RandIdx(1);
  int c_i = clus_alloc(i_idx); 
  int c_j = clus_alloc(j_idx); 
  int c_launch_i, c_launch_j;
  arma::uvec c_i_idx, c_j_idx, S_with_ij_idx, S_idx, S_i_with_i_idx, S_j_with_j_idx;

  if(c_i == c_j) {
    c_launch_i = maxlabelval;
    c_launch_j = c_j;
    c_i_idx = arma::find(clus_alloc == c_i);
    S_with_ij_idx = c_i_idx;
    S_idx = S_with_ij_idx;
    S_idx.shed_row(arma::conv_to<uword>::from(arma::find(S_idx == i_idx)));
    S_idx.shed_row(arma::conv_to<uword>::from(arma::find(S_idx == j_idx)));
    arma::uvec unirandidx = arma::randi<arma::uvec>(S_idx.n_elem, arma::distr_param(0, 1));
    S_i_with_i_idx = join_cols(S_idx.elem(arma::find(unirandidx == 1)), uvec({i_idx}));
    S_j_with_j_idx = join_cols(S_idx.elem(arma::find(unirandidx == 0)), uvec({j_idx}));

  }else{
    c_launch_i = c_i;
    c_launch_j = c_j;
    c_i_idx = arma::find(clus_alloc == c_i);
    c_j_idx = arma::find(clus_alloc == c_j);
    S_with_ij_idx = arma::join_cols(c_i_idx, c_j_idx);
    S_idx = S_with_ij_idx;
    S_idx.shed_row(arma::conv_to<uword>::from(arma::find(S_idx == i_idx)));
    S_idx.shed_row(arma::conv_to<uword>::from(arma::find(S_idx == j_idx)));
    arma::uvec unirandidx = arma::randi<arma::uvec>(S_idx.n_elem, arma::distr_param(0, 1));
    S_i_with_i_idx = join_cols(S_idx.elem(arma::find(unirandidx == 1)), uvec({i_idx}));
    S_j_with_j_idx = join_cols(S_idx.elem(arma::find(unirandidx == 0)), uvec({j_idx}));
  }
  innerGibbs_new(YX, S_idx, S_with_ij_idx, S_i_with_i_idx, S_j_with_j_idx, nGibbs, type, g_params, P0_params, sigma);
  if (c_i == c_j){
    count_split+=1;
    double c_i_split = c_launch_i;
    double c_j_split = c_launch_j;
    double logprob = innerGibbs_new(YX, S_idx, S_with_ij_idx, S_i_with_i_idx, S_j_with_j_idx, 1, type, g_params, P0_params, sigma);
    ivec c_split = clus_alloc;
    c_split.elem(S_i_with_i_idx).fill(c_i_split);  
    c_split.elem(S_j_with_j_idx).fill(c_j_split);
    double n_ci = S_i_with_i_idx.n_elem;
    double n_cj = S_j_with_j_idx.n_elem;
    double prior_ratio = log(kappa)+sigma*log(1+u)+lgamma(n_ci-sigma)+lgamma(n_cj-sigma)-lgamma(n_ci+n_cj-sigma)-lgamma(1-sigma)+log_g(X.rows(S_i_with_i_idx), type, g_params)+log_g(X.rows(S_j_with_j_idx), type, g_params)-log_g(X.rows(S_with_ij_idx), type, g_params);
    double term1 = 0.0, term2 = 0.0, term3 = 0.0;

    for(uword j = 0; j < S_i_with_i_idx.n_elem ; ++j ){
      if (j == 0){
        term1 += log_prior(P0_params, Y(S_i_with_i_idx(j)));
        //term1 += log_prior_reg(P0_params, Y(S_i_with_i_idx(j)), X.row(S_i_with_i_idx(j)));
        //term1 += log_prior_beta_bern(P0_params, Y(S_i_with_i_idx(j)));
        //term1 += eval_log_prior_beta_multibern(P0_params, Y.row(S_i_with_i_idx(j)));
        //term1 += eval_log_prior_multi(P0_params, Y.row(S_i_with_i_idx(j)));
      }else{
        term1 += log_post(Y(S_i_with_i_idx(regspace<uvec>(0,j-1))), P0_params, Y(S_i_with_i_idx(j)));
        //term1 += log_post_reg(Y(S_i_with_i_idx(regspace<uvec>(0,j-1))), X.rows(S_i_with_i_idx(regspace<uvec>(0,j-1))), P0_params, Y(S_i_with_i_idx(j)), X.row(S_i_with_i_idx(j)));
        //term1 += log_post_beta_bern(Y(S_i_with_i_idx(regspace<uvec>(0,j-1))), P0_params, Y(S_i_with_i_idx(j)));
        //term1 += eval_log_post_beta_multibern(Y.rows(S_i_with_i_idx(regspace<uvec>(0,j-1))), P0_params, Y.row(S_i_with_i_idx(j)));
        //term1 += eval_log_post_multi(Y.rows(S_i_with_i_idx(regspace<uvec>(0,j-1))), P0_params, Y.row(S_i_with_i_idx(j)));      
      }
    }
    for(uword j = 0; j < S_j_with_j_idx.n_elem ; ++j ){
      if (j == 0){
        term2 += log_prior(P0_params, Y(S_j_with_j_idx(j)));
        //term2 += log_prior_reg(P0_params, Y(S_j_with_j_idx(j)), X.row(S_j_with_j_idx(j)));
        //term2 += log_prior_beta_bern(P0_params, Y(S_j_with_j_idx(j)));
        //term2 += eval_log_prior_beta_multibern(P0_params, Y.row(S_j_with_j_idx(j)));
        //term2 += eval_log_prior_multi(P0_params, Y.row(S_j_with_j_idx(j)));
      }else{
        term2 += log_post(Y(S_j_with_j_idx(regspace<uvec>(0,j-1))), P0_params, Y(S_j_with_j_idx(j)));
        //term2 += log_post_reg(Y(S_j_with_j_idx(regspace<uvec>(0,j-1))), X.rows(S_j_with_j_idx(regspace<uvec>(0,j-1))), P0_params, Y(S_j_with_j_idx(j)), X.row(S_j_with_j_idx(j)));
        //term2 += log_post_beta_bern(Y(S_j_with_j_idx(regspace<uvec>(0,j-1))), P0_params, Y(S_j_with_j_idx(j)));
        //term2 += eval_log_post_beta_multibern(Y.rows(S_j_with_j_idx(regspace<uvec>(0,j-1))), P0_params, Y.row(S_j_with_j_idx(j)));
        //term2 += eval_log_post_multi(Y.rows(S_j_with_j_idx(regspace<uvec>(0,j-1))), P0_params, Y.row(S_j_with_j_idx(j)));
      }
    }
    for(uword j = 0; j < S_with_ij_idx.n_elem ; ++j ){
      if (j == 0){
        term3 += log_prior(P0_params, Y(S_with_ij_idx(j)));
        //term3 += log_prior_reg(P0_params, Y(S_with_ij_idx(j)), X.row(S_with_ij_idx(j)));
        //term3 += log_prior_beta_bern(P0_params, Y(S_with_ij_idx(j)));
        //term3 += eval_log_prior_beta_multibern(P0_params, Y.row(S_with_ij_idx(j)));
        //term3 += eval_log_prior_multi(P0_params, Y.row(S_with_ij_idx(j)));
      }else{
        term3 += log_post(Y(S_with_ij_idx(regspace<uvec>(0,j-1))), P0_params, Y(S_with_ij_idx(j)));
        //term3 += log_post_reg(Y(S_with_ij_idx(regspace<uvec>(0,j-1))), X.rows(S_with_ij_idx(regspace<uvec>(0,j-1))), P0_params, Y(S_with_ij_idx(j)), X.row(S_with_ij_idx(j)));
        //term3 += log_post_beta_bern(Y(S_with_ij_idx(regspace<uvec>(0,j-1))), P0_params, Y(S_with_ij_idx(j)));
        //term3 += eval_log_post_beta_multibern(Y.rows(S_with_ij_idx(regspace<uvec>(0,j-1))), P0_params, Y.row(S_with_ij_idx(j)));
        //term3 += eval_log_post_multi(Y.rows(S_with_ij_idx(regspace<uvec>(0,j-1))), P0_params, Y.row(S_with_ij_idx(j)));
      }
    }
    double likelihood_ratio =  term1 + term2 - term3;
    double proposal = -logprob;
    double accept_prob = std::min((exp(prior_ratio+likelihood_ratio+proposal)),1.0);
    
    if (randu()< accept_prob){
      clus_alloc = c_split;
      accept_rate_split +=1;
    }
  }else{
    count_merge+=1;
    double c_i_merge = c_j;
    double c_j_merge = c_j;

    ivec c_merge = clus_alloc;
    c_merge.elem(c_i_idx).fill(c_i_merge);
    c_merge.elem(c_j_idx).fill(c_j_merge);
    double n_ci = c_i_idx.n_elem;
    double n_cj = c_j_idx.n_elem;
    double prior_ratio = lgamma(1-sigma)+lgamma(n_ci+n_cj-sigma)-log(kappa)-lgamma(n_ci-sigma)-lgamma(n_cj-sigma)-sigma*log(1+u)+log_g(X.rows(S_with_ij_idx), type, g_params)-log_g(X.rows(c_i_idx), type, g_params)-log_g(X.rows(c_j_idx), type, g_params);
    double term1 = 0.0, term2 = 0.0, term3 = 0.0;
    for(uword j = 0; j < S_with_ij_idx.n_elem ; ++j ){
      if (j == 0){
        term1 += log_prior(P0_params, Y(S_with_ij_idx(j))) ;
        //term1 += log_prior_reg(P0_params, Y(S_with_ij_idx(j)), X.row(S_with_ij_idx(j)));
        //term1 += log_prior_beta_bern(P0_params, Y(S_with_ij_idx(j))) ;
        //term1 += eval_log_prior_beta_multibern(P0_params, Y.row(S_with_ij_idx(j))) ;
        //term1 += eval_log_prior_multi(P0_params, Y.row(S_with_ij_idx(j)));
      }else{
        term1 += log_post(Y(S_with_ij_idx(regspace<uvec>(0,j-1))), P0_params, Y(S_with_ij_idx(j)));
        //term1 += log_post_reg(Y(S_with_ij_idx(regspace<uvec>(0,j-1))), X.rows(S_with_ij_idx(regspace<uvec>(0,j-1))), P0_params, Y(S_with_ij_idx(j)), X.row(S_with_ij_idx(j)));
        //term1 += log_post_beta_bern(Y(S_with_ij_idx(regspace<uvec>(0,j-1))), P0_params, Y(S_with_ij_idx(j)));
        //term1 += eval_log_post_beta_multibern(Y.rows(S_with_ij_idx(regspace<uvec>(0,j-1))), P0_params, Y.row(S_with_ij_idx(j)));
        //term1 += eval_log_post_multi(Y.rows(S_with_ij_idx(regspace<uvec>(0,j-1))), P0_params, Y.row(S_with_ij_idx(j)));
      }
    }
    for(uword j = 0; j < c_i_idx.n_elem ; ++j ){
      if (j == 0){
        term2 += log_prior(P0_params, Y(c_i_idx(j)));
        //term2 += log_prior_reg(P0_params, Y(c_i_idx(j)), X.row(c_i_idx(j)));
        //term2 += log_prior_beta_bern(P0_params, Y(c_i_idx(j)));
        //term2 += eval_log_prior_beta_multibern(P0_params, Y.row(c_i_idx(j)));
        //term2 += eval_log_prior_multi(P0_params, Y.row(c_i_idx(j)));
      }else{
        term2 += log_post(Y(c_i_idx(regspace<uvec>(0,j-1))), P0_params, Y(c_i_idx(j)));
        //term2 += log_post_reg(Y(c_i_idx(regspace<uvec>(0,j-1))), X.rows(c_i_idx(regspace<uvec>(0,j-1))), P0_params, Y(c_i_idx(j)), X.row(c_i_idx(j)));
        //term2 += log_post_beta_bern(Y(c_i_idx(regspace<uvec>(0,j-1))), P0_params, Y(c_i_idx(j)));
        //term2 += eval_log_post_beta_multibern(Y.rows(c_i_idx(regspace<uvec>(0,j-1))), P0_params, Y.row(c_i_idx(j)));
        //term2 += eval_log_post_multi(Y.rows(c_i_idx(regspace<uvec>(0,j-1))), P0_params, Y.row(c_i_idx(j)));
      }
    }
    for(uword j = 0; j < c_j_idx.n_elem ; ++j ){
      if (j == 0){
        term3 += log_prior(P0_params, Y(c_j_idx(j)));
        //term3 += log_prior_reg(P0_params, Y(c_j_idx(j)), X.row(c_j_idx(j)));
        //term3 += log_prior_beta_bern(P0_params, Y(c_j_idx(j)));
        //term3 += eval_log_prior_beta_multibern(P0_params, Y.row(c_j_idx(j)));
        //term3 += eval_log_prior_multi(P0_params, Y.row(c_j_idx(j)));
      }else{
        term3 += log_post(Y(c_j_idx(regspace<uvec>(0,j-1))), P0_params, Y(c_j_idx(j)));
        //term3 += log_post_reg(Y(c_j_idx(regspace<uvec>(0,j-1))), X.rows(c_j_idx(regspace<uvec>(0,j-1))), P0_params, Y(c_j_idx(j)), X.row(c_j_idx(j)));
        //term3 += log_post_beta_bern(Y(c_j_idx(regspace<uvec>(0,j-1))), P0_params, Y(c_j_idx(j)));
        //term3 += eval_log_post_beta_multibern(Y.rows(c_j_idx(regspace<uvec>(0,j-1))), P0_params, Y.row(c_j_idx(j)));
        //term3 += eval_log_post_multi(Y.rows(c_j_idx(regspace<uvec>(0,j-1))), P0_params, Y.row(c_j_idx(j)));
      }
    }

    double likelihood_ratio =  term1 - term2 - term3;
    ivec target_labels_local(n,fill::zeros);
    target_labels_local.elem(c_i_idx).fill(1);
    target_labels_local.elem(c_j_idx).fill(0);
    double prob = innergibbs_rev_new(YX, S_idx, S_with_ij_idx, S_i_with_i_idx, S_j_with_j_idx, target_labels_local, type, g_params, P0_params, sigma);
    double proposal = prob;
    double accept_prob = std::min((exp(prior_ratio+likelihood_ratio+proposal)),1.0);
    
    if (randu()< accept_prob){
      clus_alloc = c_merge;
      accept_rate_merge +=1;
      clus_alloc(arma::find(clus_alloc > c_i)) -= 1;
    }
  } 
}

void sample_clus_allocs_Gibbs(const mat &YX, const double &u, ivec &cluster_alloc, const vec &type, List &g_params, List &ngg_params, List &P0_params){
  mat X = YX.cols(1, YX.n_cols - 1);
  vec Y = YX.col(0);
  //mat X = YX.cols(2, YX.n_cols - 1);
  //mat Y = YX.cols(0, 1);
  //mat X = YX.cols(6, YX.n_cols - 1);
  //mat Y = YX.cols(0,5);
  double sigma = ngg_params[0];
  double kappa = ngg_params[1];
  double n = YX.n_rows;
  
  ivec clus_unique = arma::unique(cluster_alloc);
  vec clus_count(clus_unique.n_elem);
  for (size_t i = 0 ; i < clus_count.n_elem ; i++){
     clus_count(i)= accu(cluster_alloc == clus_unique(i));
  }

  int k = clus_count.n_elem;
  vec log_g_j(k);
  for (int j = 0; j < k; ++j){
    uvec positions = arma::find(cluster_alloc == j);
    log_g_j(j) = log_g(X.rows(positions), type, g_params);
  }
  
  for (size_t i = 0; i < n; i++){
    bool single = false;
    int c_i = cluster_alloc(i);
    clus_count(c_i) -= 1;
    if(clus_count(c_i)==0){
      single = true;
      clus_count.shed_row(c_i);
      cluster_alloc(arma::find(cluster_alloc > c_i)) -= 1;
      log_g_j.shed_row(c_i);
    }

    int k = clus_count.n_elem;
    vec log_probs(k+1, arma::fill::zeros);
    vec log_g_j_x0(k+1, arma::fill::zeros);
    for (int j = 0; j < k; ++j) {
      uvec positions = arma::find(cluster_alloc == j);
      mat X_j = X.rows(positions);
      int n_j = clus_count(j);
      if (c_i == j && !single){
        uvec del = arma::find(positions == i);
        positions.shed_row(del(0));
        X_j.shed_row(del(0));
        log_g_j_x0(j) = log_g_j(j);
        log_g_j(j) = log_g(X_j, type, g_params);
        log_probs(j) = log(n_j-sigma)-log(1+u)+log_post(Y(positions), P0_params, Y(i))+log_g_j_x0(j)-log_g_j(j);
        //log_probs(j) = log(n_j-sigma)-log(1+u)+log_post_reg(Y(positions), X_j, P0_params, Y(i), X.row(i))+log_g_j_x0(j)-log_g_j(j);
        //log_probs(j) = log(n_j-sigma)-log(1+u)+log_post_beta_bern(Y(positions), P0_params, Y(i))+log_g_j_x0(j)-log_g_j(j);
        //log_probs(j) = log(n_j-sigma)-log(1+u)+eval_log_post_beta_multibern(Y.rows(positions), P0_params, Y.row(i))+log_g_j_x0(j)-log_g_j(j);
        //log_probs(j) = log(n_j-sigma)-log(1+u)+eval_log_post_multi(Y.rows(positions), P0_params, Y.row(i))+log_g_j_x0(j)-log_g_j(j);
      }
      else{
        mat X_j_x0 = join_vert(X_j,X.row(i));
        log_g_j_x0(j) = log_g(X_j_x0, type, g_params);
        log_probs(j) = log(n_j-sigma)-log(1+u)+log_post(Y(positions), P0_params, Y(i))+log_g_j_x0(j)-log_g_j(j);
        //log_probs(j) = log(n_j-sigma)-log(1+u)+log_post_reg(Y(positions), X_j, P0_params, Y(i), X.row(i))+log_g_j_x0(j)-log_g_j(j);
        //log_probs(j) = log(n_j-sigma)-log(1+u)+log_post_beta_bern(Y(positions), P0_params, Y(i))+log_g_j_x0(j)-log_g_j(j);
        //log_probs(j) = log(n_j-sigma)-log(1+u)+eval_log_post_beta_multibern(Y.rows(positions), P0_params, Y.row(i))+log_g_j_x0(j)-log_g_j(j);
        //log_probs(j) = log(n_j-sigma)-log(1+u)+eval_log_post_multi(Y.rows(positions), P0_params, Y.row(i))+log_g_j_x0(j)-log_g_j(j);
      }
    }
    log_g_j_x0(k) = log_g(X.row(i), type, g_params);
    log_probs(k) = log(kappa)+(sigma-1)*log(1+u)+log_prior(P0_params, Y(i))+log_g_j_x0(k);
    //log_probs(k) = log(kappa)+(sigma-1)*log(1+u)+log_prior_reg(P0_params, Y(i), X.row(i))+log_g_j_x0(k);
    //log_probs(k) = log(kappa)+(sigma-1)*log(1+u)+log_prior_beta_bern(P0_params, Y(i))+log_g_j_x0(k);
    //log_probs(k) = log(kappa)+(sigma-1)*log(1+u)+eval_log_prior_beta_multibern(P0_params, Y.row(i))+log_g_j_x0(k);
    //log_probs(k) = log(kappa)+(sigma-1)*log(1+u)+eval_log_prior_multi(P0_params, Y.row(i))+log_g_j_x0(k);
    
    int hnew = sample_log(log_probs);
    cluster_alloc(i) = hnew;
    if (hnew == k){
      clus_count.insert_rows(clus_count.n_rows,vec{1});
      log_g_j.insert_rows(log_g_j.n_rows, vec{log_g_j_x0(k)});
    } else {
      clus_count(hnew) += 1;
      log_g_j(hnew) = log_g_j_x0(hnew);
    }
  }
}

void sample_clus_allocs_Gibbs_old(const mat &YX, const double &u, ivec &cluster_alloc, const vec &type, List &g_params, List &ngg_params, List &P0_params){
  mat X = YX.cols(1, YX.n_cols - 1);
  vec Y = YX.col(0);
  //mat X = YX.cols(2, YX.n_cols - 1);
  //mat Y = YX.cols(0, 1);
  double sigma = ngg_params[0];
  double kappa = ngg_params[1];
  double n = YX.n_rows;
  ivec clus_unique = arma::unique(cluster_alloc);
  vec clus_count(clus_unique.n_elem);
  for (size_t i = 0 ; i < clus_count.n_elem ; i++){
     clus_count(i)= accu(cluster_alloc == clus_unique(i));
  }
  for (size_t i = 0; i < n; i++){
    int c_i = cluster_alloc(i);
    clus_count(c_i) -= 1;
    if(clus_count(c_i)==0){
      clus_count.shed_row(c_i);
      cluster_alloc(arma::find(cluster_alloc > c_i)) -= 1;
    }
    int k = clus_count.n_elem;
    vec log_probs(k+1, arma::fill::zeros);
    for (int j = 0; j < k; ++j) {
      uvec positions = arma::find(cluster_alloc == j);
      uvec d = arma::find(positions==i);
      if(!d.is_empty()){ 
        positions.shed_row(d(0));
      }
      mat x = X.rows(positions);
      mat x_x0 = join_vert(x,X.row(i));
      int n_j = clus_count(j); 
      //log_probs(j) = log(n_j-sigma)-log(1+u)+log_post(Y.rows(positions), P0_params, Y(i))+log_g(x_x0, type, g_params)-log_g(x, type, g_params);
      log_probs(j) = log(n_j-sigma)-log(1+u)+log_post_reg(Y.rows(positions), x, P0_params, Y(i), X.row(i))+log_g(x_x0, type, g_params)-log_g(x, type, g_params);
      //log_probs(j) = log(n_j-sigma)-log(1+u)+log_post_beta_bern(Y.rows(positions), P0_params, Y(i))+log_g(x_x0, type, g_params)-log_g(x, type, g_params);
      //log_probs(j) = log(n_j-sigma)-log(1+u)+eval_log_post_beta_multibern(Y.rows(positions), P0_params, Y.row(i))+log_g(x_x0, type, g_params)-log_g(x, type, g_params);
      //log_probs(j) = log(n_j-sigma)-log(1+u)+eval_log_post_multi(Y.rows(positions), P0_params, Y.row(i))+log_g(x_x0, type, g_params)-log_g(x, type, g_params);
    }
    //log_probs(k) = log(kappa)+(sigma-1)*log(1+u)+log_prior(P0_params, Y(i))+log_g(X.row(i), type, g_params);
    log_probs(k) = log(kappa)+(sigma-1)*log(1+u)+log_prior_reg(P0_params, Y(i), X.row(i))+log_g(X.row(i), type, g_params);
    //log_probs(k) = log(kappa)+(sigma-1)*log(1+u)+log_prior_beta_bern(P0_params, Y(i))+log_g(X.row(i), type, g_params);
    //log_probs(k) = log(kappa)+(sigma-1)*log(1+u)+eval_log_prior_beta_multibern(P0_params, Y.row(i))+log_g(X.row(i), type, g_params);
    //log_probs(k) = log(kappa)+(sigma-1)*log(1+u)+eval_log_prior_multi(P0_params, Y.row(i))+log_g(X.row(i), type, g_params);
    int hnew = sample_log(log_probs);
    cluster_alloc(i) = hnew;
    if (hnew == k){
      clus_count.insert_rows(clus_count.n_rows,vec{1});
    } else {
      clus_count(hnew) += 1;
    }
  }
}

mat get_psm(const imat &clus_alloc_chain){
  int nItems = clus_alloc_chain.n_cols; // Number of items
  int nAllocs = clus_alloc_chain.n_rows;   // Number of allocation vectors
  mat out(nItems, nItems, arma::fill::zeros); 
  for(int i = 0; i < nItems; i++){
    for(int j = 0; j <= i; j++){
      out(i,j) = accu(clus_alloc_chain.col(i) == clus_alloc_chain.col(j));
      out(j,i) = out(i,j);
    }
  }
  return(out / nAllocs);
}

ivec minbinder(const imat &clus_alloc_chain, mat &psm){
  int nAllocs = clus_alloc_chain.n_rows; // Number of allocation vectors
  int nItems = clus_alloc_chain.n_cols; // Number of items
  vec losses(nAllocs, arma::fill::zeros); 
  // Calculate losses
  int check = 0;
  for (int k = 0; k < nAllocs; ++k){
    double sum = 0;
    for (int i = 0; i < nItems; ++i) {
      for (int j = 0; j < i; ++j) {
        check = (clus_alloc_chain(k,i) == clus_alloc_chain(k,j));
        sum = sum + pow(check-psm(i,j),2);
      }
    }
    losses(k) = sum;
  }
  // Find best allocation
  ivec best = clus_alloc_chain.row(arma::index_min(losses)).t();
  return best;
}

ivec VI_LB(const imat C_mat, mat psm_mat){
  vec result(C_mat.n_rows);
  double f = 0.0;
  int n = psm_mat.n_cols;
  vec tvec(n);
  for(uword j = 0; j < C_mat.n_rows; j++){
    f = 0.0;
    for(int i = 0; i < n; i++){
      tvec = psm_mat.col(i);
      f += (log2(accu(C_mat.row(j) == C_mat(j,i))) +
        log2(accu(tvec)) -
        2 * log2(accu(tvec.elem(find(C_mat.row(j).t() == C_mat(j,i))))))/n;
    }
    result(j) = f;
    checkUserInterrupt();
  }
  ivec best = C_mat.row(index_min(result)).t();
  return best;
}

vec compute_freq_entropy(const ivec &cluster_alloc){
  double n_obs = cluster_alloc.n_elem;
  ivec clus_unique = arma::unique(cluster_alloc);
  vec clus_count(clus_unique.n_elem);
  for (size_t i=0 ; i<clus_count.n_elem ;i++){
    clus_count(i)= accu(cluster_alloc == clus_unique(i));
  }
  arma::uword num_top_clusters = std::min(static_cast<arma::uword>(4), clus_unique.n_elem);
  arma::uvec sorted_indices = sort_index(clus_count, "descend");
  arma::vec top_counts(4, arma::fill::zeros);
  top_counts.head(num_top_clusters) = clus_count(sorted_indices.head(num_top_clusters));
  top_counts(1) = top_counts(0) + top_counts(1);
  top_counts(2) = top_counts(1) + top_counts(2);
  top_counts(3) = top_counts(2) + top_counts(3);
  //top_counts(4) = top_counts(3) + top_counts(4);
  //top_counts(5) = top_counts(4) + top_counts(5);
  vec freq = top_counts/n_obs;
  return freq;
}

rowvec post_pred_dens(const mat &YX, const vec &type, const ivec &cluster_alloc, double &u, List &g_params, List &ngg_params, List &P0_params, const rowvec &xnew, const vec &grid){
  mat X = YX.cols(1, YX.n_cols - 1);
  vec Y = YX.col(0);
  double sigma = ngg_params[0];
  double kappa = ngg_params[1];
  ivec clus_unique = arma::unique(cluster_alloc);
  vec clus_count(clus_unique.n_elem); 
  rowvec res(grid.n_elem);
  for (size_t i=0 ; i<clus_count.n_elem ;i++){
    clus_count(i)=accu(cluster_alloc == clus_unique(i));
  }
  int k = clus_count.n_elem;
  vec log_w(k+1);
  vec dens(k+1);
  arma::field<vec> Y_j(k);

  for (int j = 0; j < k; ++j) {
    uvec positions = arma::find(cluster_alloc == clus_unique(j));
    mat X_j = X.rows(positions);
    mat X_j_x0 = join_vert(X_j, xnew);
    Y_j(j) = Y(positions);
    int n_j = clus_count(j);  
    log_w(j) = log(n_j-sigma)-log(1+u)+log_g(X_j_x0, type, g_params)-log_g(X_j, type, g_params);
  }
  log_w(k) = log(kappa)+(sigma-1)*log(1+u)+log_g(xnew, type, g_params);
  double accu_dens = arma::accu(exp(log_w));

  for (uword i = 0; i < grid.n_elem; ++i){
    for (int j = 0; j < k; ++j){
      uvec positions = arma::find(cluster_alloc == clus_unique(j));
      mat X_j = X.rows(positions);
      dens(j) = log_w(j) + log_post(Y_j(j), P0_params, grid(i));
      //dens(j) = log_w(j) + log_post_reg(Y_j(j), X_j, P0_params, grid(i), xnew);
      //dens(j) = log_w(j) + log_post_beta_bern(Y_j(j), P0_params, grid(i));
    }
    dens(k) = log_w(k) + log_prior(P0_params, grid(i));
    //dens(k) = log_w(k) + log_prior_reg(P0_params, grid(i), xnew);
    //dens(k) = log_w(k) + log_prior_beta_bern(P0_params, grid(i));
    res(i) = sum(exp(dens) / accu_dens);
  }
  return res;
}

mat post_pred_dens_multi(const mat &YX, const vec &type, const ivec &cluster_alloc, double &u, List &g_params, List &ngg_params, List &P0_params, const rowvec &xnew,  const vec &grid_x, const vec&grid_y){
  mat X = YX.cols(2, YX.n_cols - 1);
  mat Y = YX.cols(0, 1);
  double sigma = ngg_params[0];
  double kappa = ngg_params[1];
  ivec clus_unique = arma::unique(cluster_alloc);
  vec clus_count(clus_unique.n_elem); 
  mat res(grid_x.n_elem, grid_y.n_elem, arma::fill::zeros);
  for (size_t i=0 ; i<clus_count.n_elem ;i++){
    clus_count(i) = accu(cluster_alloc == clus_unique(i));
  }
  int k = clus_count.n_elem;
  vec dens(k+1);
  vec log_w(k+1);
  arma::field<mat> Y_j(k);

  for (int j=0; j<k; ++j){
    uvec positions = arma::find(cluster_alloc == j);
    mat X_j = X.rows(positions);
    mat X_j_x0 = arma::join_vert(X_j, xnew);
    Y_j(j) = Y.rows(positions);
    int n_j = clus_count(j);  
    log_w(j) = log(n_j-sigma)-log(1+u)+log_g(X_j_x0, type, g_params)-log_g(X_j, type, g_params);
  }
  log_w(k) = log(kappa)+(sigma-1)*log(1+u)+log_g(xnew, type, g_params);
  double accu_dens = arma::accu(exp(log_w));
  for (uword i = 0; i < grid_x.n_elem; ++i){
    for (uword t = 0; t < grid_y.n_elem; ++t){
      rowvec point = {grid_x(i), grid_y(t)};
      for (int j = 0; j < k; ++j){
        dens(j) = log_w(j) + eval_log_post_multi(Y_j(j), P0_params, point);
      }
      dens(k) = log_w(k) + eval_log_prior_multi(P0_params, point);
      res(i,t) = sum(exp(dens) / accu_dens);
    }
  }
  return res;
}

cube post_pred_dens_multi_multi(const mat &YX, const vec &type, const ivec &cluster_alloc, double &u, List &g_params, List &ngg_params, List &P0_params, const mat &xnew,  const vec &grid_x, const vec&grid_y){
  mat X = YX.cols(2, YX.n_cols - 1);
  mat Y = YX.cols(0, 1);
  double sigma = ngg_params[0];
  double kappa = ngg_params[1];
  ivec clus_unique = arma::unique(cluster_alloc);
  vec clus_count(clus_unique.n_elem);
  cube result(grid_x.n_elem, grid_y.n_elem, xnew.n_rows); 
  mat res(grid_x.n_elem, grid_y.n_elem, arma::fill::zeros);
  for (size_t i=0 ; i<clus_count.n_elem ;i++){
    clus_count(i) = accu(cluster_alloc == clus_unique(i));
  }
  int k = clus_count.n_elem;
  arma::field<mat> Y_j(k);
  arma::vec log_g_j(k);

  for (int j=0; j<k; ++j){
      uvec positions = arma::find(cluster_alloc == j);
      mat X_j = X.rows(positions);
      Y_j(j) = Y.rows(positions);
      log_g_j(j) = log_g(X_j, type, g_params);
  }

  for (uword r = 0; r < xnew.n_rows; ++r){
    vec dens(k+1);
    vec log_w(k+1);
    for (int j=0; j<k; ++j){
      uvec positions = arma::find(cluster_alloc == j);
      mat X_j = X.rows(positions);
      mat X_j_x0 = arma::join_vert(X_j, xnew.row(r));
      int n_j = clus_count(j);  
      log_w(j) = log(n_j-sigma)-log(1+u)+log_g(X_j_x0, type, g_params)-log_g_j(j);
    }
    log_w(k) = log(kappa)+(sigma-1)*log(1+u)+log_g(xnew.row(r), type, g_params);
    double accu_dens = arma::accu(exp(log_w));

    for (uword i = 0; i < grid_x.n_elem; ++i){
      for (uword t = 0; t < grid_y.n_elem; ++t){
        rowvec point = {grid_x(i), grid_y(t)};
        for (int j = 0; j < k; ++j){
          dens(j) = log_w(j) + eval_log_post_multi(Y_j(j), P0_params, point);
        }
        dens(k) = log_w(k) + eval_log_prior_multi(P0_params, point);
        res(i,t) = sum(exp(dens) / accu_dens);
      }
    }
    result.slice(r) = res;
  }
  return result;
}

rowvec post_pred1(const mat &YX, const vec &type, const ivec &cluster_alloc, double &u, List &g_params, List &ngg_params, List &P0_params, const mat &Xnew, const vec &grid){
  mat X = YX.cols(1, YX.n_cols - 1);
  vec Y = YX.col(0);
  double sigma = ngg_params[0];
  double kappa = ngg_params[1];
  ivec clus_unique = arma::unique(cluster_alloc);
  vec clus_count(clus_unique.n_elem); 
  rowvec result(Xnew.n_rows);
  for (size_t i=0 ; i<clus_count.n_elem ;i++){
     clus_count(i)=accu(cluster_alloc == clus_unique(i));
  }
  int k = clus_count.n_elem;
  field<arma::mat> X_j(k);
  field<vec> Y_j(k);
  vec log_g_elem(k);

  for (int j=0; j<k; ++j){
    uvec positions = arma::find(cluster_alloc == j);
    X_j(j) = X.rows(positions);
    log_g_elem(j) = log_g(X_j(j), type, g_params);
    Y_j(j) = Y(positions);
  }
  for (uword i = 0; i < Xnew.n_rows; ++i){
    vec res_i(grid.n_elem);
    vec log_w(k+1);
    vec dens(k+1);
    rowvec xi = Xnew.row(i);
    for (int j = 0; j < k; ++j) {
      mat x_x0 = arma::join_vert(X_j(j),xi); 
      log_w(j) = log(clus_count(j)-sigma)-log(1+u)+log_g(x_x0, type,g_params)-log_g_elem(j);
    }
    log_w(k) = log(kappa)+(sigma-1)*log(1+u)+log_g(xi, type, g_params);
    double accu_dens = accu(exp(log_w));
    for (uword t = 0; t < grid.n_elem; ++t){
      for (int j = 0; j < k; ++j){
        dens(j) = log_w(j) + log_post(Y_j(j), P0_params, grid(t));
        //dens(j) = log_w(j) + log_post_reg(Y_j(j), X_j(j), P0_params, grid(t), xi);
        //dens(j) = log_w(j) + log_post_beta_bern(Y_j(j), P0_params, grid(t));
      }
      dens(k) = log_w(k) + log_prior(P0_params, grid(t));
      //dens(k) = log_w(k) + log_prior_reg(P0_params, grid(t), xi);
      //dens(k) = log_w(k) + log_prior_beta_bern(P0_params, grid(t));
      res_i(t) = sum(exp(dens) / accu_dens);
    }
    double som = arma::accu(res_i);
    double y_pred = dot(res_i, grid) / som;
    result(i) = y_pred;
  }
  return result;
}

mat post_pred1_multi(const mat &YX, const vec &type, const ivec &cluster_alloc, double &u, List &g_params, List &ngg_params, List &P0_params, const mat &Xnew, const vec &grid_y1, const vec &grid_y2){
  mat X = YX.cols(2, YX.n_cols - 1);
  mat Y = YX.cols(0, 1);
  double sigma = ngg_params[0];
  double kappa = ngg_params[1];
  ivec clus_unique = arma::unique(cluster_alloc);
  vec clus_count(clus_unique.n_elem); 
  mat result(Xnew.n_rows, 2);
  for (size_t i=0 ; i<clus_count.n_elem ;i++){
     clus_count(i)=accu(cluster_alloc == clus_unique(i));
  }
     
  int k = clus_count.n_elem;
  arma::field<arma::mat> X_j(k);
  arma::field<arma::mat> Y_j(k);
  vec log_g_elem(k);
  for (int j=0; j<k; ++j){
    uvec positions = arma::find(cluster_alloc == j);
    X_j(j) = X.rows(positions);
    log_g_elem(j) = log_g(X_j(j), type, g_params);
    Y_j(j) = Y.rows(positions);
  }
  for (uword i = 0; i < Xnew.n_rows; ++i){
    mat res_i(grid_y1.n_elem, grid_y2.n_elem);
    vec log_w(k+1);
    vec dens(k+1);
    rowvec xi = Xnew.row(i);
    for (int j = 0; j < k; ++j) {
      mat x_x0 = arma::join_vert(X_j(j),xi); 
      log_w(j) = log(clus_count(j)-sigma)-log(1+u)+log_g(x_x0, type,g_params)-log_g_elem(j);
    }
    log_w(k) = log(kappa)+(sigma-1)*log(1+u)+log_g(xi, type, g_params);
    double accu_dens = accu(exp(log_w));
    for (uword r = 0; r < grid_y1.n_elem; ++r){
      for (uword t = 0; t < grid_y2.n_elem; ++t){
        rowvec point = {grid_y1(r), grid_y2(t)};
        for (int j = 0; j < k; ++j){
          dens(j) = log_w(j) + eval_log_post_multi(Y_j(j), P0_params, point);
        }
        dens(k) = log_w(k) + eval_log_prior_multi(P0_params, point);
        res_i(r,t) = sum(exp(dens) / accu_dens);
      }
    }
    vec marg_y1 = arma::sum(res_i, 1);
    double som_y1 = accu(marg_y1);
    vec marg_y2 = arma::sum(res_i, 0).t();
    double som_y2 = accu(marg_y2);
    result(i, 0) = dot(marg_y1, grid_y1) / som_y1;
    result(i, 1) = dot(marg_y2, grid_y2) / som_y2;
  }
  return result;
}

rowvec post_pred2(const mat &YX, const vec &type, const ivec &cluster_alloc, double &u, List &g_params, List &ngg_params, List &P0_params, const mat &Xnew){
  mat X = YX.cols(1, YX.n_cols - 1);
  vec Y = YX.col(0);
  double sigma = ngg_params[0];
  double kappa = ngg_params[1];
  ivec clus_unique = arma::unique(cluster_alloc);
  vec clus_count(clus_unique.n_elem); 
  rowvec result(Xnew.n_rows);
  for (size_t i=0 ; i<clus_count.n_elem ;i++){
     clus_count(i)=accu(cluster_alloc == clus_unique(i));
     }
     
  int k = clus_count.n_elem;
  field<arma::mat> X_j(k);
  field<vec> Y_j(k);
  vec log_g_elem(k);

  for (int j=0; j<k; ++j){
    uvec positions = arma::find(cluster_alloc == j);
    X_j(j) = X.rows(positions);
    log_g_elem(j) = log_g(X_j(j), type, g_params);
  }
  
  for (int j=0; j<k; ++j){
    uvec positions = arma::find(cluster_alloc == j);
    X_j(j) = X.rows(positions);
    log_g_elem(j) = log_g(X_j(j), type, g_params);
  }
  for (uword i = 0; i < Xnew.n_rows; ++i){
    vec log_w(k+1);
    rowvec xi = Xnew.row(i);
    for (int j = 0; j < k; ++j) {
      mat x_x0 = arma::join_vert(X_j(j),xi); 
      log_w(j) = log(clus_count(j)-sigma)-log(1+u)+log_g(x_x0, type,g_params)-log_g_elem(j);
    }
    log_w(k) = log(kappa)+(sigma-1)*log(1+u)+log_g(xi, type, g_params);
    int hnew = sample_log(log_w);
    double y_pred;
    
    if (hnew == k){
      y_pred = log_prior(P0_params, 0, true);
      //y_pred = log_prior_reg(P0_params, 0, xi, true);
      //y_pred = log_prior_beta_bern(P0_params, 0, true);
    }
    else{
      uvec positions = arma::find(cluster_alloc == hnew);
      vec Y_j = Y(positions);
      mat X_j = X.rows(positions);
      y_pred = log_post(Y_j, P0_params, 0, true);
      //y_pred = log_post_reg(Y_j, X_j, P0_params, 0, xi, true);
      //y_pred = log_post_beta_bern(Y_j, P0_params, 0, true);
    }
    result(i) = y_pred;
  }
  return result;
}

mat post_pred2_multi(const mat &YX, const vec &type, const ivec &cluster_alloc, double &u, List &g_params, List &ngg_params, List &P0_params, const mat &Xnew){
  //mat X = YX.cols(2, YX.n_cols - 1);
  //mat Y = YX.cols(0, 1);
  mat X = YX.cols(6, YX.n_cols - 1);
  mat Y = YX.cols(0, 5);
  double sigma = ngg_params[0];
  double kappa = ngg_params[1];
  ivec clus_unique = arma::unique(cluster_alloc);
  vec clus_count(clus_unique.n_elem); 
  mat result(Xnew.n_rows, Y.n_cols);
  for (size_t i=0 ; i<clus_count.n_elem ;i++){
     clus_count(i)=accu(cluster_alloc == clus_unique(i));
  }
  int k = clus_count.n_elem;
  arma::field<arma::mat> X_j(k);
  vec log_g_elem(k);

  for (int j=0; j<k; ++j){
    uvec positions = arma::find(cluster_alloc == j);
    X_j(j) = X.rows(positions);
    log_g_elem(j) = log_g(X_j(j), type, g_params);
  }
  for (uword i = 0; i < Xnew.n_rows; ++i){
    vec log_w(k+1);
    rowvec xi = Xnew.row(i);
    for (int j = 0; j < k; ++j) {
      mat x_x0 = arma::join_vert(X_j(j),xi); 
      log_w(j) = log(clus_count(j)-sigma)-log(1+u)+log_g(x_x0, type,g_params)-log_g_elem(j);
    }
    log_w(k) = log(kappa)+(sigma-1)*log(1+u)+log_g(xi, type, g_params);
    int hnew = sample_log(log_w);
    rowvec y_pred(Y.n_cols);
    if (hnew == k){
      //y_pred = gen_prior_multi(P0_params);
      y_pred = gen_prior_beta_multibern(P0_params);
    }
    else{
      uvec positions = arma::find(cluster_alloc == hnew);
      mat Y_j = Y.rows(positions);
      //mat X_j = X.rows(positions);
      //y_pred = gen_post_multi(Y_j, P0_params);
      y_pred = gen_post_beta_multibern(Y_j, P0_params);
    }
    result.row(i) = y_pred;
  }
  return result;
}

double eval_post_pred(const mat &YX, vec &type, const ivec &cluster_alloc, double &u, List &g_params, List &ngg_params, List &P0_params, const rowvec &xnew, double &y0){
  mat X = YX.cols(1, YX.n_cols - 1);
  vec Y = YX.col(0);
  double sigma = ngg_params[0];
  double kappa = ngg_params[1];
  ivec clus_unique = arma::unique(cluster_alloc);
  vec clus_count(clus_unique.n_elem); 
  for (size_t i=0 ; i<clus_count.n_elem ;i++){
     clus_count(i)=accu(cluster_alloc == clus_unique(i));
  }

  int k = clus_count.n_elem;
  vec log_w(k+1, arma::fill::zeros);
  vec dens(k+1,arma::fill::zeros);
  field<arma::mat> X_j(k);
  field<vec> Y_j(k);
  field<uvec> positions(k);
  vec log_g_elem(k);
  ivec n_j(k);
  for (int j=0; j<k; ++j){
    positions(j) = arma::find(cluster_alloc == clus_unique(j));
    X_j(j) = X.rows(positions(j));
    log_g_elem(j) = log_g(X_j(j), type, g_params);
    Y_j(j) = Y.rows(positions(j));
    n_j(j) = clus_count(j);  
  }
  
  //compute vector of cluster allocation weights
  for (int j = 0; j < k; ++j) {
    mat x_x0 = arma::join_vert(X_j(j),xnew); 
    log_w(j) = log(n_j(j)-sigma)-log(1+u)+log_g(x_x0, type,g_params)-log_g_elem(j);
  }
  log_w(k) = log(kappa)+(sigma-1)*log(1+u)+log_g(xnew, type, g_params);
  double sum = arma::sum(exp(log_w));
  arma::vec w = exp(log_w)/sum;
  
  for (size_t j = 0; j<dens.n_rows-1; ++j){
    dens(j) = w(j) * exp(log_post(Y_j(j), P0_params, y0));
  }
  dens(k) = w(k) * exp(log_prior(P0_params, y0));
  return arma::sum(dens);
}

//--------------------------------------------------------------------------------------------

//[[Rcpp::export]]
List run_mcmc(const mat &data, const vec &type, List &g_params, List &ngg_params, List &P0_params, int niter=200, int nburn=100, int thin=1, int nGibbs=0, int thin_Gibbs=0, bool display_progress=true){
  ivec cluster_alloc = regspace<ivec>(0,data.n_rows-1);
  imat clusChain(data.n_rows, (niter-nburn)/thin);
  vec uchain((niter-nburn)/thin);
  mat freqchain(4, (niter-nburn)/thin);
  double u_0 = 1;
  double accept_rate_u = 0;
  double n_split = 0;
  double n_merge = 0;
  double accept_rate_split = 0;
  double accept_rate_merge = 0;
  List result;
  
  
  Progress p(niter, display_progress);
  for (int i = 0; i < niter; ++i){
    update_u(u_0, accept_rate_u, ngg_params, cluster_alloc);
    if (i % thin_Gibbs == 0){
      sample_clus_allocs_Gibbs(data, u_0, cluster_alloc, type, g_params, ngg_params, P0_params);
    }else{
      sample_clus_allocs_Sp_Mrg(data, cluster_alloc, u_0, type, g_params, ngg_params, P0_params, accept_rate_split, accept_rate_merge, n_split, n_merge, nGibbs);
    }
    if (i >= nburn && i % thin == 0){
      clusChain.col((i-nburn)/thin) = cluster_alloc;
      uchain((i-nburn)/thin) = u_0;
      freqchain.col((i-nburn)/thin) = compute_freq_entropy(cluster_alloc);
    }
  p.increment();
  }

  if (nburn<niter){
    mat psm = get_psm(clusChain.t());
    ivec best_binder = minbinder(clusChain.t(), psm);
    //ivec best_binder = VI_LB(clusChain.t(), psm);
    result["clus_chain"] = clusChain;
    result["u_chain"] = uchain;
    result["best_clus_binder"] = best_binder;
    result["freq_chain"] = freqchain;
    if (thin_Gibbs != 1){
      result["n_split"] = n_split;
      result["n_merge"] = n_merge;
      result["accept_split"] = accept_rate_split/n_split;
      result["accept_merge"] = accept_rate_merge/n_merge;
    }
  }
  return result;
}

//[[Rcpp::export]]
double time_run(const mat &data_tot, const vec &type, List &g_params, List &ngg_params, List &P0_params, int niter=200, int nburn=100, int thin=1, int itermean=50, int nGibbs=0, int thin_Gibbs=0, bool display_progress=true){
  vec time(itermean);
  for (int i = 0; i < itermean; i++){
    uvec indices = arma::randperm(data_tot.n_rows,2000);   //cambio NUMERO 
    mat data_sample = data_tot.rows(indices);
    auto start = std::chrono::high_resolution_clock::now();
    List r1 = run_mcmc(data_sample, type, g_params, ngg_params, P0_params, niter, nburn, thin, nGibbs, thin_Gibbs, display_progress);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    time(i) = elapsed.count();
  }
  return arma::mean(time);
}

//[[Rcpp::export]] 
List run_mcmc_pred(const mat &data, const vec &type, List &g_params, List &ngg_params, List &P0_params, const rowvec &y_true, const mat &Xnew, const vec &grid, int niter=200, int nburn=100, int thin=1, int nGibbs=0, int thin_Gibbs=0, bool pred_dense=false, bool pred1=false, bool pred2=false, bool display_progress=true){
  ivec cluster_alloc = regspace<ivec>(0,data.n_rows-1);
  imat clusChain(data.n_rows, (niter-nburn)/thin);
  vec uChain((niter-nburn)/thin);
  mat freqchain(4, (niter-nburn)/thin);
  vec n_clust((niter-nburn)/thin);
  rowvec pred_grid(grid.n_elem);
  rowvec y_pred1(Xnew.n_rows);
  rowvec y_pred2(Xnew.n_rows);
  double rmse1 = 0, rmse2 = 0;
  double u_0 = 1; 
  double accept_rate_u = 0;
  double n_split = 0;
  double n_merge = 0;
  double accept_rate_split = 0;
  double accept_rate_merge = 0;
  List result;


  Progress p(niter, display_progress);
  for (int i=0; i<niter; ++i){
    update_u(u_0, accept_rate_u, ngg_params, cluster_alloc);
    if (i % thin_Gibbs == 0){
      sample_clus_allocs_Gibbs(data, u_0, cluster_alloc, type, g_params, ngg_params, P0_params);
    }else{
      sample_clus_allocs_Sp_Mrg(data, cluster_alloc, u_0, type, g_params, ngg_params, P0_params, accept_rate_split, accept_rate_merge, n_split, n_merge, nGibbs);
    }
    if (i >= nburn && i % thin == 0){
      clusChain.col((i-nburn)/thin) = cluster_alloc;
      uChain((i-nburn)/thin) = u_0;
      n_clust((i-nburn)/thin) = ((ivec)arma::unique(cluster_alloc)).n_elem;
      freqchain.col((i-nburn)/thin) = compute_freq_entropy(cluster_alloc);
      if(pred_dense){
        pred_grid += post_pred_dens(data, type, cluster_alloc, u_0, g_params, ngg_params, P0_params, Xnew.row(0), grid);
      }
      if(pred1){
        y_pred1 += post_pred1(data, type, cluster_alloc, u_0, g_params, ngg_params, P0_params, Xnew, grid);
      }
      if(pred2){
        y_pred2 += post_pred2(data, type, cluster_alloc, u_0, g_params, ngg_params, P0_params, Xnew);
      }
    }
    p.increment();
  }
  
  if (pred1) {
    y_pred1 = y_pred1/((niter-nburn)/thin);
    rmse1 = sqrt(arma::mean(arma::square((y_true - y_pred1))));
  }
  if (pred2) {
    y_pred2 = y_pred2/((niter-nburn)/thin); 
    rmse2 = sqrt(arma::mean(arma::square((y_true - y_pred2))));
    //rowvec y_pred_round = round(y_pred2);
    //rmse2 = accu(abs(y_true - y_pred_round))/ (y_true.n_elem); //Hamming Loss
  }
  mat psm = get_psm(clusChain.t());
  ivec best_binder = minbinder(clusChain.t(), psm);
  result["clus_chain"] = clusChain;
  result["u_chain"] = uChain;
  result["best_clus_binder"] = best_binder;
  result["freq_chain"] = freqchain;
  result["pred_grid"] = pred_grid/((niter-nburn)/thin);
  result["y_pred1"] = y_pred1;
  result["y_pred2"] = y_pred2;
  result["rmse1"] = rmse1;
  result["rmse2"] = rmse2;
  result["n_clust_mean"] = mean(n_clust);
  if (thin_Gibbs != 1){
    result["n_split"] = n_split;
    result["n_merge"] = n_merge;
    result["accept_split"] = accept_rate_split/n_split;
    result["accept_merge"] = accept_rate_merge/n_merge;
  }
  return result;
}

//[[Rcpp::export]]
List run_mcmc_pred_multi(const mat &data, const vec &type, List &g_params, List &ngg_params, List &P0_params, const mat &y_true, const mat &Xnew, const vec &grid_y1, const vec &grid_y2, int niter=200, int nburn=100, int thin=1, int nGibbs=0, int thin_Gibbs=0, bool pred_dense=false, bool pred1=false, bool pred2=false, bool display_progress=true){
  ivec cluster_alloc = regspace<ivec>(0,data.n_rows-1);
  imat clusChain(data.n_rows, (niter-nburn)/thin);
  vec uChain((niter-nburn)/thin);
  mat freqchain(5, (niter-nburn)/thin);
  vec n_clust((niter-nburn)/thin);
  //cube pred_grid(grid_y1.n_elem, grid_y2.n_elem, Xnew.n_rows);
  mat pred_grid(grid_y1.n_elem, grid_y2.n_elem);
  double dim = 6;
  mat y_pred1(Xnew.n_rows, dim);
  mat y_pred2(Xnew.n_rows, dim); 
  double rmse1 = 0, rmse2 = 0;
  double u_0 = 1; 
  double accept_rate_u = 0;
  double n_split = 0;
  double n_merge = 0;
  double accept_rate_split = 0;
  double accept_rate_merge = 0;
  List result;

  
  Progress p(niter, display_progress);
  for (int i=0; i<niter; ++i){
    update_u(u_0, accept_rate_u, ngg_params, cluster_alloc);
    if (i % thin_Gibbs == 0){
      sample_clus_allocs_Gibbs(data, u_0, cluster_alloc, type, g_params, ngg_params, P0_params);
    }else{
      sample_clus_allocs_Sp_Mrg(data, cluster_alloc, u_0, type, g_params, ngg_params, P0_params, accept_rate_split, accept_rate_merge, n_split, n_merge, nGibbs);
    }
    if (i >= nburn && i % thin == 0){
      clusChain.col((i-nburn)/thin) = cluster_alloc;
      uChain((i-nburn)/thin) = u_0;
      freqchain.col((i-nburn)/thin) = compute_freq_entropy(cluster_alloc);
      n_clust((i-nburn)/thin) = ((ivec)arma::unique(cluster_alloc)).n_elem;
      if(pred_dense){
        /*
        cube post_pred = post_pred_dens_multi_multi(data, type, cluster_alloc, u_0, g_params, ngg_params, P0_params, Xnew.row(0), grid_y1, grid_y2);
        for ( uword r = 0; r < Xnew.n_rows; ++r){
        pred_grid.slice(r) += post_pred.slice(r);
        }
        */
       pred_grid  += post_pred_dens_multi(data, type, cluster_alloc, u_0, g_params, ngg_params, P0_params, Xnew.row(0), grid_y1, grid_y2);
      }
      if(pred1){
        y_pred1 += post_pred1_multi(data, type, cluster_alloc, u_0, g_params, ngg_params, P0_params, Xnew, grid_y1, grid_y2);     
      }
      if(pred2){
        y_pred2 += post_pred2_multi(data, type, cluster_alloc, u_0, g_params, ngg_params, P0_params, Xnew);
      }
    }
    p.increment();
  }

  if (pred1) {
    y_pred1 = y_pred1/((niter-nburn)/thin);
    rmse1 = sqrt(arma::mean(arma::sum(arma::square(y_true - y_pred1), 1)));
  }
  if (pred2) {
    y_pred2 = y_pred2/((niter-nburn)/thin);
    //rmse2 = sqrt(arma::mean(arma::sum(arma::square(y_true - y_pred2), 1)));
    mat y_pred_round = round(y_pred2);
    rmse2 = accu(sum(abs(y_true - y_pred_round), 1)) / (y_true.n_cols*y_true.n_rows); //Hamming Loss
  }
  mat psm = get_psm(clusChain.t());
  ivec best_binder = minbinder(clusChain.t(), psm);
  //ivec best_binder = VI_LB(clusChain.t(), psm);
  result["clus_chain"] = clusChain;
  result["u_chain"] = uChain;
  result["best_clus_binder"] = best_binder;
  result["freq_chain"] = freqchain;
  result["pred_grid"] = pred_grid/((niter-nburn)/thin);
  result["y_pred1"] = y_pred1;
  result["y_pred2"] = y_pred2;
  result["rmse1"] = rmse1;
  result["rmse2"] = rmse2;
  result["n_clust_mean"] = mean(n_clust);
  if (thin_Gibbs != 1){
    result["n_split"] = n_split;
    result["n_merge"] = n_merge;
    result["accept_split"] = accept_rate_split/n_split;
    result["accept_merge"] = accept_rate_merge/n_merge;
  }
  return result;
}

// [[Rcpp::export]]
mat compute_lambda(mat X, int nrep, mat invcov,vec type){
  mat increment(nrep,2);
  uvec indices_bin = find(type == 0);
  uvec indices_cat = find(type == -1);
  uvec indices_cont = find(type == 1);
  mat cont = X.cols(indices_cont);
  mat bin = X.cols(indices_bin);
  mat cat = X.cols(indices_cat);
  double cont_frac = (double)(cont.n_cols)/X.n_cols;
  double bin_frac = (double)(bin.n_cols)/X.n_cols;
  double cat_frac = (double)(cat.n_cols)/X.n_cols;
  for(int i = 0; i < nrep; i++){
    // sample a size, and a subset of elements
    int sample_size = randi(distr_param(2, X.n_rows));
    uvec indtot = conv_to<uvec>::from(randi(sample_size, distr_param(0, X.n_rows - 1)));
    uvec ind = randi<uvec>(1, distr_param(0, X.n_rows - 1));

    for(int j=0; j<sample_size;j++){
       while(ind(0) == indtot(j)) {
        ind(0) = randi<uword>(distr_param(0, X.n_rows - 1));
    }
    }

    uvec combined_indices = join_cols(ind, indtot);
    mat cat_sample = cat.rows(indtot);
    mat cat_new = cat.rows(combined_indices);
    vec old_c(sample_size),news_c(sample_size+1);
    vec old_b(sample_size),news_b(sample_size+1);
    vec old_cat(sample_size),news_cat(sample_size+1);

    //dist_cont
    if (!cont.is_empty()){
      mat cont_sample = cont.rows(indtot);
      mat cont_new = cont.rows(combined_indices);
      rowvec centroid_cont = mean(cont_sample, 0);
      rowvec centroid_cont_new = mean(cont_new, 0);
      for(uword r = 0; r < cont_sample.n_rows; r++){
        old_c(r) = sqrt(as_scalar((cont_sample.row(r) - centroid_cont) * invcov * (cont_sample.row(r) - centroid_cont).t()));
      }
      for(uword r = 0; r < cont_new.n_rows; r++){
        news_c(r) = sqrt(as_scalar((cont_new.row(r) - centroid_cont_new) * invcov * (cont_new.row(r) - centroid_cont_new).t()));
      }
    }

    //dist_bin
    if (!bin.is_empty()){
      mat bin_sample = bin.rows(indtot);
      mat bin_new = bin.rows(combined_indices);
      rowvec centroid_bin = round(mean(bin_sample, 0));
      rowvec centroid_bin_new = round(mean(bin_new, 0));
      for (uword r = 0; r < bin_sample.n_rows; r++) {
        rowvec diff_bin = abs(bin_sample.row(r) - centroid_bin);
        old_b(r) = accu(diff_bin) / bin_sample.n_cols;
      }
      for (uword r = 0; r < bin_new.n_rows; r++) {
        rowvec diff_bin2 = abs(bin_new.row(r) - centroid_bin_new);
        news_b(r) = accu(diff_bin2) /bin_new.n_cols;
      }
    }

    //dist cat
    if (!cat.is_empty()){
      mat cat_sample = cat.rows(indtot);
      mat cat_new = cat.rows(combined_indices);
      rowvec centroid_cat = compute_mode(cat_sample);
      rowvec centroid_cat_new = compute_mode(cat_new);
      for (uword r = 0; r < cat_sample.n_rows; r++) {
        old_cat(r) = sum(cat_sample.row(r) != centroid_cat)/cat_sample.n_cols;
      }
      for (uword r = 0; r < cat_new.n_rows; r++) {
        news_cat(r) = sum(cat_new.row(r)!= centroid_cat_new)/cat_new.n_cols;
      }
    }
    //weighted distance old and new 
    double dist_new = accu(cont_frac * news_c + bin_frac * news_b + cat_frac * news_cat);
    double dist_old = accu(cont_frac * old_c + bin_frac * old_b + cat_frac * old_cat);
    
    //increment 
    increment(i,0) = dist_new - dist_old;
    increment(i,1) = sample_size;
  }
  
  return(increment);
}

int sum_error(ivec& vettore) {
  int length = vettore.size();
  if (length > 1) {
    int sum = 0;
    for (int i = 0; i < length - 1; ++i) {
      sum += vettore[i];
    }
    return sum;
  } else {
    return 0;
  }
}

long double calculate_misclassification_error(arma::ivec& labels) {
 
  arma::ivec freq1 = conv_to<arma::ivec>::from(hist(labels.subvec(0, 74)));
  arma::ivec freq2 = conv_to<arma::ivec>::from(hist(labels.subvec(75, 149)));
  arma::ivec freq3 = conv_to<arma::ivec>::from(hist(labels.subvec(150, 199)));
  freq1 = sort(freq1);
  freq2 = sort(freq2);
  freq3 = sort(freq3);
  
  int s1 = sum_error(freq1);
  int s2 = sum_error(freq2);
  int s3 = sum_error(freq3);
  
  long double misclassification_rate = (s1+s2+s3)/200.0;

  return misclassification_rate;
}

double sample_mu(const vec &y,const double &sigma2, List &P0_params){
  double mu0 = P0_params[0];
  double lambda0 = P0_params[1];
  double n = y.n_elem;  
  double y_mean = arma::mean(y);
  double mu_p = (n*y_mean+lambda0*mu0)/(n+lambda0);
  double lambda_p = lambda0+n;
  return R::rnorm(mu_p, sigma2/(lambda_p));
}

double sample_sigma2(const vec &y, List &P0_params){
  double mu0 = P0_params[0];
  double lambda0 = P0_params[1];
  double a0 = P0_params[2];
  double b0 = P0_params[3];
  double n = y.n_elem;  
  double y_mean = arma::mean(y);
  double a_p = a0 + 0.5*n;
  double b_p = b0 + 0.5*(arma::accu(pow(y-y_mean*arma::ones(y.n_elem,1),2)))+0.5*(n*lambda0*pow(mu0-y_mean,2))/(n+lambda0);
  double scale = 1.0/b_p;
  return 1.0/R::rgamma(a_p, scale);
  //return 1.0 / randg(distr_param(a_p, 1 / b_p));
}

double sample_lambda(const vec &y, List &P0_params){
  double a0 = P0_params[0];
  double b0 = P0_params[1];
  double n = y.n_elem; 
  double a_p = sum(y) + a0;
  double b_p = n + b0;
  return R::rgamma(a_p, 1/b_p);

}

rowvec sample_Mu(const mat &y, const mat &S, List &P0_params){
  double n = y.n_rows;
  rowvec mu0 = P0_params[0];
  double k0 = P0_params[1];
  
  rowvec mean = arma::mean(y, 0);
  rowvec mun = (k0*mu0 + n*mean)/(k0 + n);
  double kn = k0 + n;
  return arma::mvnrnd(mun.t(), S/kn).t();
}

mat sample_S(const mat &y, List &P0_params){
  double n = y.n_rows;
  rowvec mu0 = P0_params[0];
  double k0 = P0_params[1];
  mat S0 = P0_params[2];
  double v0 = P0_params[3];

  rowvec mean = arma::mean(y, 0);
  mat Sn = S0 + (y.each_row()-mean).t()*(y.each_row()-mean) + (k0*n)/(k0 + n) * (mean-mu0).t()*(mean-mu0);
  double vn = v0 + n;
  return arma::iwishrnd(Sn, vn);
}

double sample_theta(const vec &y, List &P0_params){
  double a0 = P0_params[0];
  double b0 = P0_params[1];
  double n = y.n_elem; 
  double a_p = sum(y) + a0;
  double b_p = n - sum(y) + b0;
  return R::rbeta(a_p, b_p);
}

rowvec sample_theta_multibern(const mat &y, List &P0_params){
  mat B = P0_params[0];
  double n = y.n_rows; 
  rowvec res(B.n_cols);
  for ( uword i = 0; i < y.n_cols; i++){
    double a_p = sum(y.col(i)) + B(1, i);
    double b_p = n - sum(y.col(i)) + B(0, i);
    res(i) = R::rbeta(a_p, b_p);
  }
  return res;
}

List sample_params_reg(const vec y, const mat x_intrcpt, List &P0_params){
  vec beta0 = P0_params[0];
  mat B0 = P0_params[1];
  double a0 = P0_params[2];
  double b0 = P0_params[3]; 
  double n = y.n_elem;  
  mat B_n = inv(inv(B0) + x_intrcpt.t()*x_intrcpt);
  vec beta_n = B_n*(x_intrcpt.t()*y + inv(B0)*beta0);
  double a_n = 2*a0 + n;
  double b_n = 2*b0 + as_scalar(beta0.t()*inv(B0)*beta0) + as_scalar(y.t()*y) - as_scalar(beta_n.t()*inv(B_n)*beta_n);
  double sigma2 =  1.0/R::rgamma(a_n, 1.0/b_n);
  rowvec beta = arma::mvnrnd(beta_n, sigma2 * B_n).t();
  List params;
  params["sigma2"] = sigma2;
  params["beta"] = beta;
  return params;
}

// [[Rcpp::export]]
double LPML(const mat &data, const vec &type, const imat &clusChain, vec &uChain, List &P0_params, List &ngg_params, List & g_params, bool display_progress = true){
  vec Y = data.col(0);
  mat X = data.cols(1, data.n_cols - 1);
  int n = clusChain.n_rows;
  int m = clusChain.n_cols;
  vec CPO_i(n);
  mat likhood(n,m);
  ivec cluster_alloc_t(n);
  double sigma = ngg_params[0];
  double u = 0.0;

  Progress p(m, display_progress);
  for (int t = 0; t < m; t++){
    cluster_alloc_t = clusChain.col(t);
    u = uChain(t);
    ivec clus_unique = arma::unique(cluster_alloc_t);
    vec clus_count(clus_unique.n_elem); 
    for (size_t i=0 ; i<clus_count.n_elem ;i++){
      clus_count(i)=accu(cluster_alloc_t == clus_unique(i));
    }
    int k = clus_count.n_elem;
    vec dens(k);
    field<arma::mat> X_j(k);
    vec log_g_elem(k);
    vec sigma2(k);
    vec mu(k);
    for (int j = 0; j < k; ++j){
      uvec positions = arma::find(cluster_alloc_t == j);
      X_j(j) = X.rows(positions);
      log_g_elem(j) = log_g(X_j(j), type, g_params);
      mat Y_j = Y.rows(positions);  
      sigma2(j) = sample_sigma2(Y_j, P0_params);
      mu(j) = sample_mu(Y_j, sigma2(j), P0_params);
    }
    for (int i = 0; i < n; i++){
      double accu_dens = 0.0;
      double ynew = Y(i);
      rowvec xnew = X.row(i);
      //compute vector of cluster allocation weights
      for (int j = 0; j < k; ++j) {
        mat x_x0 = arma::join_vert(X_j(j), xnew); 
        double log_w = log(clus_count(j)-sigma)-log(1+u)+log_g(x_x0, type,g_params)-log_g_elem(j);
        dens(j) = log_w + d_norm(ynew, mu(j), sigma2(j), true);
        accu_dens += exp(log_w);
      }
    likhood(i, t) = sum(exp(dens) / accu_dens); 
    }
  p.increment();
  }
  for (int i = 0; i < n; ++i) {
    CPO_i(i) = 1.0 / arma::mean(1.0 / likhood.row(i));
  }
  return arma::sum(log(CPO_i)); 
}

// [[Rcpp::export]]
double LPML_new(const mat &data, const vec &type, const imat &clusChain, List &P0_params, bool display_progress = true){
  vec Y = data.col(0);
  int n = clusChain.n_rows;
  int m = clusChain.n_cols;
  vec CPO_i(n);
  mat likhood(n,m);
  ivec cluster_alloc_t(n);

  Progress p(m, display_progress);
  for (int t = 0; t < m; t++){
    cluster_alloc_t = clusChain.col(t);
    ivec clus_unique = arma::unique(cluster_alloc_t);
    vec clus_count(clus_unique.n_elem); 
    for (size_t i=0 ; i<clus_count.n_elem ;i++){
      clus_count(i)=accu(cluster_alloc_t == clus_unique(i));
    }
    int k = clus_count.n_elem;
    field<arma::mat> Y_j(k);

    for (int j = 0; j < k; ++j){
      uvec positions = arma::find(cluster_alloc_t == j);
      Y_j(j) = Y(positions);
    }
    for (int i = 0; i < n; i++){
      int c_i = cluster_alloc_t(i);
      likhood(i, t) = exp(log_post(Y_j(c_i), P0_params, Y(i))); 
    }
  p.increment();
  }
  for (int i = 0; i < n; ++i) {
    CPO_i(i) = 1.0 / arma::mean(1.0 / likhood.row(i));
  }
  return arma::sum(log(CPO_i)); 
}

// [[Rcpp::export]]
double LPML_reg(const mat &data, const vec &type, const imat &clusChain, vec &uChain, List &P0_params, List &ngg_params, List & g_params, bool display_progress = true){
  vec Y = data.col(0);
  mat X = data.cols(1, data.n_cols - 1);
  vec intrcp = arma::ones<vec>(X.n_rows); 
  mat X_intrcpt = join_horiz(intrcp, X); 
  int n = clusChain.n_rows;
  int m = clusChain.n_cols;
  vec CPO_i(n);
  mat likhood(n,m);
  ivec cluster_alloc_t(n);
  double sigma = ngg_params[0];
  double u = 0.0;

  Progress p(m, display_progress);
  for (int t = 0; t < m; t++){
    cluster_alloc_t = clusChain.col(t);
    u = uChain(t);
    ivec clus_unique = arma::unique(cluster_alloc_t);
    vec clus_count(clus_unique.n_elem); 
    for (size_t i=0 ; i<clus_count.n_elem ;i++){
      clus_count(i)=accu(cluster_alloc_t == clus_unique(i));
    }
    int k = clus_count.n_elem;
    vec dens(k);
    field<arma::mat> X_j(k);
    field<arma::mat> X_j_intrcpt(k);
    vec log_g_elem(k);
    vec sigma2(k);
    mat beta(k, X_intrcpt.n_cols);
    for (int j=0; j<k; ++j){
      uvec positions = arma::find(cluster_alloc_t == j);
      X_j(j) = X.rows(positions);
      X_j_intrcpt(j) = X_intrcpt.rows(positions);
      log_g_elem(j) = log_g(X_j(j), type, g_params);
      mat Y_j = Y.rows(positions); 
      List params = sample_params_reg(Y_j, X_j_intrcpt(j), P0_params);
      sigma2(j) = params[0];
      rowvec beta_t = params[1];
      beta.row(j) = beta_t;
    }
    for (int i = 0; i < n; i++){
      double accu_dens = 0.0;
      double ynew = Y(i);
      rowvec xnew = X.row(i);
      rowvec xnew_intrcpt = X_intrcpt.row(i);
      //compute vector of cluster allocation weights
      for (int j = 0; j < k; ++j) {
        mat x_x0 = arma::join_vert(X_j(j), xnew); 
        double log_w = log(clus_count(j)-sigma)-log(1+u)+log_g(x_x0, type,g_params)-log_g_elem(j);
        dens(j) = log_w + d_norm(ynew, as_scalar(beta.row(j)*xnew_intrcpt.t()), sqrt(sigma2(j)), true);
        accu_dens += exp(log_w);
      }
    likhood(i, t) = sum(exp(dens) / accu_dens); 
    }
  p.increment();
  }
  for (int i = 0; i < n; ++i) {
    CPO_i(i) = 1.0 / arma::mean(1.0 / likhood.row(i));
  }
  Rcpp::Rcout<<arma::sum(log(CPO_i))<<std::endl;
  return arma::sum(log(CPO_i)); 
}

// [[Rcpp::export]]
double LPML_reg_new(const mat &data, const vec &type, const imat &clusChain, List &P0_params){
  vec Y = data.col(0);
  mat X = data.cols(1, data.n_cols - 1);
  int n = clusChain.n_rows;
  int m = clusChain.n_cols;
  vec CPO_i(n);
  mat likhood(n,m);
  ivec cluster_alloc_t(n);

  for (int t = 0; t < m; t++){
    cluster_alloc_t = clusChain.col(t);
    ivec clus_unique = arma::unique(cluster_alloc_t);
    vec clus_count(clus_unique.n_elem); 
    for (size_t i=0 ; i<clus_count.n_elem ;i++){
      clus_count(i)=accu(cluster_alloc_t == clus_unique(i));
    }
    int k = clus_count.n_elem;
    field<arma::mat> X_j(k);
    field<arma::vec> Y_j(k);
    for (int j=0; j<k; ++j){
      uvec positions = arma::find(cluster_alloc_t == j);
      X_j(j) = X.rows(positions);
      Y_j(j) = Y(positions);
    }
    for (int i = 0; i < n; i++){
      int c_i = cluster_alloc_t(i);
      likhood(i, t) = exp(log_post_reg(Y_j(c_i), X_j(c_i), P0_params, Y(i), X.row(i))); 
    }
  }
  for (int i = 0; i < n; ++i) {
    CPO_i(i) = 1.0 / arma::mean(1.0 / likhood.row(i));
  }
  return arma::sum(log(CPO_i)); 
}

// [[Rcpp::export]]
double LPML_beta_bern(const mat &data, const vec &type, const imat &clusChain, vec &uChain, List &P0_params, List &ngg_params, List & g_params, bool display_progress = true){
  vec Y = data.col(0);
  mat X = data.cols(1, data.n_cols - 1);
  int n = clusChain.n_rows;
  int m = clusChain.n_cols;
  vec CPO_i(n);
  mat likhood(n,m);
  ivec cluster_alloc_t(n);
  double sigma = ngg_params[0];
  double u = 0.0;

  Progress p(m, display_progress);
  for (int t = 0; t < m; t++){
    cluster_alloc_t = clusChain.col(t);
    u = uChain(t);
    ivec clus_unique = arma::unique(cluster_alloc_t);
    vec clus_count(clus_unique.n_elem); 
    for (size_t i=0 ; i<clus_count.n_elem ;i++){
      clus_count(i)=accu(cluster_alloc_t == clus_unique(i));
    }
    int k = clus_count.n_elem;
    vec dens(k);
    field<arma::mat> X_j(k);
    vec log_g_elem(k);
    vec theta(k);
    for (int j=0; j<k; ++j){
      uvec positions = arma::find(cluster_alloc_t == j);
      X_j(j) = X.rows(positions);
      log_g_elem(j) = log_g(X_j(j), type, g_params);
      mat Y_j = Y.rows(positions);
      theta(j) = sample_theta(Y_j, P0_params);
    }
    for (int i = 0; i < n; i++){
      double accu_dens = 0.0;
      double ynew = Y(i);
      rowvec xnew = X.row(i);
      //compute vector of cluster allocation weights
      for (int j = 0; j < k; ++j) {
        mat x_x0 = arma::join_vert(X_j(j), xnew); 
        double log_w = log(clus_count(j)-sigma)-log(1+u)+log_g(x_x0, type,g_params)-log_g_elem(j);
        dens(j) = log_w + ynew*log(theta(j)) + (1-ynew)*log((1-theta(j)));
        accu_dens += exp(log_w);
      }
    likhood(i, t) = sum(exp(dens) / accu_dens); 
    }
  p.increment();
  }
  for (int i = 0; i < n; ++i) {
    CPO_i(i) = 1.0 / arma::mean(1.0 / likhood.row(i));
  }
  return arma::sum(log(CPO_i)); 
}

// [[Rcpp::export]]
double LPML_beta_bern_new(const mat &data, const vec &type, const imat &clusChain, List &P0_params){
  vec Y = data.col(0);
  int n = clusChain.n_rows;
  int m = clusChain.n_cols;
  vec CPO_i(n);
  mat likhood(n,m);
  ivec cluster_alloc_t(n);

  for (int t = 0; t < m; t++){
    cluster_alloc_t = clusChain.col(t);
    ivec clus_unique = arma::unique(cluster_alloc_t);
    vec clus_count(clus_unique.n_elem); 
    for (size_t i=0 ; i<clus_count.n_elem ;i++){
      clus_count(i)=accu(cluster_alloc_t == clus_unique(i));
    }
    int k = clus_count.n_elem;
    field<arma::vec> Y_j(k);
    for (int j=0; j<k; ++j){
      uvec positions = arma::find(cluster_alloc_t == j);
      Y_j(j) = Y(positions);
    }
    for (int i = 0; i < n; i++){
      int c_i = cluster_alloc_t(i);
      likhood(i, t) = exp(log_post_beta_bern(Y_j(c_i), P0_params, Y(i))); 
    }
  }
  for (int i = 0; i < n; ++i) {
    CPO_i(i) = 1.0 / arma::mean(1.0 / likhood.row(i));
  }
  return arma::sum(log(CPO_i)); 
}

// [[Rcpp::export]]
double LPML_beta_multibern(const mat &data, const vec &type, const imat &clusChain, vec &uChain, List &P0_params, List &ngg_params, List & g_params, bool display_progress = true){
  mat Y = data.cols(0,5);
  mat X = data.cols(6, data.n_cols - 1);
  int n = clusChain.n_rows;
  int m = clusChain.n_cols;
  vec CPO_i(n);
  mat likhood(n,m);
  ivec cluster_alloc_t(n);
  double sigma = ngg_params[0];
  double u = 0.0;

  Progress p(m, display_progress);
  for (int t = 0; t < m; t++){
    cluster_alloc_t = clusChain.col(t);
    u = uChain(t);
    ivec clus_unique = arma::unique(cluster_alloc_t);
    vec clus_count(clus_unique.n_elem); 
    for (size_t i=0 ; i<clus_count.n_elem ;i++){
      clus_count(i)=accu(cluster_alloc_t == clus_unique(i));
    }
    int k = clus_count.n_elem;
    vec dens(k);
    field<mat> X_j(k);
    vec log_g_elem(k);
    mat theta(k, Y.n_cols);
    for (int j=0; j<k; ++j){
      uvec positions = arma::find(cluster_alloc_t == j);
      X_j(j) = X.rows(positions);
      log_g_elem(j) = log_g(X_j(j), type, g_params);
      mat Y_j = Y.rows(positions);  
      rowvec theta_t = sample_theta_multibern(Y_j, P0_params);
      theta.row(j) = theta_t;
    }
    for (int i = 0; i < n; i++){
      double accu_dens = 0.0;
      rowvec ynew = Y.row(i);
      rowvec xnew = X.row(i);
      //compute vector of cluster allocation weights
      for (int j = 0; j < k; ++j) {
        mat x_x0 = arma::join_vert(X_j(j), xnew); 
        double log_w = log(clus_count(j)-sigma)-log(1+u)+log_g(x_x0, type,g_params)-log_g_elem(j);
        dens(j) = log_w + as_scalar(ynew*log(theta.row(j)).t()) + as_scalar((1-ynew)*log((1-theta.row(j))).t());
        accu_dens += exp(log_w);
      }
    likhood(i, t) = sum(exp(dens) / accu_dens); 
    }
  p.increment();
  }
  for (int i = 0; i < n; ++i) {
    CPO_i(i) = 1.0 / arma::mean(1.0 / likhood.row(i));
  }
  return arma::sum(log(CPO_i)); 
}

// [[Rcpp::export]]
double LPML_beta_multibern_new(const mat &data, const vec &type, const imat &clusChain, List &P0_params){
  mat Y = data.cols(0,5);
  int n = clusChain.n_rows;
  int m = clusChain.n_cols;
  vec CPO_i(n);
  mat likhood(n,m);
  ivec cluster_alloc_t(n);

  for (int t = 0; t < m; t++){
    cluster_alloc_t = clusChain.col(t);
    ivec clus_unique = arma::unique(cluster_alloc_t);
    vec clus_count(clus_unique.n_elem); 
    for (size_t i=0 ; i<clus_count.n_elem ;i++){
      clus_count(i)=accu(cluster_alloc_t == clus_unique(i));
    }
    int k = clus_count.n_elem;
    field<mat> Y_j(k);
    for (int j=0; j<k; ++j){
      uvec positions = arma::find(cluster_alloc_t == j);
      Y_j(j) = Y.rows(positions);
    }
    for (int i = 0; i < n; i++){
      int c_i = cluster_alloc_t(i);
      likhood(i, t) = exp(eval_log_post_beta_multibern(Y_j(c_i), P0_params, Y.row(i))); 
    }
  }
  for (int i = 0; i < n; ++i) {
    CPO_i(i) = 1.0 / arma::mean(1.0 / likhood.row(i));
  }
  return arma::sum(log(CPO_i)); 
}

// [[Rcpp::export]]
double LPML_multi(const mat &data, const vec &type, const imat &clusChain, vec &uChain, List &P0_params, List &ngg_params, List & g_params, bool display_progress = true){
  mat Y = data.cols(0, 1);
  mat X = data.cols(2, data.n_cols - 1);
  int n = clusChain.n_rows;
  int m = clusChain.n_cols;
  vec CPO_i(n);
  mat likhood(n,m);
  ivec cluster_alloc_t(n);
  double sigma = ngg_params[0];
  double u = 0.0;

  Progress p(m, display_progress);
  for (int t = 0; t < m; t++){
    cluster_alloc_t = clusChain.col(t);
    u = uChain(t);
    ivec clus_unique = arma::unique(cluster_alloc_t);
    vec clus_count(clus_unique.n_elem); 
    for (size_t i=0 ; i<clus_count.n_elem ;i++){
      clus_count(i)=accu(cluster_alloc_t == clus_unique(i));
    }
    int k = clus_count.n_elem;
    vec dens(k);
    arma::field<arma::mat> X_j(k);
    vec log_g_elem(k);
    cube S(2, 2, k);
    mat mu(k, 2);
    for (int j = 0; j < k; ++j){
      uvec positions = arma::find(cluster_alloc_t == j);
      X_j(j) = X.rows(positions);
      log_g_elem(j) = log_g(X_j(j), type, g_params);
      mat Y_j = Y.rows(positions);
      S.slice(j) = sample_S(Y_j, P0_params);
      mu.row(j) = sample_Mu(Y_j, S.slice(j), P0_params);
    }
    
    for (int i = 0; i < n; i++){
      double accu_dens = 0.0;
      rowvec ynew = Y.row(i);
      rowvec xnew = X.row(i);
      //compute vector of cluster allocation weights
      for (int j = 0; j < k; ++j) {
        mat x_x0 = arma::join_vert(X_j(j), xnew); 
        double log_w = log(clus_count(j)-sigma)-log(1+u)+log_g(x_x0, type,g_params)-log_g_elem(j);
        dens(j) = log_w + dmvnrm_arma_fast(ynew, mu.row(j), S.slice(j), true)(0);
        accu_dens += exp(log_w);
      }
    likhood(i, t) = sum(exp(dens) / accu_dens); 
    }
  p.increment();
  }
  for (int i = 0; i < n; ++i) {
    CPO_i(i) = 1.0 / arma::mean(1.0 / likhood.row(i));
  }

  
  return arma::sum(log(CPO_i)); 
}

// [[Rcpp::export]]
double LPML_multi_new(const mat &data, const vec &type, const imat &clusChain, List &P0_params){
  mat Y = data.cols(0, 1);
  int n = clusChain.n_rows;
  int m = clusChain.n_cols;
  vec CPO_i(n);
  mat likhood(n,m);
  ivec cluster_alloc_t(n);

  for (int t = 0; t < m; t++){
    cluster_alloc_t = clusChain.col(t);
    ivec clus_unique = arma::unique(cluster_alloc_t);
    vec clus_count(clus_unique.n_elem); 
    for (size_t i = 0 ; i < clus_count.n_elem ; i++){
      clus_count(i) = accu(cluster_alloc_t == clus_unique(i));
    }
    int k = clus_count.n_elem;
    field<mat> Y_j(k);
    for (int j = 0; j < k; ++j){
      uvec positions = arma::find(cluster_alloc_t == j);
      Y_j(j) = Y.rows(positions);
    }
    for (int i = 0; i < n; i++){
      int c_i = cluster_alloc_t(i);
      likhood(i, t) = exp(eval_log_post_multi(Y_j(c_i), P0_params, Y.row(i))); 
    }
  }
  for (int i = 0; i < n; ++i) {
    CPO_i(i) = 1.0 / arma::mean(1.0 / likhood.row(i));
  }
  return arma::sum(log(CPO_i)); 
}

// [[Rcpp::export]]
double LPML_pois_gamma(const mat &data, const vec &type, const imat &clusChain, List &P0_params, bool display_progress = true){
  vec Y = data.col(0);
  int n = clusChain.n_rows;
  int m = clusChain.n_cols;
  vec CPO_i(n);
  //mat mu(n,m);
  //mat sigma2(n,m);
  mat likhood(n,m);
  ivec cluster_alloc_t(n);
  //Progress p(m, display_progress);
  for (int t = 0; t < m; t++){
    cluster_alloc_t = clusChain.col(t);
    ivec clus_unique = arma::unique(cluster_alloc_t);
    int k = clus_unique.n_elem;
    for (int j = 0; j < k; ++j){
      uvec positions = arma::find(cluster_alloc_t == clus_unique(j));
      vec Y_j = Y.rows(positions);
      double lambda_t = sample_lambda(Y_j, P0_params);
      //mu.submat(positions, t*arma::ones<uvec>(positions.n_elem)).fill(mu_t);
      //sigma2.submat(positions, t*arma::ones<uvec>(positions.n_elem)).fill(sigma2_t);
      for (uword i = 0; i < positions.n_elem; i++){
        likhood(positions(i), t) = pow(lambda_t, Y(positions(i)))*exp(- lambda_t)/std::tgamma(Y(positions(i))+1);  //tgamma(n)=(n−1)!
      }
    }
  //p.increment();
  }
  for (uword i = 0; i < CPO_i.n_elem; ++i) {
    CPO_i(i) = 1.0 / arma::mean(1.0 / likhood.row(i));
  }

  return arma::sum(log(CPO_i));
}

// [[Rcpp::export]]
List compute_metrics(const mat &data_tot, const vec &true_lab_tot, const vec &type, const vec &grid_y1, const vec &grid_y2, List &g_params, List &ngg_params, List &P0_params, int niter=200, int nburn=100, int thin=1, int itermean = 100, int nGibbs=0, int thinGibbs = 100000, bool pred_dense=false, bool pred1=false, bool pred2=false, bool display_progress=true){
  Environment mcclust = Environment::namespace_env("mcclust");
  Function arandi = mcclust["arandi"];
  vec arand_chain(itermean);
  vec LPML_chain(itermean);
  vec RMSE_chain(itermean);
  vec n_clust_chain(itermean);
  vec accept_split_chain(itermean);
  vec accept_merge_chain(itermean);
  
  for (int i = 0; i < itermean; i++){
    mat data_temp = data_tot;
    uvec indices_train = arma::randperm(data_temp.n_rows, 1500);
    mat data_train = data_temp.rows(indices_train);
    vec true_lab_sample = true_lab_tot(indices_train);
    data_temp.shed_rows(indices_train);
    uvec indices_test = arma::randperm(data_temp.n_rows, 200);
    mat data_test = data_temp.rows(indices_test);
    vec y_test = data_test.col(0);
    mat X_test = data_test.cols(1, data_test.n_cols - 1);
    //mat y_test = data_test.cols(0,1);
    //mat X_test = data_test.cols(2, data_test.n_cols-1);
    //mat y_test = data_test.cols(0,5);
    //mat X_test = data_test.cols(6, data_test.n_cols - 1);

    List r1 = run_mcmc_pred(data_train, type, g_params, ngg_params, P0_params, y_test.t(), X_test, grid_y1, niter, nburn, thin, nGibbs, thinGibbs, pred_dense, pred1, pred2, display_progress);
    //List r1 = run_mcmc_pred_multi(data_train, type, g_params, ngg_params, P0_params, y_test, X_test, grid_y1, grid_y2, niter, nburn, thin, nGibbs, thinGibbs, pred_dense, pred1, pred2, display_progress);
    ivec best_binder = r1["best_clus_binder"];
    imat clus_chain = r1["clus_chain"];
    vec u_chain = r1["u_chain"];
    arand_chain(i) = as<double>(arandi(true_lab_sample, best_binder));
    LPML_chain(i) = LPML_new(data_train, type, clus_chain, P0_params);
    RMSE_chain(i) = r1["rmse2"];
    n_clust_chain(i) = r1["n_clust_mean"];
    if (thinGibbs != 1){
    accept_split_chain(i) = r1["accept_split"];
    accept_merge_chain(i) = r1["accept_merge"];
    }
  }
  List result;
  double mean_arand = mean(arand_chain);
  double mean_LPML = mean(LPML_chain);
  double mean_RMSE = mean(RMSE_chain);
  double mean_n_clust = mean(n_clust_chain);
  result["arandi"] = mean_arand;
  result["lpml"] = mean_LPML;
  result["rmse"] = mean_RMSE;
  result["n_clust"] = mean_n_clust;
  if (thinGibbs != 1){
    double mean_accept_split = mean(accept_split_chain);
    double mean_accept_merge = mean(accept_merge_chain);
    result["accept_split"] = mean_accept_split;
    result["accept_merge"] = mean_accept_merge;
  }
  return result;
}
