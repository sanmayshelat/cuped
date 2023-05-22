import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, t, norm
from typing import Tuple


class RatioATE:
    def __init__(
        self, control_group: pd.DataFrame, 
        treatment_group: pd.DataFrame, 
        numer: str, denom: str, 
        suffix_expt: str='', suffix_pre: str=None
        ):
        self.n_c = len(control_group)
        self.n_t = len(treatment_group)
        self.n_total = self.n_c + self.n_t
        
        numer_expt = numer + suffix_expt
        denom_expt = denom + suffix_expt

        self.Y_c = control_group[numer_expt]
        self.N_c = control_group[denom_expt]
        self.Y_t = treatment_group[numer_expt]
        self.N_t = treatment_group[denom_expt]

        self.Y_bar_c = self.Y_c.sum()/self.n_c
        self.N_bar_c = self.N_c.sum()/self.n_c
        self.Y_bar_t = self.Y_t.sum()/self.n_t
        self.N_bar_t = self.N_t.sum()/self.n_t

        if suffix_pre is not None:
            numer_pre = numer + suffix_pre
            denom_pre = denom + suffix_pre
            
            self.X_c = control_group[numer_pre]
            self.M_c = control_group[denom_pre]
            self.X_t = treatment_group[numer_pre]
            self.M_t = treatment_group[denom_pre]

            self.X_bar_c = self.X_c.sum()/self.n_c
            self.M_bar_c = self.M_c.sum()/self.n_c
            self.X_bar_t = self.X_t.sum()/self.n_t
            self.M_bar_t = self.M_t.sum()/self.n_t


    def calculate_theta(self):
        # Calculate theta by pooling together data from both cells
        Y = pd.concat([self.Y_c, self.Y_t])
        N = pd.concat([self.N_c, self.N_t])
        X = pd.concat([self.X_c, self.X_t])
        M = pd.concat([self.M_c, self.M_t])
        n_total = self.n_c + self.n_t
        Y_bar = Y.sum()/n_total
        X_bar = X.sum()/n_total
        N_bar = N.sum()/n_total
        M_bar = M.sum()/n_total
        
        # gradient calculation
        beta_expt = np.array([[1/N_bar, -Y_bar/(N_bar)**2, 0, 0]]).T
        beta_pre = np.array([[0, 0, 1/M_bar, -X_bar/(M_bar)**2]]).T

        # estimate covariance matrix from iid samples
        covar = np.cov(np.array([Y, N, X, M]), ddof=1)

        # theta = covariance(Y/N, X/M)/ var(X/M)
        self.theta = ((beta_expt.T @ covar @ beta_pre)/(beta_pre.T @ covar @ beta_pre))[0][0]

    
        # variance reduction can be durectly calculated by:
        # var(Y/N:cuped) = var(Y/N)(1-rho_sq)
        # where rho_sq = (cov(Y/N, X/M)/sqrt(var(Y/N)*var(X/N)))^2
        self.rho_sq = (
            (beta_expt.T @ covar @ beta_pre)
            @ np.linalg.inv(
                np.sqrt(
                    (beta_pre.T @ covar @ beta_pre) 
                    @ (beta_expt.T @ covar @ beta_expt)
                )
            )
        )[0][0]**2

    def pooled_zscore(self, ate: float, var: np.array=None, n: np.array=None, var_pooled:float=None) -> Tuple[float, float, float]:
        if not var_pooled:
            var_pooled = sum((n[i]-1)*var[i] for i in range(len(n))) / (sum(n)-len(n))
        zstat = ate/np.sqrt(sum(1/n[i] for i in range(len(n))) * var_pooled)
        pval = norm.sf(np.abs(zstat))*2
        return pval, zstat, var_pooled

    def ratio_ate(self) -> Tuple[float, float, float]:
        self.treatment_metric = self.Y_bar_t/self.N_bar_t
        self.control_metric = self.Y_bar_c/self.N_bar_c
        self.ate = self.treatment_metric - self.control_metric # the difference we are interested in

        # applying the delta method in matrix form
        beta_c = np.array([[1/self.N_bar_c, -self.Y_bar_c/(self.N_bar_c)**2]]).T
        covar_c = np.cov(self.Y_c, self.N_c, ddof=1)
        var_c = ((beta_c.T @ covar_c @ beta_c))[0][0]

        beta_t = np.array([[1/self.N_bar_t, -self.Y_bar_t/(self.N_bar_t)**2]]).T
        covar_t = np.cov(self.Y_t, self.N_t, ddof=1)
        var_t = ((beta_t.T @ covar_t @ beta_t))[0][0]

        self.pval, zstat, self.var_pooled = self.pooled_zscore(self.ate, [var_c, var_t], [self.n_c, self.n_t])

        return self.ate, self.pval, self.var_pooled

    def ratio_ate_cuped(self):
        self.calculate_theta()

        self.treatment_metric_cuped = (self.Y_bar_t/self.N_bar_t - self.theta*(self.X_bar_t/self.M_bar_t))
        self.control_metric_cuped = (self.Y_bar_c/self.N_bar_c - self.theta*(self.X_bar_c/self.M_bar_c))
        self.ate_cuped = self.treatment_metric_cuped - self.control_metric_cuped

        self.var_pooled_cuped = self.var_pooled * (1 - self.rho_sq)

        self.pval_cuped, zstat, self.var_pooled_cuped = self.pooled_zscore(ate=self.ate_cuped, n=[self.n_c, self.n_t], var_pooled=self.var_pooled_cuped)
        return self.ate_cuped, self.pval_cuped, self.var_pooled_cuped

