import pandas as pd
import numpy as np
from typing import Tuple
from pydantic import BaseModel, validator

class ExperimentImpact(BaseModel):
    perc_prob_success_change: float
    width_perc_prob_change: float=None
    treatment_proportion: float=0.5

    @validator('width_perc_prob_change', pre=True, always=True)
    def set_width_perc_change(cls, v, values):
        return v or values['perc_prob_success_change']*0.1
    



class ratio_data_dgp():
    '''Class based on data generating process from Deng et al (2018), 
    Applying the Delta Method in Metric Analytics: 
    A Practical Guide with Novel Ideas (§3.3)

    K is the number of experimental units (clusters; e.g., no. of drivers). 
    The denominator is a Poisson process (e.g., no. of offers), numerator is
    a series of bernoulli trials with a normally distributed probability of
    success (clipped at 0 and 1).

    Args:
        K: number of clusters
        p: probability of being in a group with similar behaviour
        poisson_lams: denominator is a Poisson process (one lambda for each group)
        mean_prob_success: mean probability of success
        std_prob_success: std of probability of success

    Returns:
        pd.DataFrame:
            index: cluster
            Y: numerator
            N: denominator
    '''
    def __init__(
            self,
            K: int=1000,
            p: np.array=np.array([1/3, 1/2, 1/5]),
            poisson_lams: np.array=np.array([2, 5, 30]),
            poisson_multi_pre: float=1,
            poisson_multi_exp: float=1,
            mean_prob_success:np.array = np.array([0.3, 0.5, 0.8]),
            std_prob_success:np.array = np.array([0.05, 0.1, 0.05]),
            seed:int = 42
    ):
        if any([len(p)!=len(i) for i in (poisson_lams, mean_prob_success, std_prob_success)]):
            raise ValueError('The lengths of inputs are not equal.')
        
        self.K = K
        self.p = p
        self.poisson_lams = poisson_lams
        self.poisson_multi_pre = poisson_multi_pre
        self.poisson_multi_exp = poisson_multi_exp
        self.mean_prob_success = mean_prob_success
        self.std_prob_success = std_prob_success
        self.seed = seed
        self.rng = np.random.default_rng(seed=self.seed)

        '''Define characteristics of drivers'''
        # groups of clusters with similar behaviour
        self.M = self.rng.multinomial(self.K, self.p)
        self.driver_types = np.concatenate([np.ones(j)*i for i,j in enumerate(self.M)]).astype(int)

        # prob success
        prob_success = np.concatenate(
            [
                self.rng.normal(j[0], j[1], self.M[i])
                for i, j in enumerate(zip(self.mean_prob_success, self.std_prob_success))
            ]
        )
        self.prob_success = np.clip(prob_success, a_min=0, a_max=1)
    

    def experiment_changes(self, experiment_impact: ExperimentImpact):
        change_in_prob_success = self.rng.uniform(
            low=experiment_impact.perc_prob_success_change-experiment_impact.width_perc_prob_change/2,
            high=experiment_impact.perc_prob_success_change+experiment_impact.width_perc_prob_change/2,
            size=self.K
        )
        experiment_assignment = self.rng.binomial(1, 0.5, self.K)
        prob_success_new = self.prob_success * (change_in_prob_success*experiment_assignment + 100)/100
        prob_success_new = np.clip(prob_success_new, a_min=0, a_max=1)
        return prob_success_new, experiment_assignment

    def dgp(
            self,
            experiment_impact: ExperimentImpact=None
        ) -> pd.DataFrame:
        '''Data generating process
        '''
        # denominator
        N = np.concatenate(
            [
                self.rng.poisson(self.poisson_lams[i]*self.poisson_multi_pre, self.M[i]) 
                for i in range(len(self.p))
            ]
        )

        # numerator
        prob_success = self.prob_success
        Y = self.rng.binomial(N, prob_success)

        # data with experiment
        if experiment_impact:
            # denominator
            N_exp = np.concatenate(
                [
                    self.rng.poisson(self.poisson_lams[i]*self.poisson_multi_exp, self.M[i]) 
                    for i in range(len(self.p))
                ]
            )

            # numerator
            self.prob_success_new, self.experiment_assignment = self.experiment_changes(experiment_impact)
            Y_exp = self.rng.binomial(N_exp, self.prob_success_new)

            return pd.DataFrame({'Y': Y, 'N': N, 
                                 'Y_exp': Y_exp, 'N_exp': N_exp, 
                                 'treatment': self.experiment_assignment,
                                 'type': self.driver_types})

        return pd.DataFrame({'Y': Y, 'N': N, 'type': self.driver_types})