# Import libraries
import numpy as np
import scipy as sp
import pandas as pd
import os
from hurst import compute_Hc
from scipy import stats
from tempfile import gettempdir
from pyabc import MedianEpsilon, \
    LocalTransition, Distribution, RV, ABCSMC, sge, \
    AdaptivePNormDistance, PNormDistance, UniformAcceptor
from pyabc.sampler import MulticoreEvalParallelSampler
import time

# Set version number each iteration
version_number = "PNormDistance" + str(time.time())

# SMCABC parameters:
SMCABC_distance = PNormDistance(p=2)
SMCABC_population_size = 30
SMCABC_sampler = MulticoreEvalParallelSampler(40)
SMCABC_transitions = LocalTransition(k_fraction=.3)
SMCABC_eps = MedianEpsilon(500, median_multiplier=0.7)
smcabc_minimum_epsilon = 0.001
smcabc_max_nr_populations = 5
smcabc_min_acceptance_rate = SMCABC_population_size/50000

# Fixed Parameters
div_path = 1000
L = 250                 # time horizon
p_0 = 100 * div_path    # initial price
MCSteps = 10 ** 5       # MC steps to generate variance
N_A = 125               # no. market makers = no. liquidity providers

# True price trajectory
delta_true = 0.0250       # limit order cancellation rate
mu_true = 0.0250          # rate of market orders
alpha_true = 0.15       # rate of limit orders
lambda0_true = 100      # initial order placement depth
C_lambda_true = 10      # limit order placement depth coefficient
delta_S_true = 0.0010      # mean reversion strength parameter

# prior range
delta_min, delta_max = 0, 0.05
mu_min, mu_max = 0, 0.05
alpha_min, alpha_max = 0.05, 0.5
lambda0_min, lambda0_max = 0, 200
C_lambda_min, C_lambda_max = 0, 20
deltaS_min, deltaS_max = 0, 0.02

def summary_stats_extra(x):
    """outputs additional summary statistics: skewness, kurtosis, Hurst"""

    try:
        H, c, data = compute_Hc(x, kind='price', simplified=True)
    except Exception as e:
        H = 0.25
        print(e)

    return {"skew": x.skew(),
            "kurt": x.kurt(),
            "hurst": H
            }


def all_summary_stats(price_sim, price_obs):
    """ouptuts all summary statistics of price path compared to true price path (path_obs)"""

    # count, mean, std, min, 25%, 50%, 75%, max
    s1 = price_sim[0].describe()

    # skew, kurt, hurst
    s2 = summary_stats_extra(price_sim[0])

    # Kolmogorov Smirnov 2 sample test statistic (if 0 - identical)
    ks_stat = {"KS": stats.ks_2samp(np.ravel(price_sim), np.ravel(price_obs))[0]}

    return {"mean": s1.loc["mean"],
            "std": s1.loc["std"],
            **s2,
            **ks_stat}


def accept_pos(x):
    """Outputs true if entire vector is positive"""

    if min(x)>0:
        return True
    else:
        return False


def preisSim(parameters):
    """Outputs: summary statistics from Preis model,
     Inputs: dictionary with delta, mu, alpha, lambda0, C_lambda, delta
     Static parameters: L, p_0, MCSteps, N_A"""

    # Import libraries to be used in model simulation
    from preisSeed import PreisModel
    import pandas as pd

    # Fixed Parameters
    div_path = 1000
    L = 250  # time horizon
    p_0 = 100 * div_path  # initial price
    MCSteps = 10 ** 5  # MC steps to generate variance
    N_A = 125  # no. market makers = no. liquidity providers

    # continue until price path simulated is all positive
    positive_price_path = False
    while not positive_price_path:

        # Initialize preis model class with specified parameters
        p = PreisModel(N_A=N_A,
                       delta=parameters["delta"],
                       mu=parameters["mu"],
                       alpha=parameters["alpha"],
                       lambda_0=parameters["lambda0"],
                       C_lambda=parameters["C_lambda"],
                       delta_S=parameters["delta_S"],
                       p_0=p_0,
                       T=L,
                       MC=MCSteps)

        # Start model
        p.simRun()
        p.initialize()

        # Simulate price path for L time-steps
        p.simulate()

        # ensure no negative prices
        positive_price_path = accept_pos(p.intradayPrice)

    # Log and divide price path by 1000, Convert to pandas dataframe
    price_path = pd.DataFrame(np.log(p.intradayPrice / div_path))

    return price_path

def sum_stat_sim(parameters):

    price_path = preisSim(parameters)

    p_true = pd.read_csv(os.path.join("/home/gsnkel001/master_dissertation/StoreDB", version_number + "p_true.csv"))

    # summary statistics
    return all_summary_stats(price_path, p_true)


# Parameters as Random Variables
prior = Distribution(delta=RV("uniform", delta_min, delta_max),
                     mu=RV("uniform", mu_min, mu_max),
                     alpha=RV("uniform", alpha_min, alpha_max),
                     lambda0=RV("uniform", lambda0_min, lambda0_max),
                     C_lambda=RV("uniform", C_lambda_min, C_lambda_max),
                     delta_S=RV("uniform", deltaS_min, deltaS_max))

# define "true" parameters to calibrate
param_true = {"delta": delta_true,
              "mu": mu_true,
              "alpha": alpha_true,
              "lambda0": lambda0_true,
              "C_lambda": C_lambda_true,
              "delta_S": delta_S_true}


# define distance function
def distance(simulation, data):

    dist = sp.absolute(data["mean"] - simulation["mean"]) + \
           sp.absolute(data["std"] - simulation["std"]) + \
           sp.absolute(data["skew"] - simulation["skew"]) + \
           sp.absolute(data["kurt"] - simulation["kurt"]) + \
           sp.absolute(data["hurst"] - simulation["hurst"]) + \
           sp.absolute(simulation["KS"])

    return dist


if __name__ == '__main__':

    # Simulate "true" summary statistics
    p_true = preisSim(param_true)
    p_true.to_csv(os.path.join("/home/gsnkel001/master_dissertation/StoreDB", version_number + 'p_true.csv'), index=False)
    p_true_SS = all_summary_stats(p_true, p_true)





    # Initialise ABCSMC model parameters
    abc = ABCSMC(models=sum_stat_sim,
                 parameter_priors=prior,
                 distance_function=SMCABC_distance,
                 population_size=SMCABC_population_size,
                 sampler=SMCABC_sampler,
                 transitions=SMCABC_transitions,
                 eps=SMCABC_eps)#,
                 #acceptor=UniformAcceptor(use_complete_history=True))

    # Set up SQL storage facility
    db = "sqlite:///" + os.path.join("/home/gsnkel001/master_dissertation/StoreDB", "test" + version_number + ".db")

    # Input SMCABC SQL and observed summary stats
    abc.new(db, p_true_SS)
