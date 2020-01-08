import numpy as np
import pandas as pd
import time
import pathos.multiprocessing as mp

from SMC_ABC_init import div_path, L, p_0, MCSteps, N_A, \
    delta_min, mu_min, alpha_min, lambda0_min, C_lambda_min, deltaS_min, \
    delta_max, mu_max, alpha_max, lambda0_max, C_lambda_max, deltaS_max # , \
    #delta_true, mu_true, alpha_true, lambda0_true, C_lambda_true, delta_S_true, \

#N_A = 250 # override
#L = 2300 # override

def priorSimulation(numberSim):
    # generate numberSim simultations of each parameter
    delta = np.random.uniform(delta_min, delta_max, numberSim)
    mu = np.random.uniform(mu_min, mu_max, numberSim)
    alpha = np.random.uniform(alpha_min, alpha_max, numberSim)
    lambda0 = np.random.uniform(lambda0_min, lambda0_max, numberSim)
    C_lambda = np.random.uniform(C_lambda_min, C_lambda_max, numberSim)
    delta_S = np.random.uniform(deltaS_min, deltaS_max, numberSim)

    # Store and return simultations
    priorDf = np.column_stack((delta, mu, alpha, lambda0, C_lambda, delta_S))
    return pd.DataFrame(priorDf, columns=("delta", "mu", "alpha", "lambda_0", "C_lambda", "delta_S"))


class ParallelModel:
    @staticmethod
    def preisSim(param1, param2, param3, param4, param5, param6,
                 N_A, p_0, L, MCSteps):
        # initialize preis model object with specified parameters
        from preisSeed import PreisModel
        import pandas as pd

        p = PreisModel(N_A=N_A,
                       delta=param1,
                       mu=param2,
                       alpha=param3,
                       lambda_0=param4,
                       C_lambda=param5,
                       delta_S=param6,
                       p_0=p_0,
                       T=L,
                       MC=MCSteps)

        # Start model
        p.simRun()
        p.initialize()

        # Simulate price path for T=L time-steps
        p.simulate()

        return pd.DataFrame(p.intradayPrice/ div_path)

    def f(self, x):
        return self.preisSim(*x)


if __name__ == '__main__':
    np.random.seed(19950412)

    # 1. s() = I()
    # 2. Compute s(y_obs) = y_obs
    L = L  # TimeHorizon
    N = 500    # number of simulated y_obs
    p_0 = p_0
    MCSteps = MCSteps
    N_A = N_A

    param = priorSimulation(N)
    # randomise seed
    seed = int(np.random.uniform(0, 1000000))
    np.random.seed(seed)

    print(time.asctime())
    para_model = ParallelModel()
    with mp.Pool((mp.cpu_count())) as pool:
        results_list = pool.map(
            para_model.f, [(param.iloc[i, 0], param.iloc[i, 1], param.iloc[i, 2],
                            param.iloc[i, 3], param.iloc[i, 4], param.iloc[i, 5], N_A, p_0, L, MCSteps) for i in range(N)]
        )

    results_df = pd.concat(results_list, axis=1)
    results_df = results_df.div(1000)

    param = pd.DataFrame(param)
    param.to_csv('param3.csv', index=False)
    results_df.to_csv('out3.csv', index=False)
    print(time.asctime())
