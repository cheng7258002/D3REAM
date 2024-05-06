# -*- coding: utf-8 -*-
# ========================================================================
#  2022/6/6 13:24
#                 _____   _   _   _____   __   _   _____  
#                /  ___| | | | | | ____| |  \ | | /  ___| 
#                | |     | |_| | | |__   |   \| | | |     
#                | |     |  _  | |  __|  | |\   | | |  _  
#                | |___  | | | | | |___  | | \  | | |_| | 
#                \_____| |_| |_| |_____| |_|  \_| \_____/ 
# ------------------------------------------------------------------------
# 
# 
# 
# ========================================================================


import optuna
import numpy as np

from optimizers._optimizer_base import OptimizerBase


class Optuna(OptimizerBase):
    def __init__(self, fn, struc_param_init: dict, algo_param: dict, **kwargs):
        super().__init__(fn, struc_param_init, algo_param, **kwargs)
        target_num = kwargs.get('target_num', 1)

        if self.rand_seed == -1:
            self.rand_seed = None

        self.is_run_mpi = kwargs.get('is_run_mpi', False)

        if self.algorithm == 'tpe2':
            print('using Optuna --> Bayesian Optimization ...')

            def my_tpe_gamma(x: int) -> int:
                return min(int(np.ceil(0.25 * np.sqrt(x))), 25)

            if self.is_run_mpi:
                study_name = kwargs.get('study_name', 'test')
                storage = algo_param['storage']
                self.algo = optuna.create_study(
                    directions=["minimize"] * target_num,
                    study_name=study_name,
                    storage=storage,
                    sampler=optuna.samplers.TPESampler(gamma=my_tpe_gamma,
                                                       n_startup_trials=self.n_init, ),
                    load_if_exists=True,
                )
            else:
                self.algo = optuna.create_study(
                    # direction="minimize",
                    directions=["minimize"] * target_num,
                    sampler=optuna.samplers.TPESampler(gamma=my_tpe_gamma,
                                                       n_startup_trials=self.n_init,
                                                       seed=self.rand_seed, ),
                )
        elif self.algorithm == 'rand2':
            print('using Optuna --> Random Search ...')

            def my_tpe_gamma(x: int) -> int:
                return min(int(np.ceil(0.25 * np.sqrt(x))), 25)

            if self.is_run_mpi:
                study_name = kwargs.get('study_name', 'test')
                storage = algo_param['storage']
                self.algo = optuna.create_study(
                    directions=["minimize"] * target_num,
                    study_name=study_name,
                    storage=storage,
                    sampler=optuna.samplers.RandomSampler(),
                    load_if_exists=True,
                )
            else:
                self.algo = optuna.create_study(
                    # direction="minimize",
                    directions=["minimize"] * target_num,
                    sampler=optuna.samplers.RandomSampler(seed=self.rand_seed, ),
                )
        else:
            raise Exception('The algorithm %s not in HyperOpt, please check!' % self.algorithm)

    def struc_param_to_input(self, struc_param_init: dict, **kwargs):
        trial = kwargs.get('trial')
        struc_param = {}
        for k, v in struc_param_init.items():
            struc_param[k] = self.hy_parameter_setting(k, v, trial)

        return struc_param

    def struc_param_to_fn(self, struc_param_opt, **kwargs) -> dict:
        pass

    @staticmethod
    def hy_parameter_setting(label, config: dict, trial):
        param_type = config['_type']
        param_value = config['_value']

        if param_type == 'int_uniform':
            parameter = trial.suggest_int(label, param_value[0], param_value[1])
        elif param_type == 'uniform':
            parameter = trial.suggest_float(label, param_value[0], param_value[1])
        elif param_type == 'choice':
            parameter = trial.suggest_categorical(label, param_value)
        else:
            parameter = trial.suggest_float(label, param_value[0], param_value[1])

        return parameter

    def fn_out(self, result, **kwargs):
        return result

    def target_func(self, trial, **kwargs):
        struc_param = self.struc_param_to_input(self.struc_param_init, trial=trial)
        result = self.fn(struc_param, trial=trial)
        return self.fn_out(result, **kwargs)

    def run(self):
        self.algo.optimize(func=self.target_func,
                           n_trials=self.max_step,
                           n_jobs=1,
                           show_progress_bar=True, )
        print('best:', self.algo.best_trial.params)
