# -*- coding: utf-8 -*-
# ========================================================================
#  2022/5/24 8:25
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

import numpy as np
import hyperopt as hy

from optimizers._optimizer_base import OptimizerBase


class HyperOpt(OptimizerBase):
    def __init__(self, fn, struc_param_init: dict, algo_param: dict, **kwargs):
        super().__init__(fn, struc_param_init, algo_param, **kwargs)

        self.input_struc_param = self.struc_param_to_input(struc_param_init, **kwargs)

        if self.algorithm == 'rand':
            print('using HyperOpt --> Random Search ...')
            self.algo = hy.rand.suggest
        elif self.algorithm == 'anneal':
            print('using HyperOpt --> Simulated Annealing ...')
            self.algo = hy.partial(hy.anneal.suggest)
        elif self.algorithm == 'tpe':
            print('using HyperOpt --> Bayesian Optimization ...')
            self.algo = hy.partial(hy.tpe.suggest, n_startup_jobs=self.n_init)
        else:
            raise Exception('The algorithm %s not in HyperOpt, please check!' % self.algorithm)

        if self.rand_seed == -1:
            self.rand_seed = None
        else:
            self.rand_seed = np.random.default_rng(self.rand_seed)

    def struc_param_to_input(self, struc_param_init: dict, **kwargs):
        struc_param = {}
        for k, v in struc_param_init.items():
            struc_param[k] = self.hy_parameter_setting(k, v)

        return struc_param

    @staticmethod
    def hy_parameter_setting(label, config: dict):
        param_type = config['_type']
        param_value = config['_value']

        if param_type == 'int_uniform':
            parameter = hy.hp.uniformint(label, param_value[0], param_value[1])
        elif param_type == 'uniform':
            parameter = hy.hp.uniform(label, param_value[0], param_value[1])
        elif param_type == 'choice':
            parameter = hy.hp.choice(label, param_value)
        else:
            parameter = hy.hp.uniform(label, param_value[0], param_value[1])

        return parameter

    def struc_param_to_fn(self, struc_param_opt, **kwargs):
        return struc_param_opt

    def fn_out(self, result, **kwargs):
        return {'loss': result, 'status': hy.STATUS_OK}

    def run(self):
        trials = hy.Trials()
        # print(pbounds, rand_seed, max_step)
        best = hy.fmin(fn=self.target_func,
                       space=self.input_struc_param,
                       algo=self.algo,
                       max_evals=self.max_step,
                       trials=trials,
                       rstate=self.rand_seed  # 随机种子
                       )
        print(best)
