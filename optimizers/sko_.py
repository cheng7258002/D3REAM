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
import sko

from optimizers._optimizer_base import OptimizerBase


class ScikitOpt(OptimizerBase):
    def __init__(self, fn, struc_param_init: dict, algo_param: dict, **kwargs):
        super().__init__(fn, struc_param_init, algo_param, **kwargs)

        if self.rand_seed != -1:
            np.random.seed(self.rand_seed)

        lbs, ubs = self.struc_param_to_input(struc_param_init)
        if self.algorithm == 'pso':
            print('using ScikitOpt --> PSO ...')
            self.algo = sko.PSO.PSO(
                func=self.target_func,
                n_dim=len(lbs), lb=lbs, ub=ubs,
                pop=self.n_init, max_iter=self.max_step, w=0.8, c1=0.5, c2=0.5,
                verbose=True
            )
        # elif self.algorithm == 'ga':
        #     pass
        else:
            raise Exception('The algorithm %s not in HyperOpt, please check!' % self.algorithm)

    def struc_param_to_input(self, struc_param_init: dict, **kwargs):
        lbs, ubs = [], []
        for k, v in struc_param_init.items():
            lb, ub = self.sko_parameter_setting(k, v)
            lbs.append(lb)
            ubs.append(ub)

        return lbs, ubs

    @staticmethod
    def sko_parameter_setting(label, config: dict):
        param_type = config['_type']
        param_value = config['_value']

        if param_type == 'choice':
            lb, ub = 0, len(param_value)-0.00001
        elif param_type == 'int_uniform':
            lb, ub = param_value[0], param_value[1] + 1
        else:
            lb, ub = param_value[0], param_value[1]

        return lb, ub

    def struc_param_to_fn(self, *args) -> dict:
        args = args[0]

        _dict = {}
        for i, (k, v) in enumerate(self.struc_param_init.items()):
            param_type = v['_type']
            param_value = v['_value']

            if param_type == 'int_uniform':
                _dict[k] = int(args[i])
            elif param_type == 'choice':
                _dict[k] = param_value[int(args[i])]
            else:
                _dict[k] = args[i]

        return _dict

    def fn_out(self, result, **kwargs):
        return result

    def run(self):
        my_sko = self.algo.run()
        print('best_x is ', my_sko.gbest_x, 'best_y is', my_sko.gbest_y)
