# -*- coding: utf-8 -*-
# ========================================================================
#  2022/5/23 18:45
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

from abc import ABCMeta, abstractmethod


class OptimizerBase(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, fn, struc_param_init: dict, algo_param: dict, **kwargs):
        self.fn = fn
        self.struc_param_init = struc_param_init

        # {'algo': algorithm, 'n_init': n_init, 'max_step': max_step, 'rand_seed': rand_seed}
        self.algorithm = algo_param['algo']
        self.n_init = algo_param['n_init']
        self.max_step = algo_param['max_step']
        self.rand_seed = algo_param['rand_seed']
        self.n_mpi = algo_param['n_mpi']

    @abstractmethod
    def struc_param_to_input(self, struc_param_init: dict, **kwargs):
        pass

    @abstractmethod
    def struc_param_to_fn(self, struc_param_opt, **kwargs) -> dict:
        pass

    @abstractmethod
    def fn_out(self, result, **kwargs):
        pass

    def target_func(self, struc_param_opt, **kwargs):
        struc_param = self.struc_param_to_fn(struc_param_opt, **kwargs)
        result = self.fn(struc_param)
        return self.fn_out(result, **kwargs)

    @abstractmethod
    def run(self):
        pass
