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

import ast

import numpy as np
import ultraopt as uo

from optimizers._optimizer_base import OptimizerBase
from ultraopt.hdl import hdl2cs


class UltraOpt(OptimizerBase):
    def __init__(self, fn, struc_param_init: dict, algo_param: dict, **kwargs):
        super().__init__(fn, struc_param_init, algo_param, **kwargs)

        self.input_struc_param = self.struc_param_to_input(struc_param_init, **kwargs)

        if self.algorithm == 'etpe':
            print('using UltraOpt --> ETPE ...')
            self.algo = 'ETPE'
        # elif self.algorithm == 'rand':
        #     print('using HyperOpt --> Random Search ...')
        else:
            raise Exception('The algorithm %s not in HyperOpt, please check!' % self.algorithm)

        if self.rand_seed == -1:
            self.rand_seed = None

        self.init_points = [config.get_dictionary() for config in hdl2cs(self.input_struc_param).sample_configuration(self.n_init)]

        use_bohb = kwargs.get("use_bohb", False)
        if use_bohb:
            from ultraopt.multi_fidelity import HyperBandIterGenerator
            self.hb = HyperBandIterGenerator(min_budget=1 / 4, max_budget=1, eta=2)

            # from ultraopt.multi_fidelity import SuccessiveHalvingIterGenerator
            # self.hb = SuccessiveHalvingIterGenerator(min_budget=1 / 4, max_budget=1, eta=2)
        else:
            self.hb = None

    def struc_param_to_input(self, struc_param_init: dict, **kwargs):
        return struc_param_init

    def struc_param_to_fn(self, struc_param_opt: dict, **kwargs) -> dict:
        _struc_param_opt = {}
        for k, v in struc_param_opt.items():
            if isinstance(v, str) and ":" in v:
                # vv = v.split(':')[0]
                # if 'int' in v:
                #     v = int(vv)
                # elif 'float' in v:
                #     v = float(vv)
                v = v.split(':')[0]
                v = ast.literal_eval(v)
            _struc_param_opt[k] = v
        return _struc_param_opt

    def fn_out(self, result, **kwargs):
        return result

    def run(self):
        opt_result = uo.fmin(
            self.target_func,
            config_space=self.input_struc_param,
            optimizer=self.algo,
            initial_points=self.init_points,
            multi_fidelity_iter_generator=self.hb,
            n_iterations=self.max_step,
            random_state=self.rand_seed,
            n_jobs=self.n_mpi,
        )
        print(opt_result)
