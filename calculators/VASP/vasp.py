# -*- coding: utf-8 -*-
# ========================================================================
#  2022/6/1 9:13
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
from pymatgen.core import Structure

from calculators._calculator_base import CalculatorBase


class VASP(CalculatorBase):
    def __init__(self, sub_file_path, pred_result_path='.', is_save_result=False, use_relax=False, **kwargs):
        if use_relax:
            from calculators.VASP.vasp_relax import VASPRelax
            self.vasp_method = VASPRelax(sub_file_path, pred_result_path=pred_result_path, is_save_result=is_save_result, **kwargs)
        else:
            from calculators.VASP.vasp_static import VASPStatic
            self.vasp_method = VASPStatic(sub_file_path, pred_result_path=pred_result_path, is_save_result=is_save_result, **kwargs)

    def get_target(self, struc: Structure, **kwargs):
        self.vasp_method.get_target(struc, **kwargs)
