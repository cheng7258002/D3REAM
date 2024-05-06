# -*- coding: utf-8 -*-
# ========================================================================
#  2022/8/4 9:09
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

import os
import logging
logger = logging.getLogger('m3gnet.graph._converters')
logger.setLevel(logging.ERROR)

import numpy as np
from pymatgen.core import Structure
from m3gnet.models import M3GNet

from calculators._calculator_base import CalculatorBase

CWD = os.path.dirname(os.path.abspath(__file__))


class BulkModulusByM3GNET(CalculatorBase):
    def __init__(self, model_path=None, **kwargs):
        if model_path is None:
            model_path = os.path.join(CWD, 'model')
        self.m3gnet = M3GNet.from_dir(dirname=model_path)

    def get_target(self, struc: Structure, **kwargs):
        pred = self.m3gnet.predict_structure(structure=struc)
        pred = np.array(pred).reshape(-1)[0]
        return pred


if __name__ == '__main__':
    _struc = Structure.from_file(r"F:\desktop\MouseWithoutBorders\Si20_20_-5.167386_1655_2744.cif")
    print(10**BulkModulusByM3GNET().get_target(_struc), 'GPa')
