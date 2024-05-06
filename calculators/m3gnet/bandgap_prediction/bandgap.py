# -*- coding: utf-8 -*-
# ========================================================================
#  2022/7/26 23:57
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


class EgClassificationByM3GNET(CalculatorBase):
    def __init__(self, model_path=None, **kwargs):
        if model_path is None:
            model_path = os.path.join(CWD, 'model/classification')
        self.m3gnet = M3GNet.from_dir(dirname=model_path)

    def get_target(self, struc: Structure, **kwargs):
        pred = self.m3gnet.predict_structure(structure=struc)
        pred = np.array(pred).reshape(-1)[0]
        if pred >= 0.5:
            return True
        return False


class EgRegressionByM3GNET(CalculatorBase):
    def __init__(self, model_path=None, **kwargs):
        if model_path is None:
            model_path = os.path.join(CWD, 'model/regression')
        self.m3gnet = M3GNet.from_dir(dirname=model_path)

    def get_target(self, struc: Structure, **kwargs):
        pred = self.m3gnet.predict_structure(structure=struc)
        pred = np.array(pred).reshape(-1)[0]
        return pred


class EgPredByM3GNET(CalculatorBase):
    def __init__(self, classifier_model_path=None, regressor_model_path=None, **kwargs):
        self.classifier = EgClassificationByM3GNET(classifier_model_path)
        self.regressor = EgRegressionByM3GNET(regressor_model_path)

    def get_target(self, struc: Structure, **kwargs):
        is_metal = self.classifier.get_target(struc)
        if is_metal:
            Eg = 0
        else:
            Eg = self.regressor.get_target(struc)

        return Eg


if __name__ == '__main__':
    Eg_predictor = EgPredByM3GNET()
    struct = Structure.from_file(r'F:\d3ream\test\results\test2\Si32\results\structures\-5.191170_Si32_1_13.cif')
    res = Eg_predictor.get_target(struct)
    print(res)
