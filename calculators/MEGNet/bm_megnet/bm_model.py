# -*- coding: utf-8 -*-
# ========================================================================
#  2023/5/17 22:29
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

import numpy as np
from pymatgen.core import Structure
from keras.models import load_model
from megnet.layers import _CUSTOM_OBJECTS
from megnet.data.crystal import CrystalGraph
from megnet.data.graph import GaussianDistance
from megnet.models import MEGNetModel, GraphModel

from calculators._calculator_base import CalculatorBase


class MEGNetBMCalculator(CalculatorBase):
    def __init__(self):
        r_cutoff = 5
        nfeat_bond = 100
        gaussian_width = 0.5
        gaussian_centers = np.linspace(0, r_cutoff + 1, nfeat_bond)
        graph_converter = CrystalGraph(cutoff=r_cutoff)

        model_path = os.path.join(os.path.dirname(__file__), 'bm_megnet_model.hdf5')

        model = load_model(model_path, custom_objects=_CUSTOM_OBJECTS)
        self.model = GraphModel(model=model,
                                graph_converter=graph_converter,
                                centers=gaussian_centers,
                                width=gaussian_width, )

    def get_target(self, struc: Structure, **kwargs):
        """
        Get the bulk modulus of the structure, unit: GPa
        :param struc:
        :param kwargs:
        :return:
        """
        target = self.model.predict_structure(struc).reshape(-1)[0]
        return target


if __name__ == '__main__':
    pass
