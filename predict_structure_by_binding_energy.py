# -*- coding: utf-8 -*-
# ========================================================================
#  2021/04/26 下午 09:57
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
import warnings

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from predict_structure import PredictStructure
from utils.compound_utils import get_single_atom_energy


class FindMaxBindingEnergyMaterial(PredictStructure):
    def __init__(self, input_file_path='dream.in', input_config=None, **kwargs):
        self.single_atom_energy = get_single_atom_energy()
        super().__init__(input_file_path, input_config)

    def properties_process(self, _properties, struc_param, struc, **kwargs):
        if struc:
            binding_energy = _properties * struc.num_sites
            for p in struc.species:
                # single_atom_energy = self.single_atom_energy[str(p)][0]  # no spin
                single_atom_energy = self.single_atom_energy[str(p)][1]  # with spin
                binding_energy -= single_atom_energy
            binding_energy /= struc.num_sites
            return binding_energy

        return 999


if __name__ == '__main__':
    csp = FindMaxBindingEnergyMaterial(input_file_path=r'dream.in')
