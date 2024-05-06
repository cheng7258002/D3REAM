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

from predict_structure_mpi import PredictStructureMPI
from utils.compound_utils import get_single_compound_energy


class FindMaxFormationEnergyMaterialMPI(PredictStructureMPI):
    def __init__(self, input_file_path='dream.in', input_config=None, study_name='formation_energy', **kwargs):
        self.single_compound_energy = get_single_compound_energy()
        super().__init__(input_file_path, input_config, study_name, **kwargs)

    def properties_process(self, _properties, struc_param, struc, **kwargs):
        if struc:
            formation_energy = _properties * struc.num_sites
            for p in struc.species:
                single_atom_energy = self.single_compound_energy[str(p)]
                formation_energy -= single_atom_energy
            formation_energy /= struc.num_sites
            return formation_energy

        return 999


if __name__ == '__main__':
    csp = FindMaxFormationEnergyMaterialMPI(input_file_path=r'dream.in')
