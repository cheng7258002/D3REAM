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
from utils.compound_utils import get_single_compound_energy


class FindCoordinationNumberMaterial(PredictStructure):
    def __init__(self, input_file_path='dream.in', input_config=None, **kwargs):
        self.single_compound_energy = get_single_compound_energy()
        super().__init__(input_file_path, input_config)

    @staticmethod
    def get_struct_NN(_struct):
        _struct_NN = _struct.get_all_neighbors(6.0)

        NN_dist = []
        for s_i, NN_i in zip(_struct, _struct_NN):
            NN_d_i = []
            for s_j in NN_i:
                dist_ij = s_i.distance(s_j)
                # NN_d_i.append(dist_ij)
                # s_i.specie
                r_ij = s_i.specie.atomic_radius + s_j.specie.atomic_radius
                r_ij *= 1.2
                if dist_ij < r_ij:
                    NN_d_i.append(dist_ij)
            NN_dist.append(NN_d_i)

        NNN = [len(i) for i in NN_dist]

        return _struct_NN, NN_dist, NNN

    def properties_process(self, _properties, struc_param, struc, **kwargs):
        _, _, NNN = self.get_struct_NN(struc)
        if not (4 in NNN and 6 in NNN):
            raise 'mismatch condition! skip!'

        formation_energy = _properties * struc.num_sites
        for p in struc.species:
            sce = self.single_compound_energy[str(p)]
            formation_energy -= sce
        formation_energy /= struc.num_sites

        if formation_energy <= 1.0:
            return formation_energy
        else:
            raise 'mismatch condition! skip!'


if __name__ == '__main__':
    csp = FindCoordinationNumberMaterial(input_file_path=r'dream.in')
