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
import glob
import warnings

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from predict_structure_mpi import PredictStructureMPI
from utils.compound_utils import get_single_atom_energy, get_single_compound_energy
from calculators.m3gnet.bulk_modulus_prediction.EOS.bulk_modulus import BulkModulusByM3GNetEOS
from calculators.m3gnet.bulk_modulus_prediction.stress_strain.bulk_modulus import BulkModulusByM3GNetStressStrain
from calculators.MEGNet.bm_megnet.bm_model import MEGNetBMCalculator


class FindMaxBulkModulusMPI(PredictStructureMPI):
    def __init__(self, input_file_path='dream.in', input_config=None, study_name='binding_energy', **kwargs):
        self.single_atom_energy = get_single_atom_energy()
        self.single_compound_energy = get_single_compound_energy()
        super().__init__(input_file_path, input_config, study_name, **kwargs)

    def some_settings_before_run(self, **kwargs):
        # BMClass = BulkModulusByM3GNetEOS
        # BMClass = BulkModulusByM3GNetStressStrain
        # self.bm_calculator = BMClass(calculator=self.calculator.calcu)

        # BMClass = MEGNetBMCalculator
        # self.bm_calculator = BMClass()

        self.bm_calculator0 = BulkModulusByM3GNetEOS(calculator=self.calculator.calcu)
        self.bm_calculator1 = BulkModulusByM3GNetStressStrain(calculator=self.calculator.calcu)
        self.bm_calculator2 = MEGNetBMCalculator()

    def get_formation_energy(self, _Et, struc):
        formation_energy = _Et * struc.num_sites
        for p in struc.species:
            sae = self.single_compound_energy[str(p)]
            formation_energy -= sae
        formation_energy /= struc.num_sites
        return formation_energy

    def get_structure_properties(self, struc_param: dict, **kwargs):
        trial = kwargs.get('trial')
        step_number = trial.number

        try:
            struc, expand_param = self.convert_structure_from_struc_param(struc_param)
            # self.atomic_dist_and_volume_limit(struc)
            self.ACS_limit(struc)

            if self.use_relax:
                target_property, target_struc = self.calculator.get_target(struc)
                self.atom_distance_limit(target_struc)
            else:
                target_property = self.calculator.get_target(struc)
                target_struc = struc

            Ef = self.get_formation_energy(target_property, target_struc)
            if Ef > 1.0:
                raise Exception('Ef > 1.0, unstable structure')

            # target_property = self.properties_process(target_property, struc_param, target_struc)
            formula = struc.formula.replace(' ', '')
            self.structure_number = len(glob.glob1(self.structures_path, "*.cif"))
            # target_property1 = self.bm_calculator1.get_target(target_struc, results_dir='./tmp_%s_%d_%d' % (formula, self.structure_number, step_number))
            target_property = self.bm_calculator2.get_target(target_struc)
            # if abs(target_property1-target_property) > 200:
            #     raise Exception('pass')
            self.save_data_for_successful_step(target_property, struc_param, target_struc, expand_param, step_number=step_number)

        except Exception as e:
            # print(e)
            # print(e.__traceback__.tb_frame.f_globals["__file__"])  # 发生异常所在的文件
            # print(e.__traceback__.tb_lineno)  # 发生异常所在的行数
            target_property = -99999

        return -target_property


if __name__ == '__main__':
    csp = FindMaxBulkModulusMPI(input_file_path=r'dream.in')
