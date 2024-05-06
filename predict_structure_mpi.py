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
import time
import warnings
import itertools

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from pymatgen.core import Structure, Lattice

from predict_structure import PredictStructure


class PredictStructureMPI(PredictStructure):
    def __init__(self, input_file_path='dream.in', input_config=None, study_name='test', is_mpi_run=True, **kwargs):
        self.study_name = study_name
        super().__init__(input_file_path, input_config, is_mpi_run=is_mpi_run)

    def create_save_results_data_file(self):
        if os.path.isfile(self.step_result_file):
            return
        with open(self.step_result_file, 'w+') as f:
            f.write("number,step,compound,target,sg_number,child_cell,time\n")

    def find_stable_structure(self):
        struc_param = self.struc_parameter_config()
        algo_param = self.opt_algo_parameter_config()

        opt = None
        if self.input_config.algorithm in ['tpe', 'rand', 'anneal']:
            from optimizers.hyperopt_ import HyperOpt
            opt = HyperOpt(self.get_structure_properties, struc_param, algo_param)
        elif self.input_config.algorithm in ['pso', ]:
            from optimizers.sko_ import ScikitOpt
            opt = ScikitOpt(self.get_structure_properties, struc_param, algo_param)
        elif self.input_config.algorithm in ['etpe', ]:
            from optimizers.ultraopt_ import UltraOpt
            opt = UltraOpt(self.get_structure_properties, struc_param, algo_param, use_bohb=False)
        elif self.input_config.algorithm in ['tpe2', ]:
            from optimizers.optuna_ import Optuna
            opt = Optuna(self.get_structure_properties, struc_param, algo_param, is_run_mpi=self.is_mpi_run, study_name=self.study_name)
        else:
            raise Exception("The parameter `algorithm` setting error!")

        self.start_time = time.time()
        opt.run()

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

            target_property = self.properties_process(target_property, struc_param, target_struc)
            self.structure_number = len(glob.glob1(self.structures_path, "*.cif"))
            self.save_data_for_successful_step(target_property, struc_param, target_struc, expand_param, step_number=step_number)

        except Exception as e:
            # print(e)
            # print(e.__traceback__.tb_frame.f_globals["__file__"])  # 发生异常所在的文件
            # print(e.__traceback__.tb_lineno)  # 发生异常所在的行数
            target_property = 999

        return target_property

    def properties_process(self, _properties, struc_param: dict, struc: Structure, **kwargs):
        return _properties

    def save_data_for_successful_step(self, target_property, struc_param: dict, struc: Structure, child_cell, **kwargs):
        step_number = kwargs.get('step_number', 0) + 1

        struc.make_supercell(child_cell['expand_param'])
        formula = struc.formula.replace(' ', '')

        # structure_number = len(glob.glob1(self.structures_path, "*.cif")) + 1
        structure_number = self.structure_number
        with open(self.step_result_file, 'a+') as f:
            f.write(','.join([str(structure_number),
                              str(step_number),
                              str(formula),
                              str(target_property),
                              str(struc_param['sg_index']),
                              str(child_cell['child_cell']),
                              str(time.time() - self.start_time)]) + '\n')

        structure_file_name = os.path.join(
            self.structures_path,
            '%f_%s_%d_%d.cif' % (target_property, formula, structure_number, step_number))
        struc.to(fmt='cif', filename=structure_file_name)


if __name__ == '__main__':
    csp = PredictStructureMPI(input_file_path=r'dream.in')
