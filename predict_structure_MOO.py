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
# Multi-objective Optimization for Formation Energy and Band Gap
#
# ========================================================================

import os
import time
import warnings

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from pymatgen.core import Structure

from predict_structure import PredictStructure
from calculators.m3gnet.bandgap_prediction.bandgap import EgPredByM3GNET
from utils.compound_utils import get_single_compound_energy


class FindPhotovoltaicMaterial(PredictStructure):
    def __init__(self, input_file_path='dream.in', input_config=None, study_name=None, is_mpi_run=False, **kwargs):
        self.single_compound_energy = get_single_compound_energy()
        self.calculator_Eg = EgPredByM3GNET()
        self.study_name = study_name
        super().__init__(input_file_path, input_config, is_mpi_run=is_mpi_run)

    def find_stable_structure(self):
        struc_param = self.struc_parameter_config()
        algo_param = self.opt_algo_parameter_config()

        opt = None
        if self.input_config.algorithm in ['tpe2', ]:
            from optimizers.optuna_ import Optuna
            opt = Optuna(self.get_structure_properties, struc_param, algo_param, is_run_mpi=self.is_mpi_run, study_name=self.study_name, target_num=2)
        else:
            raise Exception("The parameter `algorithm` setting error!")

        self.start_time = time.time()
        opt.run()

    def get_structure_properties(self, struc_param: dict, **kwargs):
        # self.step_number += 1
        trial = kwargs.get('trial')
        step_number = trial.number + 1

        try:
            struc, expand_param = self.convert_structure_from_struc_param(struc_param)
            # self.atomic_dist_and_volume_limit(struc)
            self.ACS_limit(struc)

            if self.use_relax:
                total_energy, target_struc = self.calculator.get_target(struc,
                                                                        keep_symmetry=self.input_config.is_use_keep_symmetry,
                                                                        symprec=self.input_config.symprec)
                self.atom_distance_limit(target_struc)
            else:
                total_energy = self.calculator.get_target(struc)
                target_struc = struc

            formation_energy, formation_energy_moo = self.properties_process(total_energy, struc_param, target_struc)

            Eg = self.calculator_Eg.get_target(target_struc)
            Eg_moo = self.Eg_process_2(Eg)

            target_property = [formation_energy_moo, Eg_moo]

            save_target = [formation_energy, formation_energy_moo, Eg, Eg_moo]

            # self.structure_number += 1
            structure_number = trial.study.user_attrs.get("structure_number", 0) + 1
            trial.study.set_user_attr("structure_number", structure_number)
            # print(structure_number, trial.study.user_attrs.get("structure_number", 0))
            self.save_data_for_successful_step(save_target, struc_param, target_struc, expand_param, step_number=step_number, structure_number=structure_number)

        except Exception as e:
            # print(e)
            # print(e.__traceback__.tb_frame.f_globals["__file__"])  # 发生异常所在的文件
            # print(e.__traceback__.tb_lineno)  # 发生异常所在的行数
            target_property = [999, 999]

        return target_property

    def create_save_results_data_file(self):
        with open(self.step_result_file, 'w+') as f:
            f.write("number,step,compound,sg_number,child_cell,formation_energy_moo,band_gap_moo,formation_energy,band_gap,time\n")

    def save_data_for_successful_step(self, target_property, struc_param: dict, struc: Structure, expand_param, **kwargs):
        step_number = kwargs.get('step_number', 0)
        structure_number = kwargs.get('structure_number', 0)
        # print(step_number, structure_number)

        struc.make_supercell(expand_param)
        formula = struc.formula.replace(' ', '')
        elements_combination_index = int(struc_param['elements_combination_index'])
        elements = self.elements_combinations[elements_combination_index]
        elements_type_count = len(elements)
        elements_count_index = int(
            struc_param['elements_count_combination_index'] * len(self.elements_count_combination_dict[elements_type_count]) / self.max_elements_count_combination_count)

        formation_energy, formation_energy_moo, Eg, Eg_moo = target_property

        # structure_number = len(glob.glob1(self.structures_path, "*.cif")) + 1
        with open(self.step_result_file, 'a+') as f:
            f.write(','.join([str(structure_number),
                              str(step_number),
                              str(formula),
                              str(self.input_config.space_group[struc_param['sg_index']]),
                              str(list(self.all_children_cell_dict[elements_type_count][elements_count_index].keys())[struc_param['child_cell_index']]),
                              str(formation_energy_moo),
                              str(Eg_moo),
                              str(formation_energy),
                              str(Eg),
                              str(time.time() - self.start_time)]) + '\n')

        structure_file_name = os.path.join(
            self.structures_path,
            '%f_%f_%s_%d_%d.cif' % (formation_energy, Eg, formula, structure_number, step_number))
        struc.to(fmt='cif', filename=structure_file_name, symprec=self.input_config.symprec)

    def properties_process(self, _properties, struc_param, struc, **kwargs):
        if struc:
            try:
                formation_energy = _properties * struc.num_sites
                sum_sce = 0
                for p in struc.species:
                    single_atom_energy = self.single_compound_energy[str(p)]
                    formation_energy -= single_atom_energy
                    sum_sce += single_atom_energy
                formation_energy_normalization = - formation_energy / sum_sce * 10
                formation_energy /= struc.num_sites
                return formation_energy, formation_energy_normalization
            except Exception as e:
                print(e)
                print(e.__traceback__.tb_frame.f_globals["__file__"])  # 发生异常所在的文件
                print(e.__traceback__.tb_lineno)  # 发生异常所在的行数
                return 999, 999

        return 999, 999

    def Eg_process(self, _Eg, biases=1.5, **kwargs):
        Eg = abs(_Eg - biases)
        return Eg

    def Eg_process_2(self, _Eg, **kwargs):
        biases_range = [0.1, 1.0, 3.0]
        biases_scale = [999.0, 0.5, 20]
        if _Eg <= biases_range[0]:
            Eg = biases_scale[0] * abs(_Eg - biases_range[0])
        elif biases_range[1] <= _Eg <= biases_range[2]:
            Eg = biases_scale[1] * abs(_Eg - biases_range[1])
        elif biases_range[2] <= _Eg:
            Eg = biases_scale[2] * abs(_Eg - biases_range[2]) + biases_scale[1] * abs(biases_range[2] - biases_range[1])
        else:
            # Eg = (_Eg - 0.55) ** 2 - 0.2025
            Eg = 0

        return Eg


if __name__ == '__main__':
    csp = FindPhotovoltaicMaterial(input_file_path=r'dream.in')
