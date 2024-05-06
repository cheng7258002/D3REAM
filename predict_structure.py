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
import time
import warnings

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
from pymatgen.core import Structure, Lattice

from utils.file_utils import check_and_rename_path, check_path
from utils.read_input import ReadInput
from utils.compound_utils import get_elements_info, get_element_combinations_2, get_element_count_combinations_2
# from utils.children_cell_utils import get_all_children_cell, get_all_children_wyckoff_combination
from utils.children_cell_utils import get_all_children_cell_2, get_children_cell_2
from utils.parameter_utils import parameter_config
from utils.print_utils import print_header, print_run_info
from utils.gen_crystal_WyckPos import GenCrystalWyckPos, get_wyckoffs_strings, wp_alphabet
from utils.gen_crystal_WyckPos_old import get_all_wyckoff_combination


class PredictStructure:
    @print_header
    def __init__(self, input_file_path='dream.in', input_config=None, **kwargs):
        self.input_config = ReadInput(input_file_path=input_file_path, input_config=input_config)

        if not self.input_config.is_use_gpu:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

        # self.compound = self.input_config.compound
        # self.elements = list(itertools.product(*self.input_config.elements))
        # self.elements_combination_dict, self.max_elements_combination_count = get_element_combinations_3(self.input_config.elements)
        # print(self.elements_combination_dict)
        # self.elements_combinations = get_element_combinations_2(self.input_config.elements)
        # self.elements_count = list(itertools.product(*self.input_config.elements_count))
        # self.max_elements_count = max([int(sum(ec)) for ec in self.elements_count])
        # self.elements_count_combination_dict, self.max_elements_count_combination_count, self.max_atom_count = get_element_count_combinations_2(self.input_config.elements_count)
        # print(self.elements_count_combination_dict)

        # print(self.input_config.elements, self.input_config.elements_count)
        # exit()
        self.max_atom_count = np.sum(np.max(self.input_config.elements_count, axis=1))
        self.max_cells_count = np.max(self.input_config.elements_count)
        self.composition_count = len(self.input_config.elements)

        if self.input_config.wyck_pos_gen in [1, 2]:
            self.WyckPos_comb, self.max_comb_count = get_all_wyckoff_combination(self.input_config.space_group,
                                                                                 self.input_config.elements_count,
                                                                                 max_count=self.input_config.max_wyck_pos_count)
        elif self.input_config.wyck_pos_gen in [3, 4]:
            self.is_use_flexible_site = self.input_config.is_use_flexible_site
        self.space_group = self.input_config.space_group
        # self.all_children_cell, self.max_cells_count = get_all_children_cell(self.elements_count)
        # self.all_children_wyckoffs_dict, self.max_wyckoffs_count = get_all_children_wyckoff_combination(self.all_children_cell, self.space_group)
        # self.all_children_cell_dict, self.max_cells_count = get_all_children_cell_2(self.elements_count_combination_dict,
        #                                                                             is_use_children_cell=self.input_config.is_use_children_cell)
        # self.all_children_wyckoffs_dict_dict, self.max_wyckoffs_count = get_all_children_wyckoff_combination_2(self.all_children_cell_dict, self.space_group)

        self.output_path = os.path.join(self.input_config.output_path, 'results')
        self.structures_path = os.path.join(self.output_path, 'structures')
        self.step_result_file = os.path.join(self.output_path, 'step_result.csv')
        self.is_mpi_run = kwargs.get('is_mpi_run', False)
        if self.is_mpi_run:
            check_path(self.output_path)
            check_path(self.structures_path)
            if not os.path.exists(self.step_result_file):
                self.create_save_results_data_file()
        else:
            check_and_rename_path(self.output_path)
            check_and_rename_path(self.structures_path)
            self.create_save_results_data_file()

        self.use_relax = self.input_config.is_use_calculator_relax
        self.calculator = None
        self.calculator_config()

        self.elements_info = get_elements_info()
        self.step_number = 0
        self.structure_number = 0
        self.start_time = 0

        self.some_settings_before_run(**kwargs)

        self.find_stable_structure()

    def some_settings_before_run(self, **kwargs):
        pass

    def create_save_results_data_file(self):
        with open(self.step_result_file, 'w+') as f:
            f.write("number,step,compound,energy,sg_number,child_cell,time\n")

    def calculator_config(self):
        if self.input_config.calculator == 'megnet':
            from calculators.MEGNet.orig_megnet import OrigMEGNet
            self.use_relax = False
            self.calculator = OrigMEGNet()
            self.calculator.from_file(self.input_config.calculator_path)
        elif self.input_config.calculator == 'm3gnet':
            from calculators.m3gnet.origin_m3gnet import OrinM3GNET
            self.calculator = OrinM3GNET(self.input_config.calculator_path, use_relax=self.use_relax)
        elif self.input_config.calculator == 'vasp':
            from calculators.VASP.vasp import VASP
            # self.use_relax = False
            self.calculator = VASP(None, self.output_path, True, use_relax=self.use_relax)
        else:
            raise Exception("The parameter `calculator` setting error!")

        # return calculator

    def struc_parameter_config(self):
        # len_elements = len(self.elements)
        # elements_type_key = sorted(list(self.elements_combination_dict.keys()))
        # elements_type_count = parameter_config('elements_type_count', elements_type_key, vtype='choice')
        # len_elements_combinations = len(self.elements_combinations)
        # elements = parameter_config('elements_combination_index', [0, len_elements_combinations])
        # elements_index = parameter_config('elements_combination_index', [0, len_elements_combinations])
        # elements_count_index = parameter_config('elements_count_index', list(range(0, len(self.elements_count))), vtype='choice')
        # elements_count_index = parameter_config('elements_count_combination_index', [0, self.max_elements_count_combination_count])

        elements = {}
        elements_count = {}
        for i in range(self.composition_count):
            elements.update(parameter_config('ele' + str(i), [0, len(self.input_config.elements[i])]))
            elements_count.update(parameter_config('ele_c' + str(i), [0, len(self.input_config.elements_count[i])]))

        # a = parameter_config('a', self.input_config.lattice_a, vtype='choice')
        # b = parameter_config('b', self.input_config.lattice_b, vtype='choice')
        # c = parameter_config('c', self.input_config.lattice_c, vtype='choice')
        # alpha = parameter_config('alpha', self.input_config.lattice_alpha, vtype='choice')
        # beta = parameter_config('beta', self.input_config.lattice_beta, vtype='choice')
        # gamma = parameter_config('gamma', self.input_config.lattice_gamma, vtype='choice')
        a = parameter_config('a', self.input_config.lattice_a)
        b = parameter_config('b', self.input_config.lattice_b)
        c = parameter_config('c', self.input_config.lattice_c)
        alpha = parameter_config('alpha', self.input_config.lattice_alpha)
        beta = parameter_config('beta', self.input_config.lattice_beta)
        gamma = parameter_config('gamma', self.input_config.lattice_gamma)
        # sg = parameter_config('sg', self.input_config.space_group, vtype='choice')
        sg_index = parameter_config('sg_index', [0, len(self.input_config.space_group)])
        child_cell_index = parameter_config('child_cell_index', [0, self.max_cells_count])

        wp_sites = {}
        atom_pos = {}
        for i in range(self.max_atom_count):
            i += 1

            wp_sites.update(parameter_config('wp' + str(i), [0, 26]))

            atom_pos.update(parameter_config('x' + str(i), [0, 1]))
            atom_pos.update(parameter_config('y' + str(i), [0, 1]))
            atom_pos.update(parameter_config('z' + str(i), [0, 1]))

        if self.input_config.wyck_pos_gen in [1, 2]:
            # wp_index = parameter_config('wp_index', [0, self.max_comb_count])
            wp_sites = parameter_config('wp_index', [0, self.max_comb_count])

        parameter = {
            # **elements_type_count,
            # **elements, **elements_count_index,
            **elements, **elements_count,
            # **elements_index, **elements_count_index,
            **a, **b, **c, **alpha, **beta, **gamma,
            # **sg, **child_cell_index,
            **sg_index, **child_cell_index,
            # **wp_index,
            **wp_sites,
            **atom_pos}

        return parameter

    def opt_algo_parameter_config(self):
        algorithm = self.input_config.algorithm
        n_init = self.input_config.n_init
        max_step = self.input_config.max_step
        rand_seed = self.input_config.rand_seed
        n_mpi = self.input_config.n_mpi
        storage = self.input_config.storage

        parameter = {'algo': algorithm,
                     'n_init': n_init, 'max_step': max_step, 'rand_seed': rand_seed,
                     'n_mpi': n_mpi, 'storage': storage}
        return parameter

    @print_run_info('Predict crystal structure')
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
            opt = Optuna(self.get_structure_properties, struc_param, algo_param)
        else:
            raise Exception("The parameter `algorithm` setting error!")

        self.start_time = time.time()
        opt.run()

    def get_structure_properties(self, struc_param: dict, **kwargs):
        self.step_number += 1

        try:
            struc, child_cell = self.convert_structure_from_struc_param(struc_param)
            # self.atomic_dist_and_volume_limit(struc)
            self.ACS_limit(struc)

            if self.use_relax:
                target_property, target_struc = self.calculator.get_target(struc,
                                                                           keep_symmetry=self.input_config.is_use_keep_symmetry,
                                                                           symprec=self.input_config.symprec)
                self.atom_distance_limit(target_struc)
            else:
                target_property = self.calculator.get_target(struc)
                target_struc = struc

            target_property = self.properties_process(target_property, struc_param, target_struc)
            self.structure_number += 1
            self.save_data_for_successful_step(target_property, struc_param, target_struc, child_cell)

        except Exception as e:
            # print(e)
            # print(e.__traceback__.tb_frame.f_globals["__file__"])  # 发生异常所在的文件
            # print(e.__traceback__.tb_lineno)  # 发生异常所在的行数
            target_property = 999

        return target_property

    def properties_process(self, _properties, struc_param: dict, struc: Structure, **kwargs):
        return _properties

    def save_data_for_successful_step(self, target_property, struc_param: dict, struc: Structure, child_cell, **kwargs):
        struc.make_supercell(child_cell['expand_param'])
        formula = struc.formula.replace(' ', '')
        # elements_combination_index = int(struc_param['elements_combination_index'])
        # elements = self.elements_combinations[elements_combination_index]
        # elements_type_count = len(elements)
        # elements_count_index = int(
        #     struc_param['elements_count_combination_index'] * len(self.elements_count_combination_dict[elements_type_count]) / self.max_elements_count_combination_count)

        with open(self.step_result_file, 'a+') as f:
            f.write(','.join([str(self.structure_number),
                              str(self.step_number),
                              str(formula),
                              str(target_property),
                              str(self.input_config.space_group[struc_param['sg_index']]),
                              # str(list(self.all_children_cell_dict[elements_type_count][elements_count_index].keys())[struc_param['child_cell_index']]),
                              str(child_cell['child_cell']),
                              str(time.time() - self.start_time)]) + '\n')

        structure_file_name = os.path.join(
            self.structures_path,
            '%f_%s_%d_%d.cif' % (target_property, formula, self.structure_number, self.step_number))
        # struc.to(fmt='cif', filename=structure_file_name, symprec=self.input_config.symprec)
        struc.to(fmt='cif', filename=structure_file_name)

    def convert_structure_from_struc_param(self, struc_parameters):
        # elements_combination_index = int(struc_parameters['elements_combination_index'])
        # elements = self.elements_combinations[elements_combination_index]
        # elements_type_count = len(elements)

        # elements_count = self.elements_count[struc_parameters['elements_count_index']]
        sg_index = struc_parameters['sg_index'] = int(struc_parameters['sg_index'])
        sg = self.input_config.space_group[sg_index]

        # child_cells_from_elements_count = self.all_children_cell[struc_parameters['elements_count_index']]
        # elements_count_index = int(
        #     struc_parameters['elements_count_combination_index'] * len(self.elements_count_combination_dict[elements_type_count]) / self.max_elements_count_combination_count)

        elements, elements_count = [], []
        for i in range(self.composition_count):
            elements.append(self.input_config.elements[i][int(struc_parameters['ele' + str(i)])])
            elements_count.append(self.input_config.elements_count[i][int(struc_parameters['ele_c' + str(i)])])

        # child_cells_from_elements_count = self.all_children_cell_dict[elements_type_count][elements_count_index]
        # child_cell_index = struc_parameters['child_cell_index']
        # struc_parameters['child_cell_index'] = int(child_cell_index * len(child_cells_from_elements_count) / self.max_cells_count)
        # child_cell_key = list(child_cells_from_elements_count.keys())[struc_parameters['child_cell_index']]

        child_cell_tmp = get_children_cell_2(elements_count, self.input_config.is_use_children_cell)
        child_cell_index = int(struc_parameters['child_cell_index'] * len(child_cell_tmp) / self.max_cells_count)
        child_cell = child_cell_tmp[child_cell_index]
        child_cell_elements_count = child_cell['atoms']
        child_cell_expand_param = child_cell['expand_param']

        if self.input_config.wyck_pos_gen in [1, 2]:
            # wp_list_from_elements_count = self.all_children_wyckoffs_dict[struc_parameters['elements_count_index']]
            # wp_list_from_elements_count = self.all_children_wyckoffs_dict_dict[n_elements_count][elements_count_index]
            # wp_list = wp_list_from_elements_count[child_cell_key][sg]
            wp_list = self.WyckPos_comb['_'.join([str(i) for i in child_cell_elements_count])][sg]
            struc_parameters['wp_index'] = int(struc_parameters['wp_index'] * len(wp_list) / self.max_comb_count)
            wp = wp_list[struc_parameters['wp_index']]
        else:
            wyckoffs_strings = get_wyckoffs_strings(sg)
            wyckoffs_strings_len = len(wyckoffs_strings)
            wp_site = []
            wp_i = 1
            for ccec in child_cell_elements_count:
                wps_tmp = []
                for i in range(ccec):
                    wp_letter_index = int(struc_parameters['wp' + str(wp_i)] * wyckoffs_strings_len / 27.0)
                    wps_tmp.append(wp_alphabet[wp_letter_index])
                wp_site.append(wps_tmp)
                wp_i += ccec
            gcw = GenCrystalWyckPos(
                sg, elements, child_cell_elements_count,
                sites=wp_site,
                flexible_site=self.is_use_flexible_site,
            )
            gcw.wyckoff_string = wyckoffs_strings
            gcw.wyckoff_string_len = wyckoffs_strings_len
            wp = gcw.get_WyckPos_combination()

        atoms = []
        atom_positions = []
        count = 1
        for i, wp_i in enumerate(wp):
            for wp_i_j in wp_i:
                atoms += [elements[i]] * len(wp_i_j)

                for wp_i_j_k in wp_i_j:
                    if 'x' in wp_i_j_k:
                        wp_i_j_k = wp_i_j_k.replace('x', str(struc_parameters['x' + str(count)]))
                    if 'y' in wp_i_j_k:
                        wp_i_j_k = wp_i_j_k.replace('y', str(struc_parameters['y' + str(count)]))
                    if 'z' in wp_i_j_k:
                        wp_i_j_k = wp_i_j_k.replace('z', str(struc_parameters['z' + str(count)]))
                    atom_positions.append(list(eval(wp_i_j_k)))
                count += len(wp_i_j)

        if sg in [0, 1, 2]:
            lattice = Lattice.from_parameters(a=struc_parameters['a'], b=struc_parameters['b'], c=struc_parameters['c'],
                                              alpha=struc_parameters['alpha'], beta=struc_parameters['beta'], gamma=struc_parameters['gamma'])
        elif sg in list(range(3, 15 + 1)):
            lattice = Lattice.from_parameters(a=struc_parameters['a'], b=struc_parameters['b'], c=struc_parameters['c'],
                                              alpha=90, beta=struc_parameters['beta'], gamma=90)
        elif sg in list(range(16, 74 + 1)):
            lattice = Lattice.from_parameters(a=struc_parameters['a'], b=struc_parameters['b'], c=struc_parameters['c'],
                                              alpha=90, beta=90, gamma=90)
        elif sg in list(range(75, 142 + 1)):
            lattice = Lattice.from_parameters(a=struc_parameters['a'], b=struc_parameters['a'], c=struc_parameters['c'],
                                              alpha=90, beta=90, gamma=90)
        elif sg in list(range(143, 194 + 1)):
            lattice = Lattice.from_parameters(a=struc_parameters['a'], b=struc_parameters['a'], c=struc_parameters['c'],
                                              alpha=90, beta=90, gamma=120)
        elif sg in list(range(195, 230 + 1)):
            lattice = Lattice.from_parameters(a=struc_parameters['a'], b=struc_parameters['a'], c=struc_parameters['a'],
                                              alpha=90, beta=90, gamma=90)
        else:
            lattice = Lattice.from_parameters(a=struc_parameters['a'], b=struc_parameters['b'], c=struc_parameters['c'],
                                              alpha=struc_parameters['alpha'], beta=struc_parameters['beta'], gamma=struc_parameters['gamma'])

        structure = Structure(lattice, atoms, atom_positions)
        structure.to(fmt='cif', filename=os.path.join(self.output_path, 'structures', 'temp.cif'))
        return structure, child_cell

    def ACS_limit(self, struc: Structure):
        if not self.use_relax:
            self.atom_distance_limit(struc)
        # self.atom_distance_limit(struc)
        self.structure_volume_limit(struc)
        self.vacuum_size_limit(struc=struc.copy())

    def atom_distance_limit(self, struc: Structure):
        total_atom_count = struc.num_sites
        min_atomic_dist = self.input_config.min_atomic_dist_limit

        if min_atomic_dist > 0:
            for i in range(total_atom_count - 1):
                for j in range(i + 1, total_atom_count):
                    if struc.get_distance(i, j) < min_atomic_dist:
                        raise Exception()
        elif min_atomic_dist < 0:
            for i in range(total_atom_count - 1):
                for j in range(i + 1, total_atom_count):
                    if struc.get_distance(i, j) < (struc.species[i].atomic_radius + struc.species[j].atomic_radius) * abs(min_atomic_dist):
                        raise Exception()
        else:
            return

    def structure_volume_limit(self, struc: Structure):
        volume_limit = self.input_config.volume_limit
        if volume_limit[1] <= 0:
            return

        atom_volume = [4.0 * np.pi * ss.atomic_radius ** 3 / 3.0 for ss in struc.species]
        sum_atom_volume = sum(atom_volume)

        if sum_atom_volume > volume_limit[1]:
            return

        if not (sum_atom_volume * volume_limit[0] <= struc.volume <= sum_atom_volume * volume_limit[1]):
            raise Exception()

    def vacuum_size_limit(self, struc: Structure):
        max_vacuum_limit = self.input_config.max_vacuum_limit
        if max_vacuum_limit <= 0:
            return

        def get_foot(p, a, b):
            p = np.array(p)
            a = np.array(a)
            b = np.array(b)
            ap = p - a
            ab = b - a
            result = a + np.dot(ap, ab) / np.dot(ab, ab) * ab
            return result

        def get_distance(a, b):
            return np.sqrt(np.sum(np.square(b - a)))

        struc.make_supercell([2, 2, 2], to_unit_cell=False)
        line_a_points = [[0, 0, 0], ]
        line_b_points = [[0, 0, 1], [0, 1, 0], [1, 0, 0],
                         [0, 1, 1], [1, 0, 1], [1, 1, 0], [0, 1, -1], [1, 0, -1], [1, -1, 0],
                         [1, 1, 1], [1, 1, -1], [1, -1, 1], [-1, 1, 1]]
        for a in line_a_points:
            for b in line_b_points:
                foot_points = []
                for p in struc.frac_coords:
                    f_p = get_foot(p, a, b)
                    foot_points.append(f_p)
                foot_points = sorted(foot_points, key=lambda x: [x[0], x[1], x[2]])

                # 转为笛卡尔坐标
                foot_points = np.asarray(np.mat(foot_points) * np.mat(struc.lattice.matrix))
                for fp_i in range(0, len(foot_points) - 1):
                    fp_distance = get_distance(foot_points[fp_i + 1], foot_points[fp_i])
                    if fp_distance > max_vacuum_limit:
                        raise Exception()


if __name__ == '__main__':
    csp = PredictStructure(input_file_path=r'F:\d3ream\example\CaS\dream.in')
    # csp = PredictStructure(input_file_path=r'dream.in')
