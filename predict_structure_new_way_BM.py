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

import pandas as pd

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import random
import itertools

import numpy as np
from scipy.stats import norm

from pymatgen.core import Structure, Lattice

from utils.file_utils import check_and_rename_path, check_path, save_data_csv
from utils.read_input import ReadInput
from utils.compound_utils import get_elements_info, get_element_combinations_2, get_element_count_combinations_2
# from utils.children_cell_utils import get_all_children_cell, get_all_children_wyckoff_combination
from utils.children_cell_utils import get_all_children_cell_2, get_children_cell_2
from utils.parameter_utils import parameter_config
from utils.print_utils import print_header, print_run_info
from utils.gen_crystal_WyckPos import GenCrystalWyckPos, get_wyckoffs_strings, wp_alphabet
from utils.compound_utils import get_single_atom_energy
from utils.gen_crystal_WyckPos_old import get_all_wyckoff_combination
from utils.compound_utils import get_single_atom_energy, get_single_compound_energy
from calculators.m3gnet.bulk_modulus_prediction.EOS.bulk_modulus import BulkModulusByM3GNetEOS
from calculators.m3gnet.bulk_modulus_prediction.stress_strain.bulk_modulus import BulkModulusByM3GNetStressStrain
from calculators.MEGNet.bm_megnet.bm_model import MEGNetBMCalculator


class PredictStructure:
    @print_header
    def __init__(self, input_file_path='dream.in', input_config=None, **kwargs):
        self.input_config = ReadInput(input_file_path=input_file_path, input_config=input_config)

        if not self.input_config.is_use_gpu:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

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

        self.output_path = os.path.join(self.input_config.output_path, 'results')
        self.structures_path = os.path.join(self.output_path, 'eles_screen', 'structures')
        self.step_result_file = os.path.join(self.output_path, 'eles_screen', 'step_result.csv')
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

        self.single_atom_energy = get_single_atom_energy()
        self.single_compound_energy = get_single_compound_energy()
        self.bm_calculator1 = BulkModulusByM3GNetStressStrain(calculator=self.calculator.calcu)
        self.bm_calculator2 = MEGNetBMCalculator()

        self.elements_info = get_elements_info()

        self.step_number = 0
        self.structure_number = 0
        self.start_time = 0

        all_elements = [i for e in self.input_config.elements for i in e]
        all_elements = list(set(all_elements))
        elements_used_count = {i: 0 for i in all_elements}
        elements_used_count_range = [20, 25]
        # elements_used_count_range = [1, 2]
        self.elements_energy_list = {i: [] for i in all_elements}
        self.max_sample_count_per_ele_comb = 1
        self.is_screen = 1

        comb_count = 0
        while all_elements:
            comb_count += 1
            # random_items = random.sample(my_list, ele_n)
            random_items = random.choices(all_elements, k=self.composition_count)
            for i in set(random_items):
                elements_used_count[i] += 1
                if elements_used_count[i] >= elements_used_count_range[1] and i in all_elements:
                    all_elements.remove(i)

            self.elements = [[i] for i in random_items]
            try:
                self.step_number = 0
                self.structure_number = 0
                self.find_stable_structure()
            except Exception as e:
                print(comb_count, self.elements, 'OK!')

            if min(elements_used_count.values()) >= elements_used_count_range[0]:
                break

        elements_energy_norms = []
        for k, v in self.elements_energy_list.items():
            mu = np.average(v)
            sigma = np.std(v)
            p = norm.cdf(250, loc=mu, scale=sigma)
            p = 1 - p
            elements_energy_norms.append([k, mu, sigma, p, v])
        elements_energy_norms.sort(key=lambda x: x[3], reverse=True)
        save_data_csv(os.path.join(self.output_path, 'eles_screen'), 'ele_prob.csv',
                      data=elements_energy_norms,
                      header=['elements', 'mu', 'sigma', 'prob(> 250)', 'energies'])

        print('='*50)

        self.structures_path = os.path.join(self.output_path, 'comb_screen', 'structures')
        self.step_result_file = os.path.join(self.output_path, 'comb_screen', 'step_result.csv')
        check_path(self.structures_path)
        self.step_result_file = os.path.join(self.output_path, 'comb_screen', 'step_result.csv')
        if not os.path.exists(self.step_result_file):
            self.create_save_results_data_file()

        self.is_screen = 2
        self.element_comb_energy_list = {}
        # self.is_screen = 1
        # self.elements_energy_list = {i: [] for i in all_elements}

        # self.step_number = 0
        # self.structure_number = 0
        # self.max_sample_count_per_ele_comb = 100
        # self.elements = [[i[0] for i in elements_energy_norms[:10]], ]*2
        # print(self.elements)
        # self.input_config.algorithm = 'rand'
        # self.input_config.max_step = 2000
        # try:
        #     self.find_stable_structure()
        # except Exception as e:
        #     print('OK!')

        all_elements = [i[0] for i in elements_energy_norms[:10]]
        print(all_elements)
        elements_used_count = {i: 0 for i in all_elements}
        elements_used_count_range = [40, 45]
        # elements_used_count_range = [1, 2]
        self.max_sample_count_per_ele_comb = 1

        comb_count = 0
        while all_elements:
            comb_count += 1
            # random_items = random.sample(my_list, ele_n)
            random_items = random.choices(all_elements, k=self.composition_count)
            for i in set(random_items):
                elements_used_count[i] += 1
                if elements_used_count[i] >= elements_used_count_range[1] and i in all_elements:
                    all_elements.remove(i)

            self.elements = [[i] for i in random_items]
            try:
                self.step_number = 0
                self.structure_number = 0
                self.find_stable_structure()
            except Exception as e:
                print(comb_count, self.elements, 'OK!')

            if min(elements_used_count.values()) >= elements_used_count_range[0]:
                break

        print(self.element_comb_energy_list)
        # elements_energy_norms = []
        # for k, v in self.elements_energy_list.items():
        #     mu = np.average(v)
        #     sigma = np.std(v)
        #     p = norm.cdf(-7.9, loc=mu, scale=sigma)
        #     elements_energy_norms.append([k, mu, sigma, p, v])
        # elements_energy_norms.sort(key=lambda x: x[3], reverse=True)
        # save_data_csv(os.path.join(self.output_path, 'comb_screen'), 'ele_prob.csv',
        #               data=elements_energy_norms,
        #               header=['elements', 'mu', 'sigma', 'prob(> -7.9)', 'energies'])
        print('='*50)

        # element_comb_energy_list_top = []
        # for comb_prob in itertools.combinations_with_replacement(elements_energy_norms, self.composition_count):
        #     k = []
        #     v = 1
        #     for i in comb_prob:
        #         k.append(i[0])
        #         v *= i[3]
        #     k.sort()
        #     k = '_'.join(k)
        #     element_comb_energy_list_top.append([k, v])
        # element_comb_energy_list_top.sort(key=lambda x: x[1], reverse=True)

        element_comb_energy_list_top = sorted(self.element_comb_energy_list.items(), key=lambda x: min(x[1]))

        element_comb_energy_list_top = element_comb_energy_list_top[:5]
        print(element_comb_energy_list_top)
        for k, v in element_comb_energy_list_top:
            self.structures_path = os.path.join(self.output_path, 'comp_pred', k, 'structures')
            self.step_result_file = os.path.join(self.output_path, 'comp_pred', k, 'step_result.csv')
            check_path(self.structures_path)
            self.step_result_file = os.path.join(self.output_path, 'comp_pred', k, 'step_result.csv')
            if not os.path.exists(self.step_result_file):
                self.create_save_results_data_file()
            self.is_screen = 0
            self.step_number = 0
            self.structure_number = 0
            self.elements = [[i] for i in k.split('_')]
            print(self.elements)
            # self.element_comb_energy_list = {}
            self.input_config.algorithm = 'tpe'
            self.input_config.max_step = 500
            self.find_stable_structure()

    def get_formation_energy(self, _Et, struc):
        formation_energy = _Et * struc.num_sites
        for p in struc.species:
            sae = self.single_compound_energy[str(p)]
            formation_energy -= sae
        formation_energy /= struc.num_sites
        return formation_energy

    def create_save_results_data_file(self):
        with open(self.step_result_file, 'w+') as f:
            f.write("number,step,compound,bulk_modulus,sg_number,child_cell," + \
                    ','.join(['ele_%d' % (i+1) for i in range(self.composition_count)]) + \
                    ',' + \
                    ','.join(['ele_count_%d' % (i+1) for i in range(self.composition_count)]) + \
                    ",time\n")

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
            elements.update(parameter_config('ele' + str(i), [0, len(self.elements[i])]))
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

        if (self.is_screen > 0) and (self.structure_number >= self.max_sample_count_per_ele_comb):
            raise Exception('done!')

        try:
            struc, expand_param = self.convert_structure_from_struc_param(struc_param)
            # self.atomic_dist_and_volume_limit(struc)
            self.ACS_limit(struc)
            # print(self.calculator)

            if self.use_relax:
                target_property, target_struc = self.calculator.get_target(struc,
                                                                           keep_symmetry=self.input_config.is_use_keep_symmetry,
                                                                           symprec=self.input_config.symprec)
                self.atom_distance_limit(target_struc)
            else:
                target_property = self.calculator.get_target(struc)
                target_struc = struc

            Ef = self.get_formation_energy(target_property, target_struc)
            if Ef > 1.0:
                raise Exception('Ef > 1.0, unstable structure')

            # target_property = self.properties_process(target_property, struc_param, target_struc)
            formula = struc.formula.replace(' ', '')
            # self.structure_number = len(glob.glob1(self.structures_path, "*.cif"))
            target_property1 = self.bm_calculator1.get_target(target_struc,
                                                              results_dir=os.path.join(self.output_path,
                                                                                       'tmp_%s_%d_%d' % (formula, self.structure_number, self.step_number)))
            target_property = self.bm_calculator2.get_target(target_struc)
            if abs(target_property1 - target_property) > 200:
                raise Exception('pass')
            self.structure_number += 1
            self.save_data_for_successful_step(target_property, struc_param, target_struc, expand_param, step_number=self.step_number)

            if self.is_screen == 1:
                for i in range(self.composition_count):
                    ele_i = self.elements[i][int(struc_param['ele' + str(i)])]
                    self.elements_energy_list[ele_i].append(target_property)
            if self.is_screen == 2:
                e_comb = [self.elements[i][int(struc_param['ele' + str(i)])] for i in range(self.composition_count)]
                e_comb.sort()
                e_comb_key = '_'.join(e_comb)
                self.element_comb_energy_list[e_comb_key] = self.element_comb_energy_list.get(e_comb_key, [])
                self.element_comb_energy_list[e_comb_key].append(target_property)

        except Exception as e:
            # print(e)
            # print(e.__traceback__.tb_frame.f_globals["__file__"])  # 发生异常所在的文件
            # print(e.__traceback__.tb_lineno)  # 发生异常所在的行数
            target_property = -99999

        return -target_property

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

        elements, elements_count = [], []
        for i in range(self.composition_count):
            elements.append(self.elements[i][int(struc_param['ele' + str(i)])])
            elements_count.append(str(self.input_config.elements_count[i][int(struc_param['ele_c' + str(i)])]))

        with open(self.step_result_file, 'a+') as f:
            f.write(','.join([str(self.structure_number),
                              str(self.step_number),
                              str(formula),
                              str(target_property),
                              str(self.input_config.space_group[struc_param['sg_index']]),
                              # str(list(self.all_children_cell_dict[elements_type_count][elements_count_index].keys())[struc_param['child_cell_index']]),
                              str(child_cell['child_cell']),
                              ','.join(elements),
                              ','.join(elements_count),
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
            elements.append(self.elements[i][int(struc_parameters['ele' + str(i)])])
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
        structure.to(fmt='cif', filename=os.path.join(self.structures_path, 'temp.cif'))
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
