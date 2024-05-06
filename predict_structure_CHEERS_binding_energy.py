# -*- coding: utf-8 -*-
# ========================================================================
#  2023/6/20 10:09
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
import sys
import time
import random
import configparser
import functools
import multiprocessing
from multiprocessing.dummy import Pool as ThreadPool

import traceback
from joblib import Parallel, delayed
from joblib.externals.loky import set_loky_pickler

from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy.stats import norm
from pymatgen.core import Structure
from pymatgen.core import Element

from utils.file_utils import check_and_rename_path, check_path, save_data_csv
from utils.read_input import ReadInput
from utils.compound_utils import get_elements_info, get_element_combinations_2, get_element_count_combinations_2
# from utils.children_cell_utils import get_all_children_cell, get_all_children_wyckoff_combination
from utils.children_cell_utils import get_all_children_cell_2, get_children_cell_2
from utils.parameter_utils import parameter_config
from utils.print_utils import print_header, print_run_info, header
from utils.gen_crystal_WyckPos import GenCrystalWyckPos, get_wyckoffs_strings, wp_alphabet
from utils.gen_crystal_WyckPos_old import get_all_wyckoff_combination
from utils.compound_utils import get_single_atom_energy
from predict_structure import PredictStructure

header()


class PredictStructureMC(PredictStructure):
    def __init__(self, input_file_path='dream.in', input_config=None, max_sample_count_per_ele_comb=1, **kwargs):
        self.input_config = ReadInput(input_file_path=input_file_path, input_config=input_config)
        self.single_atom_energy = get_single_atom_energy()

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

        # self.output_path = os.path.join(self.input_config.output_path, 'results')
        self.output_path = self.input_config.output_path
        self.structures_path = os.path.join(self.output_path, 'structures')
        self.step_result_file = os.path.join(self.output_path, 'step_result.csv')
        self.is_mpi_run = True
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

        self.some_settings_before_run(**kwargs)
        self.max_sample_count_per_ele_comb = max_sample_count_per_ele_comb
        self.elements = self.input_config.elements

        if self.max_sample_count_per_ele_comb is None:
            rec_count = 1
        else:
            rec_count = 4
        self.structure_number = 0
        for _ in range(rec_count):
            self.step_number = 0
            self.start_time = 0
            try:
                self.find_stable_structure()
            except:
                break

    def create_save_results_data_file(self):
        with open(self.step_result_file, 'w+') as f:
            f.write("number,step,compound,target,sg_number,child_cell," + \
                    ','.join(['ele_%d' % (i + 1) for i in range(self.composition_count)]) + \
                    ',' + \
                    ','.join(['ele_count_%d' % (i + 1) for i in range(self.composition_count)]) + \
                    ",time\n")

    def struc_parameter_config(self):
        elements = {}
        elements_count = {}
        for i in range(self.composition_count):
            elements.update(parameter_config('ele' + str(i), [0, len(self.elements[i])]))
            elements_count.update(parameter_config('ele_c' + str(i), [0, len(self.input_config.elements_count[i])]))
        a = parameter_config('a', self.input_config.lattice_a)
        b = parameter_config('b', self.input_config.lattice_b)
        c = parameter_config('c', self.input_config.lattice_c)
        alpha = parameter_config('alpha', self.input_config.lattice_alpha)
        beta = parameter_config('beta', self.input_config.lattice_beta)
        gamma = parameter_config('gamma', self.input_config.lattice_gamma)
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
            wp_sites = parameter_config('wp_index', [0, self.max_comb_count])

        parameter = {
            **elements, **elements_count,
            **a, **b, **c, **alpha, **beta, **gamma,
            **sg_index, **child_cell_index,
            **wp_sites,
            **atom_pos}

        return parameter

    def save_data_for_successful_step(self, target_property, struc_param: dict, struc: Structure, child_cell, **kwargs):
        struc.make_supercell(child_cell['expand_param'])
        formula = struc.formula.replace(' ', '')

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

    def get_structure_properties(self, struc_param: dict, **kwargs):
        self.step_number += 1

        # if (self.is_screen > 0) and (self.structure_number >= self.max_sample_count_per_ele_comb):
        if (self.max_sample_count_per_ele_comb is not None) and (self.structure_number >= self.max_sample_count_per_ele_comb):
            raise Exception('done!')

        try:
            struc, child_cell = self.convert_structure_from_struc_param(struc_param)
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

            target_property = self.properties_process(target_property, struc_param, target_struc)
            self.structure_number += 1
            self.save_data_for_successful_step(target_property, struc_param, target_struc, child_cell)

        except Exception as e:
            # print(e)
            # print(e.__traceback__.tb_frame.f_globals["__file__"])  # 发生异常所在的文件
            # print(e.__traceback__.tb_lineno)  # 发生异常所在的行数
            target_property = 999

        return target_property

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


def get_config(_eles, _out_path='./results/', _algorithm='rand', _max_step='5000', _wyck_pos_gen='1', _space_group='[1-230]'):
    config = configparser.ConfigParser()
    config.add_section('BASE')
    config.set('BASE', 'atom_element', _eles)
    config.set('BASE', 'atom_count', '[1-5] [1-5]')
    config.set('BASE', 'use_children_cell', 'True')
    config.set('BASE', 'min_atomic_dist_limit', '-0.6')
    config.set('BASE', 'volume_limit', '[0, 0]')
    config.set('BASE', 'max_vacuum_limit', '5.0')
    config.set('BASE', 'output_path', _out_path)

    config.add_section('CALCULATOR')
    config.set('CALCULATOR', 'calculator', 'm3gnet')
    config.set('CALCULATOR', 'calculator_path', '')
    config.set('CALCULATOR', 'use_calculator_relax', 'True')
    config.set('CALCULATOR', 'use_keep_symmetry', 'False')
    config.set('CALCULATOR', 'symprec', '0.001')
    config.set('CALCULATOR', 'use_gpu', 'False')

    config.add_section('OPTIMIZER')
    config.set('OPTIMIZER', 'algorithm', _algorithm)
    config.set('OPTIMIZER', 'n_init', '50')
    config.set('OPTIMIZER', 'max_step', _max_step)
    config.set('OPTIMIZER', 'rand_seed', '-1')
    config.set('OPTIMIZER', 'use_resume', 'False')
    config.set('OPTIMIZER', 'n_mpi', '1')
    config.set('OPTIMIZER', 'storage', '')

    config.add_section('LATTICE')
    config.set('LATTICE', 'space_group', _space_group)
    # config.set('LATTICE', 'use_flexible_site', 'False')
    config.set('LATTICE', 'wyck_pos_gen', _wyck_pos_gen)
    config.set('LATTICE', 'max_wyck_pos_count', '200000')
    config.set('LATTICE', 'lattice_a', '[%f-%f]' % (2.0, 20.0))
    config.set('LATTICE', 'lattice_b', '[%f-%f]' % (2.0, 20.0))
    config.set('LATTICE', 'lattice_c', '[%f-%f]' % (2.0, 20.0))
    config.set('LATTICE', 'lattice_alpha', '[%f-%f]' % (20.0, 160.0))
    config.set('LATTICE', 'lattice_beta', '[%f-%f]' % (20.0, 160.0))
    config.set('LATTICE', 'lattice_gamma', '[%f-%f]' % (20.0, 160.0))

    return config


def with_timeout(timeout):
    def decorator(decorated):
        @functools.wraps(decorated)
        def inner(*args, **kwargs):
            pool = ThreadPool(1)
            async_result = pool.apply_async(decorated, args, kwargs)
            try:
                return async_result.get(timeout)
            finally:
                pool.close()

        return inner

    return decorator


# @with_timeout(600)
def my_task(_config, max_sample_count_per_ele_comb=1):
    PredictStructureMC(None, _config, max_sample_count_per_ele_comb=max_sample_count_per_ele_comb)


# @with_timeout(6000)
def my_task2(_config, max_sample_count_per_ele_comb=None):
    PredictStructureMC(None, _config, max_sample_count_per_ele_comb=max_sample_count_per_ele_comb)


if __name__ == '__main__':

    elements = [['H', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'K', 'Ca', 'Sc', 'Ti', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Rb', 'Sr', 'Y',
                 'Zr', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Cs', 'Ba', 'Tl', 'Pb', 'Bi'],
                ['H', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'K', 'Ca', 'Sc', 'Ti', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Rb', 'Sr', 'Y',
                 'Zr', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Cs', 'Ba', 'Tl', 'Pb', 'Bi']]
    all_elements = [i for e in elements for i in e]
    all_elements = list(set(all_elements))
    elements_used_count = {i: 0 for i in all_elements}
    elements_used_count_range = [20, 25]
    # elements_used_count_range = [1, 2]
    elements_energy_list = {i: [] for i in all_elements}
    max_sample_count_per_ele_comb = 1

    composition_count = 2
    ele_comb_list = []
    while all_elements:
        random_items = random.choices(all_elements, k=composition_count)
        for i in set(random_items):
            elements_used_count[i] += 1
            if elements_used_count[i] >= elements_used_count_range[1] and i in all_elements:
                all_elements.remove(i)

        ele_comb_list.append(random_items)

        if min(elements_used_count.values()) >= elements_used_count_range[0]:
            break
    print(ele_comb_list, len(ele_comb_list))

    output_path = './results/1_eles_screen'
    num_process = 2
    config_list = []
    for ecl in ele_comb_list:
        ic = get_config(_eles='[%d] [%d]' % (Element(ecl[0]).Z, Element(ecl[1]).Z),
                        _out_path=output_path)
        config_list.append(ic)
    Parallel(n_jobs=num_process)(delayed(my_task)(ic) for ic in config_list)

    step_result = pd.read_csv(os.path.join(output_path, 'step_result.csv'))
    for dd in step_result[['ele_1', 'ele_2', 'target']].values:
        for d in set(dd[:2]):
            elements_energy_list[d].append(dd[2])

    elements_energy_norms = []
    for k, v in elements_energy_list.items():
        mu = np.average(v)
        sigma = np.std(v)
        p = norm.cdf(-7.9, loc=mu, scale=sigma)
        elements_energy_norms.append([k, mu, sigma, p, v])
    elements_energy_norms.sort(key=lambda x: x[3], reverse=True)
    save_data_csv(output_path, 'ele_prob.csv',
                  data=elements_energy_norms,
                  header=['elements', 'mu', 'sigma', 'prob(> -7.9)', 'target'])

    #########################################################################################

    element_comb_energy_list = {}
    all_elements = [i[0] for i in elements_energy_norms[:10]]
    print(all_elements)
    elements_used_count = {i: 0 for i in all_elements}
    elements_used_count_range = [40, 45]
    # elements_used_count_range = [1, 2]
    max_sample_count_per_ele_comb = 1

    ele_comb_list = []
    while all_elements:
        random_items = random.choices(all_elements, k=composition_count)
        for i in set(random_items):
            elements_used_count[i] += 1
            if elements_used_count[i] >= elements_used_count_range[1] and i in all_elements:
                all_elements.remove(i)

        ele_comb_list.append(random_items)

        if min(elements_used_count.values()) >= elements_used_count_range[0]:
            break

    output_path = './results/2_comb_screen'
    num_process = 2
    config_list = []
    for ecl in ele_comb_list:
        ic = get_config(_eles='[%d] [%d]' % (Element(ecl[0]).Z, Element(ecl[1]).Z),
                        _out_path=output_path)
        config_list.append(ic)
    Parallel(n_jobs=num_process)(delayed(my_task)(ic) for ic in config_list)

    step_result = pd.read_csv(os.path.join(output_path, 'step_result.csv'))
    for dd in step_result[['ele_1', 'ele_2', 'target']].values:
        e_comb = [dd[0], dd[1]]
        e_comb.sort()
        e_comb_key = '_'.join(e_comb)
        element_comb_energy_list[e_comb_key] = element_comb_energy_list.get(e_comb_key, [])
        element_comb_energy_list[e_comb_key].append(dd[2])

    element_comb_energy_list_top = sorted(element_comb_energy_list.items(), key=lambda x: min(x[1]))

    #########################################################################################

    count = 10
    element_comb_energy_list_top = element_comb_energy_list_top[:count]
    print(element_comb_energy_list_top)
    num_process = 2
    config_list = []
    for i, (k, v) in enumerate(element_comb_energy_list_top):
        output_path = os.path.join('./results/3_comp_pred', '%d_%s' % (i, k))
        ele = k.split('_')
        ic = get_config(_eles='[%d] [%d]' % (Element(ele[0]).Z, Element(ele[1]).Z),
                        _out_path=output_path,
                        _algorithm='tpe',
                        _max_step='5000',
                        _wyck_pos_gen='3',
                        _space_group='[1-230]')
        config_list.append(ic)
    Parallel(n_jobs=num_process)(delayed(my_task2)(ic, None) for ic in config_list)
