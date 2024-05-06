# -*- coding: utf-8 -*-
# ========================================================================
#  2021/04/27 上午 08:28
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
import re
import itertools
import collections
from typing import Dict, List, Tuple, Union

import numpy as np

from utils.file_utils import get_program_path, read_data_bin, read_data_csv


def get_elements_info():
    elements_info_path = os.path.join(get_program_path(),
                                      'data/elements_info.pkl')
    elements_info = read_data_bin(None, elements_info_path)

    return elements_info


def get_single_atom_energy():
    file_path = os.path.join(get_program_path(),
                             'data/single_atom_energy.csv')
    data = read_data_csv(None, file_path).values[1:]
    energy_dict = {}
    for d in data:
        energy_dict[d[0]] = [float(d[1]), float(d[2])]

    return energy_dict


def get_single_compound_energy():
    file_path = os.path.join(get_program_path(),
                             'data/single_compound_total_energy/data.csv')
    data = read_data_csv(None, file_path).values[1:]
    energy_dict = {}
    for d in data:
        ele = d[1].replace('2', '')
        energy_dict[ele] = float(d[3])

    return energy_dict


def compound_split(compound):
    """
        Split the compound into elements and corresponding count
        :param compound:
        :return:
    """
    temp_str = compound.replace(' ', '')
    pattern = re.compile(r'\d+')
    count = re.findall(pattern, temp_str)
    count = [int(x) for x in count]
    elements = re.split(pattern, temp_str)[:len(count)]

    return elements, count


def get_element_combinations(elements_list):
    '''
    get element combinations
    :param elements_list: from input
    :return: element combinations without same
    '''

    results = []

    all_combinations = itertools.product(*elements_list)
    tmp_combinations = []
    for ac in all_combinations:
        ac = sorted(ac)
        tmp_combinations.append(','.join(ac))
    tmp_combinations = set(tmp_combinations)
    for tc in tmp_combinations:
        results.append(collections.Counter(tc.split(',')))

    return results


def get_elements_and_count_from_optimizer(_elements: Dict, _count: Union[List, Tuple]):
    elements = []
    count = []

    i = 0
    for k, v in _elements.items():
        elements.append(k)
        c = int(sum(_count[i:i+v])/v)
        count.append(c)
        i = i+v

    return elements, count


def get_element_combinations_2(elements_list):
    '''
    get element combinations
    :param elements_list: from input
    :return: element combinations without same
    '''

    results = []

    all_combinations = itertools.product(*elements_list)
    tmp_combinations = []
    tmp_combinations_str = []
    for ac in all_combinations:
        ac = list(set(ac))
        ac_str = ','.join(sorted(ac))
        if ac_str not in tmp_combinations_str:
            tmp_combinations_str.append(ac_str)
            tmp_combinations.append(ac)
    # tmp_combinations = set(tmp_combinations)
    # for tc in tmp_combinations:
    #     results.append(tc.split(','))
    results = tmp_combinations

    return results


def get_element_combinations_3(elements_list):
    combinations = {}
    max_combination_count = 0

    results = get_element_combinations_2(elements_list)
    for res in results:
        i_res = len(res)
        combinations[i_res] = combinations.get(i_res, []) + [res]
    max_combination_count = max([len(c) for c in combinations.values()])

    return combinations, max_combination_count


def get_element_count_combinations(element_count_list):
    min_count = np.min(element_count_list)
    max_count = np.max(element_count_list)
    children_element_count_list = list(range(min_count, max_count+1))
    n_compositions = len(element_count_list)

    combinations = {}
    max_count = 0
    for i in range(1, n_compositions):
        tmp_element_count_list = [children_element_count_list, ]*i
        combinations[i] = list(itertools.product(*tmp_element_count_list))
        max_count = max(max_count, max([sum(c) for c in combinations[i]]))
    combinations[n_compositions] = list(itertools.product(*element_count_list))
    max_count = max(max_count, max([sum(c) for c in combinations[n_compositions]]))
    max_count = int(max_count)

    return combinations, max_count


def get_element_count_combinations_2(element_count_list):
    min_count = np.min(element_count_list)
    max_count = np.max(element_count_list)
    children_element_count_list = list(range(min_count, max_count+1))
    n_compositions = len(element_count_list)

    combinations = {}
    max_combination_count = 0
    max_atom_count = 0
    for i in range(1, n_compositions):
        tmp_element_count_list = [children_element_count_list, ]*i
        combinations[i] = list(itertools.product(*tmp_element_count_list))
        max_combination_count = max(max_combination_count, len(combinations[i]))
        max_atom_count = max(max_atom_count, max([sum(c) for c in combinations[i]]))

    combinations[n_compositions] = list(itertools.product(*element_count_list))
    max_combination_count = max(max_combination_count, len(combinations[n_compositions]))
    max_atom_count = max(max_atom_count, max([sum(c) for c in combinations[n_compositions]]))

    # for c in itertools.product(*element_count_list):
    #     c_tmp = []
    #     for c_i in c:
    #         if c_i not in c_tmp:
    #             c_tmp.append(c_i)
    #     len_c_tmp = len(c_tmp)
    #     combinations[len_c_tmp] = combinations.get(len_c_tmp, []) + [c_tmp]
    #     max_atom_count = max(max_atom_count, sum(c_tmp))
    # max_combination_count = max([len(c) for c in combinations.values()])

    return combinations, max_combination_count, max_atom_count


if __name__ == '__main__':
    # print(get_elements_info())
    # print(get_single_atom_energy())
    print(get_single_compound_energy())
