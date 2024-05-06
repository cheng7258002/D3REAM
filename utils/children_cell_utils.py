# -*- coding: utf-8 -*-
# ========================================================================
#  2022/5/25 23:31
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


from typing import List, Dict

import numpy as np

from utils.print_utils import print_run_info
# from utils.wyckoff_position.get_wyckoff_position import get_all_wyckoff_combination


def is_integer(number):
    if int(number) == number:
        return True
    else:
        return False


def crack_number(n):
    result = []

    start = int(np.power(n, 1 / 3))
    for i in range(start, n + 1):
        if n % i == 0:
            n = int(n / i)
            result.append(i)
            break

    start = int(np.power(n, 1 / 2))
    for i in range(start, n + 1):
        if n % i == 0:
            n = int(n / i)
            result.append(i)
            result.append(n)
            break

    return result


def get_children_cell(atoms: List, is_use_children_cell=True):
    if not is_use_children_cell:
        return {1: {'atoms': atoms, 'expand_param': [1, 1, 1]}}

    atoms = np.array(atoms)

    cells = {}
    for i in range(1, np.min(atoms) + 1):
        # print(i, atoms%i, atoms/i)
        # c = atoms/i

        if np.all(atoms % i == 0):
            ca = [int(x) for x in atoms / i]
            cep = crack_number(i)
            cells[i] = {'atoms': ca, 'expand_param': cep}

        # if np.any(c == 2):
        #     break

    return cells


def get_children_cell_2(atoms: List, is_use_children_cell=True):
    if not is_use_children_cell:
        return [{'child_cell': 1, 'atoms': atoms, 'expand_param': [1, 1, 1]}]

    atoms = np.array(atoms)

    cells = []
    for i in range(1, np.min(atoms) + 1):
        # print(i, atoms%i, atoms/i)
        # c = atoms/i

        if np.all(atoms % i == 0):
            ca = [int(x) for x in atoms / i]
            cep = crack_number(i)
            cells.append({'child_cell': i, 'atoms': ca, 'expand_param': cep})

        # if np.any(c == 2):
        #     break

    return cells


def get_all_children_cell(atoms_list: List):
    all_cells = []
    max_cell_count = 0
    for al in atoms_list:
        cells = get_children_cell(al)
        all_cells.append(cells)
        max_cell_count = max(max_cell_count, len(cells))
    return all_cells, max_cell_count


def get_all_children_cell_2(atoms_list_dict: Dict, is_use_children_cell=True):
    all_cells_dict = {}
    max_cell_count = 0
    for k, v in atoms_list_dict.items():
        atoms_list = v
        all_cells = []
        for al in atoms_list:
            cells = get_children_cell(al, is_use_children_cell=is_use_children_cell)
            all_cells.append(cells)
            max_cell_count = max(max_cell_count, len(cells))
        all_cells_dict[k] = all_cells
    return all_cells_dict, max_cell_count


# def get_children_wyckoff_combination(children_cell: dict, space_group):
#     children_wyckoffs_dict = {}
#     max_wyckoffs_count = 0
#
#     for k, v in children_cell.items():
#         _wyckoffs_dict, _max_wyckoffs_count = get_all_wyckoff_combination(space_group, v['atoms'])
#         children_wyckoffs_dict[k] = _wyckoffs_dict
#         max_wyckoffs_count = max(max_wyckoffs_count, _max_wyckoffs_count)
#
#     return children_wyckoffs_dict, max_wyckoffs_count


# @print_run_info("Get the Wyckoff position combinations")
# def get_all_children_wyckoff_combination(children_cell_list: List, space_group):
#     all_children_wyckoffs_dict = []
#     max_wyckoffs_count = 0
#     for ccl in children_cell_list:
#         _children_wyckoffs_dict, _max_wyckoffs_count = get_children_wyckoff_combination(ccl, space_group)
#         all_children_wyckoffs_dict.append(_children_wyckoffs_dict)
#         max_wyckoffs_count = max(max_wyckoffs_count, _max_wyckoffs_count)
#     return all_children_wyckoffs_dict, max_wyckoffs_count


# @print_run_info("Get the Wyckoff position combinations")
# def get_all_children_wyckoff_combination_2(children_cell_list_dict: Dict, space_group):
#     all_children_wyckoffs_dict_dict = {}
#     max_wyckoffs_count = 0
#
#     for k, v in children_cell_list_dict.items():
#         children_cell_list = v
#         all_children_wyckoffs_dict = []
#         for ccl in children_cell_list:
#             _children_wyckoffs_dict, _max_wyckoffs_count = get_children_wyckoff_combination(ccl, space_group)
#             all_children_wyckoffs_dict.append(_children_wyckoffs_dict)
#             max_wyckoffs_count = max(max_wyckoffs_count, _max_wyckoffs_count)
#         all_children_wyckoffs_dict_dict[k] = all_children_wyckoffs_dict
#
#     return all_children_wyckoffs_dict_dict, max_wyckoffs_count


if __name__ == '__main__':
    cc = get_children_cell([5, 10])
    print(cc)
    print(tuple(cc.keys()))
