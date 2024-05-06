# -*- coding: utf-8 -*-
# ========================================================================
#  2023/6/15 20:31
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
import joblib
import sys
d3ream_path = "../../../d3ream"  # The path of d3ream project
sys.path.append(d3ream_path)

import configparser


def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_pred_class(_target='binding_energy'):
    if _target == 'binding_energy':
        from predict_structure_mpi_by_binding_energy import FindMaxBindingEnergyMaterialMPI
        return FindMaxBindingEnergyMaterialMPI
    elif _target == 'bulk_modulus':
        from predict_structure_mpi_by_bulk_modulus import FindMaxBulkModulusMPI
        return FindMaxBulkModulusMPI
    else:
        raise RuntimeError('Target class not exist!')


def get_config(_output_path='.', _storage=None, **kwargs):
    _config = configparser.ConfigParser()
    _config.add_section('BASE')
    _config.set('BASE', 'atom_element', '[1,3-9,11-17,19-22,29-35,37-40,47-53,55-56,81-83] [1,3-9,11-17,19-22,29-35,37-40,47-53,55-56,81-83]')
    _config.set('BASE', 'atom_count', '[1-5] [1-5]')
    _config.set('BASE', 'use_children_cell', 'True')
    _config.set('BASE', 'min_atomic_dist_limit', '-0.6')
    _config.set('BASE', 'volume_limit', '[0, 0]')
    _config.set('BASE', 'max_vacuum_limit', '7.0')
    _config.set('BASE', 'output_path', _output_path)

    _config.add_section('CALCULATOR')
    _config.set('CALCULATOR', 'calculator', 'm3gnet')
    _config.set('CALCULATOR', 'calculator_path', os.path.join(d3ream_path, 'calculators/m3gnet/origin_model/EFS2021'))
    _config.set('CALCULATOR', 'use_calculator_relax', 'True')
    _config.set('CALCULATOR', 'use_keep_symmetry', 'False')
    _config.set('CALCULATOR', 'symprec', '0.001')
    _config.set('CALCULATOR', 'use_gpu', 'False')

    _config.add_section('OPTIMIZER')
    _config.set('OPTIMIZER', 'algorithm', "tpe2")
    _config.set('OPTIMIZER', 'n_init', '30')
    _config.set('OPTIMIZER', 'max_step', '30000')
    _config.set('OPTIMIZER', 'rand_seed', '-1')
    _config.set('OPTIMIZER', 'n_mpi', '1')
    _config.set('OPTIMIZER', 'storage', _storage)

    _config.add_section('LATTICE')
    _config.set('LATTICE', 'space_group', '[1-230]')
    _config.set('LATTICE', 'wyck_pos_gen', '3')
    _config.set('LATTICE', 'max_wyck_pos_count', '200000')
    _config.set('LATTICE', 'lattice_a', '[2.0-30.0]')
    _config.set('LATTICE', 'lattice_b', '[2.0-30.0]')
    _config.set('LATTICE', 'lattice_c', '[2.0-30.0]')
    _config.set('LATTICE', 'lattice_alpha', '[20.0-160.0]')
    _config.set('LATTICE', 'lattice_beta', '[20.0-160.0]')
    _config.set('LATTICE', 'lattice_gamma', '[20.0-160.0]')

    return _config


if __name__ == '__main__':
    
    target = 'binding_energy'  # Search high cohesive energy materials by UPot-BO
    # target = 'bulk_modulus'  # Search high bulk modulus materials bu UPot-BO

    target_class = get_pred_class(target)

    _output_path = './%s' % target

    output_path = os.path.join(_output_path, 'results')
    structures_path = os.path.join(output_path, 'structures')
    check_path(output_path)
    check_path(structures_path)
    
    ####################################################
    #    running on a single thread                    #
    ####################################################
    config = get_config(_output_path, 'test')
    target_class(None, config, is_mpi_run=False)

    ########################################################
    # running on a multi-thread, recommend to run on linux #
    ########################################################

    # storage = "sqlite:///test.db"
    # config = get_config(_output_path, storage)
    # study_name = "test"
    # os.system('''optuna create-study --study-name "%s" --storage "%s" ''' % (study_name, storage))
    #
    # n_mpi = 2
    # joblib.Parallel(n_jobs=n_mpi, backend='multiprocessing')(
    #     joblib.delayed(target_class)(None, config, study_name)
    #     for _ in range(n_mpi)
    # )
