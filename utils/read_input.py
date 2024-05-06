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
# Read input parameters from dream.in
#
# ========================================================================


import os
import ast
import configparser

from utils.print_utils import print_run_info
from utils.compound_utils import compound_split, get_elements_info
from utils.parameter_utils import range_parameter_to_list


class ReadInput:
    @print_run_info("Read input file")
    def __init__(self, input_file_path, input_config=None):
        if input_file_path:
            if not os.path.isfile(input_file_path):
                raise IOError("Could not find `dream.in` file!")

            config = configparser.RawConfigParser()
            config.read(input_file_path, encoding='utf-8')
        elif input_file_path is None and input_config:
            config = input_config
        else:
            raise RuntimeError('Please input some thing!')

        self.elements_info = get_elements_info()

        # region ## BASE ##
        self.compound = config.get('BASE', 'compound', fallback=None)
        self.elements = config.get('BASE', 'atom_element', fallback=None)
        self.elements_count = config.get('BASE', 'atom_count', fallback=None)
        # if not (self.compound or (self.elements and self.elements_count)):
        #     raise RuntimeError('Parameter `compound/atom_element/atom_count` error! Please check!')

        if self.compound:
            self.compound = self.compound.replace(' ', '')
            elements, elements_count = compound_split(self.compound)
            self.elements = [[ele, ] for ele in elements]
            self.elements_count = [[ele_c, ] for ele_c in elements_count]
        elif self.elements and self.elements_count:
            self.elements = range_parameter_to_list(self.elements, ptype='int')
            self.elements = [[self.elements_info['name'][j-1] for j in i] for i in self.elements]
            self.elements_count = range_parameter_to_list(self.elements_count, ptype='int')
        else:
            raise RuntimeError('Parameter `compound/atom_element/atom_count` error! Please check!')

        self.is_use_children_cell = config.getboolean('BASE', 'use_children_cell', fallback=True)
        self.min_atomic_dist_limit = config.getfloat('BASE', 'min_atomic_dist_limit', fallback=0.4)
        self.volume_limit = ast.literal_eval(config.get('BASE', 'volume_limit', fallback='[0, 0]'))
        self.max_vacuum_limit = config.getfloat('BASE', 'max_vacuum_limit', fallback=7.0)
        self.output_path = config.get('BASE', 'output_path', fallback='.')
        # endregion

        # region ## CALCULATOR ##
        self.calculator = config.get('CALCULATOR', 'calculator', fallback='m3gnet').lower()
        self.calculator_path = config.get('CALCULATOR', 'calculator_path', fallback=None)
        self.is_use_calculator_relax = config.getboolean('CALCULATOR', 'use_calculator_relax', fallback=True)
        self.is_use_keep_symmetry = config.getboolean('CALCULATOR', 'use_keep_symmetry', fallback=True)
        self.symprec = config.getfloat('CALCULATOR', 'symprec', fallback=0.001)
        self.is_use_gpu = config.getboolean('CALCULATOR', 'use_gpu', fallback=False)
        # endregion

        # region ## OPTIMIZER ##
        self.algorithm = config.get('OPTIMIZER', 'algorithm', fallback='tpe').lower()
        self.n_init = config.getint('OPTIMIZER', 'n_init', fallback=100)
        self.max_step = config.getint('OPTIMIZER', 'max_step', fallback=2000)
        self.rand_seed = config.getint('OPTIMIZER', 'rand_seed', fallback=100)
        self.is_use_resume = config.getboolean('OPTIMIZER', 'use_resume', fallback=False)
        self.n_mpi = config.getint('OPTIMIZER', 'n_mpi', fallback=1)
        self.storage = config.get('OPTIMIZER', 'storage', fallback='sqlite:///%s/results/results.db' % self.output_path)
        # endregion

        # region ## LATTICE ##
        _space_group = config.get('LATTICE', 'space_group', fallback='[1-1]')
        _lattice_a = config.get('LATTICE', 'lattice_a', fallback='[2-30]')
        _lattice_b = config.get('LATTICE', 'lattice_b', fallback='[2-30]')
        _lattice_c = config.get('LATTICE', 'lattice_c', fallback='[2-30]')
        _lattice_alpha = config.get('LATTICE', 'lattice_alpha', fallback='[20-160]')
        _lattice_beta = config.get('LATTICE', 'lattice_beta', fallback='[20-160]')
        _lattice_gamma = config.get('LATTICE', 'lattice_gamma', fallback='[20-160]')
        # self.lattice_precision = config.getfloat('LATTICE', 'lattice_precision', fallback=0.1)
        self.wyck_pos_gen = config.getint('LATTICE', 'wyck_pos_gen', fallback=1)
        if self.wyck_pos_gen == 1:
            self.max_wyck_pos_count = config.getint('LATTICE', 'max_wyck_pos_count', fallback=20e4)
        elif self.wyck_pos_gen == 2:
            self.max_wyck_pos_count = -1
        elif self.wyck_pos_gen == 3:
            self.is_use_flexible_site = True
        elif self.wyck_pos_gen == 4:
            self.is_use_flexible_site = False
        else:
            raise RuntimeError('Parameter `wyck_pos_gen` error! Please check!')

        self.space_group = range_parameter_to_list(_space_group, ptype='int')[0]
        # self.lattice_a = range_parameter_to_list(_lattice_a, step=self.lattice_precision)[0]
        # self.lattice_b = range_parameter_to_list(_lattice_b, step=self.lattice_precision)[0]
        # self.lattice_c = range_parameter_to_list(_lattice_c, step=self.lattice_precision)[0]
        # self.lattice_alpha = range_parameter_to_list(_lattice_alpha, step=self.lattice_precision)[0]
        # self.lattice_beta = range_parameter_to_list(_lattice_beta, step=self.lattice_precision)[0]
        # self.lattice_gamma = range_parameter_to_list(_lattice_gamma, step=self.lattice_precision)[0]
        self.lattice_a = ast.literal_eval(_lattice_a.replace('-', ','))
        self.lattice_b = ast.literal_eval(_lattice_b.replace('-', ','))
        self.lattice_c = ast.literal_eval(_lattice_c.replace('-', ','))
        self.lattice_alpha = ast.literal_eval(_lattice_alpha.replace('-', ','))
        self.lattice_beta = ast.literal_eval(_lattice_beta.replace('-', ','))
        self.lattice_gamma = ast.literal_eval(_lattice_gamma.replace('-', ','))
        # endregion

        # region ## Print parameter ##
        print('  Atom elements: %s ' % self.elements)
        print('  Atom count: %s ' % self.elements_count)
        print('  Calculator: %s    Use relax: %s' % (self.calculator, self.is_use_calculator_relax))
        print('  Optimizer: %s    Max step: %d' % (self.algorithm, self.max_step))
        print('  Lattice:  a:', _lattice_a, '  b:', _lattice_b, '  c:', _lattice_c)
        print('            alpha:', _lattice_alpha, '  beta:', _lattice_beta, '  gamma:', _lattice_gamma)
        # print('            Precision: %s      Space group: %s' % (self.lattice_precision, self.space_group))
        print('            Space group:', _space_group, '  WyckPos method:', self.wyck_pos_gen)
        # endregion

    # @property
    # def calculator(self):
    #     return self._calculator.lower()
    #
    # @calculator.setter
    # def calculator(self, calculator):
    #     self._calculator = str(calculator)
    #
    # @property
    # def calculator_path(self):
    #     return self._calculator_path
    #
    # @calculator_path.setter
    # def calculator_path(self, calculator_path):
    #     self._calculator_path = calculator_path
    #
    # @property
    # def is_use_gpu(self):
    #     return self._is_use_gpu
    #
    # @is_use_gpu.setter
    # def is_use_gpu(self, is_use_gpu):
    #     self._is_use_gpu = is_use_gpu
    #
    # @property
    # def is_use_calculator_relax(self):
    #     return self._is_use_calculator_relax
    #
    # @is_use_calculator_relax.setter
    # def is_use_calculator_relax(self, is_use_calculator_relax):
    #     self._is_use_calculator_relax = is_use_calculator_relax
    #
    # @property
    # def is_use_keep_symmetry(self):
    #     return self._is_use_keep_symmetry
    #
    # @is_use_keep_symmetry.setter
    # def is_use_keep_symmetry(self, is_use_keep_symmetry):
    #     self._is_use_keep_symmetry = is_use_keep_symmetry
    #
    # @property
    # def output_path(self):
    #     return self._output_path
    #
    # @output_path.setter
    # def output_path(self, output_path):
    #     self._output_path = output_path
    #
    # @property
    # def compound(self):
    #     return self._compound
    #
    # @compound.setter
    # def compound(self, compound):
    #     self._compound = compound
    #
    # @property
    # def elements(self):
    #     return self._elements
    #
    # @elements.setter
    # def elements(self, elements):
    #     self._elements = elements
    #
    # @property
    # def elements_count(self):
    #     return self._elements_count
    #
    # @elements_count.setter
    # def elements_count(self, elements_count):
    #     self._elements_count = elements_count
    #
    # @property
    # def elements_info(self):
    #     return self._elements_info
    #
    # @elements_info.setter
    # def elements_info(self, elements_info):
    #     self._elements_info = elements_info
    #
    # @property
    # def is_use_children_cell(self):
    #     return self._is_use_children_cell
    #
    # @is_use_children_cell.setter
    # def is_use_children_cell(self, is_use_children_cell):
    #     self._is_use_children_cell = is_use_children_cell
    #
    # @property
    # def space_group(self):
    #     return self._space_group
    #
    # @space_group.setter
    # def space_group(self, space_group):
    #     self._space_group = space_group
    #
    # @property
    # def is_use_flexible_site(self):
    #     return self._is_use_flexible_site
    #
    # @is_use_flexible_site.setter
    # def is_use_flexible_site(self, is_use_flexible_site):
    #     self._is_use_flexible_site = is_use_flexible_site
    #
    # @property
    # def lattice_a(self):
    #     return self._lattice_a
    #
    # @lattice_a.setter
    # def lattice_a(self, lattice_a):
    #     self._lattice_a = lattice_a
    #
    # @property
    # def lattice_b(self):
    #     return self._lattice_b
    #
    # @lattice_b.setter
    # def lattice_b(self, lattice_b):
    #     self._lattice_b = lattice_b
    #
    # @property
    # def lattice_c(self):
    #     return self._lattice_c
    #
    # @lattice_c.setter
    # def lattice_c(self, lattice_c):
    #     self._lattice_c = lattice_c
    #
    # @property
    # def lattice_alpha(self):
    #     return self._lattice_alpha
    #
    # @lattice_alpha.setter
    # def lattice_alpha(self, lattice_alpha):
    #     self._lattice_alpha = lattice_alpha
    #
    # @property
    # def lattice_beta(self):
    #     return self._lattice_beta
    #
    # @lattice_beta.setter
    # def lattice_beta(self, lattice_beta):
    #     self._lattice_beta = lattice_beta
    #
    # @property
    # def lattice_gamma(self):
    #     return self._lattice_gamma
    #
    # @lattice_gamma.setter
    # def lattice_gamma(self, lattice_gamma):
    #     self._lattice_gamma = lattice_gamma
    #
    # @property
    # def algorithm(self):
    #     return self._algorithm.lower()
    #
    # @algorithm.setter
    # def algorithm(self, algorithm):
    #     self._algorithm = str(algorithm)
    #
    # @property
    # def n_init(self):
    #     return self._n_init
    #
    # @n_init.setter
    # def n_init(self, n_init):
    #     self._n_init = n_init
    #
    # @property
    # def max_step(self):
    #     return self._max_step
    #
    # @max_step.setter
    # def max_step(self, max_step):
    #     self._max_step = max_step
    #
    # @property
    # def rand_seed(self):
    #     return self._rand_seed
    #
    # @rand_seed.setter
    # def rand_seed(self, rand_seed):
    #     self._rand_seed = rand_seed
    #
    # @property
    # def n_mpi(self):
    #     return self._n_mpi
    #
    # @n_mpi.setter
    # def n_mpi(self, n_mpi):
    #     self._n_mpi = n_mpi
    #
    # @property
    # def storage(self):
    #     return self._storage
    #
    # @storage.setter
    # def storage(self, storage):
    #     self._storage = storage
