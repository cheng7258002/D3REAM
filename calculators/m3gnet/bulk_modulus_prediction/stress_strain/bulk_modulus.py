# -*- coding: utf-8 -*-
# ========================================================================
#  2022/8/4 9:09
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
import shutil
import logging

logger = logging.getLogger('m3gnet.graph._converters')
logger.setLevel(logging.ERROR)

import numpy as np
from ase.io import read, write
from ase.optimize import BFGS
from ase.constraints import ExpCellFilter
from ase.units import GPa
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from m3gnet.models import M3GNet, Potential, M3GNetCalculator

from calculators._calculator_base import CalculatorBase
from utils.file_utils import check_path

CWD = os.path.dirname(os.path.abspath(__file__))


class BulkModulusByM3GNetStressStrain(CalculatorBase):
    def __init__(self, calculator=None, model_path=None, is_rm_res_dir=True, **kwargs):
        if calculator is not None:
            self.calculator = calculator
        else:
            if model_path is None:
                model_path = os.path.join(CWD, '../../origin_model/EFS2021')
            self.m3gnet = M3GNet.from_dir(dirname=model_path)
            self.potential = Potential(model=self.m3gnet)
            self.calculator = M3GNetCalculator(potential=self.potential)

        self.is_rm_res_dir = is_rm_res_dir

    @staticmethod
    def relax(atoms, calc, logfile=None, pressure=0.0, maxstep=0.1, eps=None, max_step=None):
        atoms.set_calculator(calc)
        mask = [False, False, False, False, False, False]
        ucf = ExpCellFilter(atoms, scalar_pressure=pressure * GPa, mask=mask, constant_volume=True)
        gopt = BFGS(ucf, maxstep=maxstep, logfile=logfile)
        gopt.run(fmax=eps, steps=max_step)
        return atoms

    def calcu_energy(self, calcu_dir, **kwargs):
        outcar_file = os.path.join(calcu_dir, 'OUTCAR')
        if os.path.exists(outcar_file):
            os.remove(outcar_file)
        struct = read(os.path.join(calcu_dir, "POSCAR"))
        f_max = 0.001
        Relaxed_atoms = self.relax(atoms=struct, calc=self.calculator, logfile=outcar_file, eps=f_max, max_step=1000)
        # atom_force = Relaxed_atoms.get_forces()
        U_atom = Relaxed_atoms.get_potential_energy()[-1]
        text = f"  free  energy   TOTEN  = {U_atom} eV\n" + \
               "                 Voluntary context switches:"
        with open(outcar_file, 'a+') as f:
            f.write(text)

    def gen_vaspkit_in(self, in_type=1):
        text = f'{in_type}\n' + \
               '3D\n' + \
               '7\n' + \
               '-0.024 -0.016 -0.008 0.000 0.008 0.016 0.024\n'
               # '-0.005 -0.003 -0.001 0.000 0.001 0.003 0.005\n'
        with open(os.path.join(self.results_dir, 'VPKIT.in'), 'w+') as f:
            f.write(text)

    def calcu_flow(self, struct: Structure, **kwargs):
        struct.to(fmt='poscar', filename=os.path.join(self.results_dir, 'POSCAR'))
        self.gen_vaspkit_in(in_type=1)
        os.system('cd %s; vaspkit -task 201 ' % self.results_dir)
        for cij in os.listdir(self.results_dir):
            if 'C' not in cij:
                continue
            cij_dir = os.path.join(self.results_dir, cij)
            if not os.path.isdir(cij_dir):
                continue
            for s in os.listdir(cij_dir):
                if 'strain_' not in s:
                    continue
                s_dir = os.path.join(cij_dir, s)
                print(s_dir)
                self.calcu_energy(calcu_dir=s_dir)

        self.gen_vaspkit_in(in_type=2)
        os.system('cd %s; vaspkit -task 201 > BM_SS.log ' % self.results_dir)
        with open(os.path.join(self.results_dir, 'BM_SS.log'), 'r') as f:
            lines = f.readlines()
        if self.is_rm_res_dir:
            shutil.rmtree(self.results_dir)

        res_table_line_num = -1
        for i, line in enumerate(lines):
            if 'Mechanical Properties' in line and 'Voigt' in line and 'Reuss' in line and 'Hill' in line:
                res_table_line_num = i

            if 'This Structure is Mechanically Unstable' in line:
                raise 'This structure is mechanically unstable!'

        if res_table_line_num:
            line_list = lines[res_table_line_num+2].split()
            target = float(line_list[-2])
            return target

        # return 999
        raise 'No bulk modulus!'

    def get_target(self, struct: Structure, results_dir='./BM_SS_tmp', **kwargs):
        check_path(results_dir)
        self.results_dir = results_dir

        target = self.calcu_flow(struct=struct)
        return target


if __name__ == '__main__':
    _struc = Structure.from_file(r"F:\desktop\MouseWithoutBorders\Si20_20_-5.167386_1655_2744.cif")
    bm = BulkModulusByM3GNetStressStrain().get_target(struct=_struc)
    print(bm)
