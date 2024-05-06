# -*- coding: utf-8 -*-
# ========================================================================
#  2022/5/15 8:27
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

from ase.io.vasp import read_vasp_out
from pymatgen.core import Structure

from calculators.VASP.vasp_base import VASPBase


class VASPStatic(VASPBase):
    def __init__(self, sub_file_path, pred_result_path='.', is_save_result=False, **kwargs):
        super().__init__(sub_file_path, pred_result_path, is_save_result, **kwargs)

    def generate_incar(self, file_path):
        with open(file_path, 'w+') as f:
            f.write(
'''
Global Parameters
ISTART    =  0            (Read existing wavefunction; if there)
PREC      =  Normal       (Precision level)
LWAVE     = .F.        (Write WAVECAR or not)
LCHARG    = .F.        (Write CHGCAR or not)
KSPACING  =  0.3

Static Calculation
ISMEAR    =  0            (gaussian smearing method)
SIGMA     =  0.05         (please check the width of the smearing)
LORBIT    =  11           (PAW radii for projected DOS)
NELM      =  60           (Max electronic SCF steps)
EDIFF     =  1E-04        (SCF energy convergence; in eV)

'''
            )

    def get_target(self, struc: Structure, **kwargs):
        self.generate_input_files(struc, **kwargs)
        return_code = self.run_vasp(self.current_calc_path)

        if return_code:
            outcar = read_vasp_out(os.path.join(self.current_calc_path, 'OUTCAR'))
            energy = outcar.get_total_energy() / struc.num_sites
        else:
            energy = 999999

        return energy
