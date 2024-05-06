# -*- coding: utf-8 -*-
# ========================================================================
#  2022/5/31 8:12
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

from pymatgen.core import Structure
from pymatgen.io.vasp.sets import MPRelaxSet
from pymatgen.io.vasp.outputs import Oszicar

from calculators.VASP.vasp_base import VASPBase
from utils.file_utils import check_and_rename_path


class VASPRelax(VASPBase):
    def generate_incar(self, file_path):
        pass

    def generate_input_files(self, struc: Structure, **kwargs):
        if self.is_save_result:
            self.count += 1
            self.current_calc_path = os.path.join(self.vasp_result_path, str(self.count))
            check_and_rename_path(self.current_calc_path)

        vasp_input_set = MPRelaxSet(struc,
                                    force_gamma=True,
                                    user_potcar_functional='PBE',
                                    user_incar_settings={"LCHARG": False},
                                    )
        struc.to(filename=os.path.join(self.current_calc_path, 'POSCAR'), fmt='poscar')
        vasp_input_set.incar.write_file(os.path.join(self.current_calc_path, 'INCAR'))
        vasp_input_set.kpoints.write_file(os.path.join(self.current_calc_path, 'KPOINTS'))
        vasp_input_set.potcar.write_file(os.path.join(self.current_calc_path, 'POTCAR'))

    def get_target(self, struc: Structure, **kwargs):
        self.generate_input_files(struc)
        code = self.run_vasp(vasp_cores=6)
        # if code == 0:
        #     raise ''
        end_struct = Structure.from_file(os.path.join(self.current_calc_path, 'CONTCAR'))
        end_energy = Oszicar(os.path.join(self.current_calc_path, 'OSZICAR')).final_energy / end_struct.num_sites

        return end_energy, end_struct


