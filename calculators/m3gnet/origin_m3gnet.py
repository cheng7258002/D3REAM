# -*- coding: utf-8 -*-
# ========================================================================
#  2022-05-19  09:22:07
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
import logging
logger = logging.getLogger('m3gnet.graph._converters')
logger.setLevel(logging.ERROR)

from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core import Structure
from m3gnet.models import M3GNet, Potential, M3GNetCalculator

from calculators._calculator_base import CalculatorBase
from calculators.m3gnet.relax_plus import RelaxerPlus


class OrinM3GNET(CalculatorBase):
    def __init__(self, model_path=None, use_relax=False, **kwargs):
        # m3gnet = M3GNet.from_dir(dirname='./EFS2021')
        if (model_path is None) or (not os.path.exists(model_path)):
            model_path = os.path.join(os.path.dirname(__file__), 'origin_model/EFS2021')
        self.m3gnet = M3GNet.from_dir(dirname=model_path)
        self.potential = Potential(model=self.m3gnet)
        self.calcu = M3GNetCalculator(potential=self.potential)
        self.use_relax = use_relax
        # if self.use_relax:
        #     self.get_target_method = self.get_target_from_relax
        # else:
        #     self.get_target_method = self.get_target_from_predict

    def get_target(self, struc: Structure, **kwargs):
        if self.use_relax:
            return self.get_target_from_relax(struc, **kwargs)
        else:
            return self.get_target_from_relax(struc, steps=1, **kwargs)[0]

    def get_target_from_predict(self, struc: Structure, **kwargs):
        target = self.predict(struc)['energy'].reshape(-1)[0]
        target /= struc.num_sites
        return target

    def get_target_from_relax(self, struc: Structure, steps=100, **kwargs):
        keep_symmetry = kwargs.get('keep_symmetry', False)
        symprec = kwargs.get('symprec', False)
        # use_local_optimizer = kwargs.get('use_local_optimizer', True)

        relaxer = RelaxerPlus(potential=self.potential)
        relax_results = relaxer.relax(struc, steps=steps, logfile=None, keep_symmetry=keep_symmetry, symprec=symprec)
        final_structure = relax_results['final_structure']
        target = relax_results['trajectory'].energies[-1].reshape(-1)[0]
        target /= struc.num_sites
        return target, final_structure

    def train(self):
        pass

    def predict(self, struc):
        atoms = AseAtomsAdaptor().get_atoms(struc)
        self.calcu.calculate(atoms=atoms)
        return self.calcu.results


if __name__ == '__main__':
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    # _struc = Structure.from_file(r"F:\d3ream\analysis\DFT_M3GNet_MEGNet\structures\Na3Zn_mp-1186099.cif")
    # _struc = Structure.from_file(r"F:\d3ream\analysis\DFT_M3GNet_MEGNet\structures\-5.872375_V10Si10_12226_20081.cif")
    # _struc = Structure.from_file(r"F:\d3ream\test\CaS_mp-1672_conventional_standard_666.cif")
    _struc = Structure.from_file(r"./Ti4C4.vasp")
    # print(_struc)
    # print(OrinM3GNET(r'F:\d3ream\calculators\m3gnet\origin_model\EFS2021').predict(_struc))
    # _target, _final_structure = OrinM3GNET(r'F:\d3ream\calculators\m3gnet\origin_model\EFS2021', use_relax=False).get_target(_struc, keep_symmetry=False)
    # _final_structure.to(fmt='cif',
    #                     filename='%s_%f.cif' % (_final_structure.formula.replace(' ', ''), _target),
    #                     symprec=0.001)
    # print(_target, _final_structure)
    # _final_structure.to(fmt='cif', filename='bbbbbbb.cif', symprec=0.001)
    model = OrinM3GNET(r'F:\d3ream\calculators\m3gnet\origin_model\EFS2021', use_relax=True)
    _target, _final_structure = model.get_target(_struc, keep_symmetry=False)
    # aaaa = model.predict(_struc)
    # print(aaaa)
    print(_target)
    _final_structure.to(fmt='poscar', filename='Ti4C4_pred_relax.vasp')

