# -*- coding: utf-8 -*-
# ========================================================================
#  2022/6/24 15:54
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

from ase import Atoms
from ase.constraints import ExpCellFilter
from pymatgen.core import Molecule, Structure
from m3gnet.models import Potential, Relaxer
from m3gnet.models._dynamics import TrajectoryObserver


class RelaxerPlus(Relaxer):
    def relax(self, atoms: Atoms, fmax: float = 0.1, steps: int = 500, traj_file: str = None, interval=1, keep_symmetry=False, symprec=0.001, **kwargs):
        if isinstance(atoms, (Structure, Molecule)):
            atoms = self.ase_adaptor.get_atoms(atoms)

        if keep_symmetry:
            from ase.spacegroup.symmetrize import FixSymmetry
            fs = FixSymmetry(atoms=atoms, symprec=symprec)
            atoms.set_constraint([fs, ])

        atoms.set_calculator(self.calculator)
        obs = TrajectoryObserver(atoms)
        if self.relax_cell:
            atoms = ExpCellFilter(atoms)
        optimizer = self.opt_class(atoms, **kwargs)
        optimizer.attach(obs, interval=interval)
        optimizer.run(fmax=fmax, steps=steps)
        obs()
        if traj_file is not None:
            obs.save(traj_file)
        if isinstance(atoms, ExpCellFilter):
            atoms = atoms.atoms
        return {
            "final_structure": self.ase_adaptor.get_structure(atoms),
            "trajectory": obs,
        }


if __name__ == '__main__':
    pass
