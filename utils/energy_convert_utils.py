# -*- coding: utf-8 -*-
# ========================================================================
#  2022/10/5 9:14
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

from utils.compound_utils import get_single_atom_energy


single_atom_energy = get_single_atom_energy()


def Et2Eb(Et, struc, **kwargs):
    # Eb = Et * struc.num_sites
    Eb = Et
    for p in struc.species:
        # sae = single_atom_energy[str(p)][0]  # no spin
        sae = single_atom_energy[str(p)][1]  # with spin
        Eb -= sae
    Eb /= struc.num_sites
    return Eb

