# -*- coding: utf-8 -*-
# ========================================================================
#  2022/6/9 13:14
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
from mp_api import MPRester

from utils.file_utils import check_path

# Please set your MP API key
mp_api_key = ''
if not mp_api_key:
    try:
        from test.my_password import mp_api_key
    except:
        raise Exception('Please set your MP API key!')

with MPRester(mp_api_key) as mpr:
    docs = mpr.summary.search(
        num_elements=(1, 1),
        formation_energy=(0, 0),
        fields=['structure', 'material_id', 'formula_pretty', 'nsites', 'energy_per_atom', 'formation_energy_per_atom']
    )


save_path = 'single_compound_total_energy'
check_path(save_path)
save_structure_path = os.path.join(save_path, 'structures')
check_path(save_structure_path)

data_file = open(os.path.join(save_path, 'data.csv'), 'w+')
data_file.write('material_id,formula_pretty,nsites,energy_per_atom,formation_energy_per_atom\n')

for d in docs:
    material_id = str(d.material_id)
    formula_pretty = str(d.formula_pretty)
    nsites = str(d.nsites)
    energy_per_atom = str(d.energy_per_atom)
    formation_energy_per_atom = str(d.formation_energy_per_atom)
    data_file.write(','.join([material_id, formula_pretty, nsites, energy_per_atom, formation_energy_per_atom]) + '\n')

    structure = d.structure
    structure.to(fmt='cif', filename=os.path.join(save_structure_path, '%s.cif' % formula_pretty))

data_file.close()

