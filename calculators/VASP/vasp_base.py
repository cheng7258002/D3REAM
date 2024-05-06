# -*- coding: utf-8 -*-
# ========================================================================
#  2022/5/14 22:42
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
import time
import shutil
from abc import abstractmethod
from multiprocessing import Process

from pymatgen.core import Structure

from calculators._calculator_base import CalculatorBase
from utils.file_utils import check_and_rename_path, get_program_path


class VASPBase(CalculatorBase):
    def __init__(self, sub_file_path, pred_result_path='.', is_save_result=False, **kwargs):
        # if not os.path.isfile(sub_file_path):
        #     raise Exception('VASP submit script file is not exist!')

        if is_save_result:
            vasp_result_path_name = 'VASP_results'
        else:
            vasp_result_path_name = 'VASP_tmp'

        self.vasp_result_path = os.path.join(pred_result_path, vasp_result_path_name)
        check_and_rename_path(self.vasp_result_path)

        self.vasp_input_file_path = kwargs.get('vasp_input_file_path', '.')

        self.sub_file_path = sub_file_path
        self.is_save_result = is_save_result
        self.count = 0
        self.current_calc_path = self.vasp_result_path

    @staticmethod
    def generate_poscar(struc: Structure, file_path):
        struc.to(filename=file_path, fmt='poscar')

    @abstractmethod
    def generate_incar(self, file_path):
        pass

    def generate_potcar(self, file_path):
        path = os.path.join(get_program_path(), self.current_calc_path)
        path = path[0].lower() + path[1:]
        path = '/mnt/' + '/'.join(path.replace(':', '').split('\\'))
        path = path.replace('./', '')

        cmd = r" cd '%s'; echo -e '103\n' | vaspkit >> output.log  2>&1 " % path
        self.exec_shell(cmd)

    def generate_input_files(self, struc: Structure, **kwargs):
        if self.is_save_result:
            self.count += 1
            self.current_calc_path = os.path.join(self.vasp_result_path, str(self.count))
            check_and_rename_path(self.current_calc_path)

        poscar_path = os.path.join(self.current_calc_path, 'POSCAR')
        if not os.path.isfile(poscar_path):
            self.generate_poscar(struc, poscar_path)

        incar_path = os.path.join(self.current_calc_path, 'INCAR')
        if not os.path.isfile(incar_path):
            self.generate_incar(incar_path)
        # shutil.copy(incar_path, self.current_calc_path)

        potcar_path = os.path.join(self.current_calc_path, 'POTCAR')
        if not os.path.isfile(potcar_path):
            self.generate_potcar(potcar_path)
        # shutil.copy(potcar_path, self.current_calc_path)

    @staticmethod
    def exec_shell(cmd):
        os.system(cmd)

        # wslenv_file = '/mnt/d/python/D3REAM/calculators/VASP/wslenv.sh'
        # # print('''bash -c " source '%s'; %s "''' % (wslenv_file, cmd))
        # os.system('''bash -c " source '%s'; %s "''' % (wslenv_file, cmd))

    def run_vasp(self, vasp_cores=6, **kwargs):
        def gpid(fpid):
            time.sleep(2)
            allpid = []
            allpid.append(fpid)
            b = os.popen('ps --ppid ' + fpid).read().split('\n')
            while len(b) == 3:
                spid = b[1].split()[0]
                allpid.append(spid)
                b = os.popen('ps --ppid ' + spid).read().split('\n')
            return ' '.join(allpid)

        # path = os.path.join(get_program_path(), run_path)
        # path = path[0].lower() + path[1:]
        # path = '/mnt/' + '/'.join(path.replace(':', '').split('\\'))
        # path = path.replace('./', '')
        cmd = r'''cd "%s"; mpirun -np %d vasp_std >> output.log  2>&1 ''' % (self.current_calc_path, vasp_cores)

        # self.exec_shell(cmd)

        p = Process(target=self.exec_shell, args=(cmd,))
        p.start()
        start = time.time()
        apid = gpid(str(p.pid))
        while 1:
            time.sleep(3)

            if p.is_alive():
                end = time.time()
                if end - start < 60*10:
                    time.sleep(10)
                    continue
                else:
                    self.exec_shell("kill -9 " + apid)
                    print('Timeout! Job is killed!')
                    return 0
            else:
                break

        return 1

    @abstractmethod
    def get_target(self, struc: Structure, **kwargs):
        pass
