# -*- coding: utf-8 -*-
# ========================================================================
#  2022/5/23 21:34
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

import numpy as np


def parameter_config(label, in_value, vtype=None, ptype='float'):
    """
    {"_type": "choice", "_value": options}
    {"_type": "ordinal", "_value": sequence}
    {"_type": "uniform", "_value": [low, high]}
    {"_type": "quniform", "_value": [low, high, q]}
    {"_type": "loguniform", "_value": [low, high]}
    {"_type": "qloguniform", "_value": [low, high, q]}
    {"_type": "int_uniform", "_value": [low, high]}
    {label: {"_type": "int_quniform", "_value": [low, high, q]}}
    :param label:
    :param in_value:
    :param vtype:
    :param ptype:
    :return:
    """

    if vtype:
        return {label: {"_type": vtype, "_value": in_value}}

    if type(in_value) is list:
        if len(in_value) == 2 and in_value[0] == in_value[1]:
            parameter = {label: {"_type": "choice", "_value": [in_value[0], ]}}
        elif ptype == 'int':
            parameter = {label: {"_type": "int_uniform", "_value": in_value}}
        else:
            parameter = {label: {"_type": "uniform", "_value": in_value}}
    elif type(in_value) is tuple:
        parameter = {label: {"_type": "choice", "_value": list(in_value)}}
    elif type(in_value) is int or type(in_value) is float:
        parameter = {label: {"_type": "choice", "_value": [in_value, ]}}
    else:
        raise Exception("Parameter `" + label + "` setting error! Please check!")

    return parameter


def range_parameter_to_list(param: str, step: float = 1, ptype='float') -> list:
    new_param = []

    param = ''.join(param.split())
    param = param.replace('(', '[').replace('{', '[').replace('<', '[')
    param = param.replace(')', ']').replace('}', ']').replace('>', ']')

    param = param.replace('][', ' ').replace('[', '').replace(']', '')
    param = param.split()

    for p in param:
        p_tmp = []
        for pp in p.split(','):
            if '-' in pp:
                pp = pp.split('-')
                if len(pp) != 2:
                    raise RuntimeError('Parameter `atom_element/atom_count` error! Please check!')
                p_tmp += list(np.arange(float(pp[0]), float(pp[1]) + step, step))
            else:
                p_tmp.append(float(pp))

        if ptype == 'int':
            p_tmp = [int(pt) for pt in p_tmp]
        new_param.append(p_tmp)

    return new_param
