# -*- coding: utf-8 -*-
# ========================================================================
#  2022/6/21 19:44
#                 _____   _   _   _____   __   _   _____  
#                /  ___| | | | | | ____| |  \ | | /  ___| 
#                | |     | |_| | | |__   |   \| | | |     
#                | |     |  _  | |  __|  | |\   | | |  _  
#                | |___  | | | | | |___  | | \  | | |_| | 
#                \_____| |_| |_| |_____| |_|  \_| \_____/ 
# ------------------------------------------------------------------------
#
# Get the WyckPos combination
# The code is inspired by the PyXtal
# 
# ========================================================================

import os
import itertools
from copy import deepcopy
import numpy as np
import pandas as pd


class GenCrystalWyckPos:

    def __init__(
            self,
            group,
            species,
            numIons,
            sites,
            flexible_site=True,
    ):
        self.flexible_site = flexible_site

        self.group = group
        self.numIons = np.array(numIons)
        self.species = species

        self.wyckoff_string = None
        self.wyckoff_string_len = None

        self.wp_com_tmp = []
        self.set_sites(sites)

    def set_sites(self, sites):
        """
        initialize Wyckoff sites

        Args:
            sites: list
        """
        # Symmetry sites
        self.sites = {}
        for i, specie in enumerate(self.species):
            if sites is not None and sites[i] is not None and len(sites[i]) > 0:
                if type(sites[i]) is dict:
                    self.sites[specie] = []
                    for item in sites[i].items():
                        self.sites[specie].append({item[0]: item[1]})
                else:
                    self.sites[specie] = sites[i]
            else:
                self.sites[specie] = None

    def get_WyckPos_combination(self):
        WyckPos_combinations = []

        for numIon, specie in zip(self.numIons, self.species):
            output = self._set_ion_wyckoffs(numIon, specie)

            len_output = len(list(itertools.chain(*output)))

            if not self.flexible_site and len_output != numIon:
                raise Exception("bad wp combination, recommend use flexible site!")

            while len_output > numIon:
                len_output -= len(output[-1])
                output.pop(-1)
            if len_output < numIon:
                output.extend([['x, y, z']] * (numIon - len_output))

            WyckPos_combinations.append(output)

        return WyckPos_combinations

    def _set_ion_wyckoffs(self, numIon, specie):
        numIon_added = 0
        wyckoff_sites_tmp = []

        # Now we start to add the specie to the wyckoff position
        sites_list = deepcopy(self.sites[specie])  # the list of Wyckoff site
        wyckoff_attempts = max(len(sites_list) * 2, 10)

        cycle = 0
        while cycle < wyckoff_attempts:
            # Choose a random WP for given multiplicity: 2a, 2b
            if sites_list is not None and len(sites_list) > 0:
                site = sites_list[0]
            else:  # Selecting the merging
                site = None

            wp = self.choose_wyckoff(site)
            if wp is False:
                cycle += 1
                continue

            wp_len = len(wp)

            if wp is not False:
                # Generate a list of coords from ops
                mult = wp_len  # remember the original multiplicity
                # For pure planar structure

                # If site the pre-assigned, do not accept merge
                if wp is not False:
                    if site is not None and mult != wp_len:
                        cycle += 1
                        continue

            if sites_list is not None and len(sites_list) > 0:
                sites_list.pop(0)

            ws_str_tmp = ';'.join(wp)
            if ('x' not in ws_str_tmp and 'y' not in ws_str_tmp and 'z' not in ws_str_tmp) and \
                    (ws_str_tmp in self.wp_com_tmp):
                cycle += 1
                continue
            wyckoff_sites_tmp.append(wp)
            self.wp_com_tmp.append(ws_str_tmp)
            numIon_added += wp_len
            if numIon_added >= numIon:
                return wyckoff_sites_tmp

            cycle += 1

        return wyckoff_sites_tmp

    def choose_wyckoff(self, site=None):
        if site is not None:
            try:
                index = self.wyckoff_string_len - 1 - wp_alphabet.index(site)
                return self.wyckoff_string[index]
            except:
                return False
        else:
            return False

    # @property
    # def wyckoff_string(self):
    #     return self.wyckoff_string
    #
    # @wyckoff_string.setter
    # def wyckoff_string(self, _wyckoff_string):
    #     self.wyckoff_string = _wyckoff_string
    #
    # @property
    # def wyckoff_string_len(self):
    #     return self.wyckoff_string_len
    #
    # @wyckoff_string_len.setter
    # def wyckoff_string_len(self, _wyckoff_string_len):
    #     self.wyckoff_string_len = _wyckoff_string_len


wp_alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g',
               'h', 'i', 'j', 'k', 'l', 'm', 'n',
               'o', 'p', 'q', 'r', 's', 't',
               'u', 'v', 'w', 'x', 'y', 'z',
               'A', ]


wyckoff_df = pd.read_csv(os.path.join(os.path.split(__file__)[0], "wyckoff_list.csv"))
# layer_df = pd.read_csv(rf("pyxtal", "database/layer.csv"))
# rod_df = pd.read_csv(rf("pyxtal", "database/rod.csv"))
# point_df = pd.read_csv(rf("pyxtal", "database/point.csv"))


def get_wyckoffs_strings(num, dim=3):
    if dim == 3:
        wyckoff_strings = eval(wyckoff_df["0"][num])
    # elif dim == 2:
    #     wyckoff_strings = eval(layer_df["0"][num])
    # elif dim == 1:
    #     wyckoff_strings = eval(rod_df["0"][num])
    # elif dim == 0:
    #     wyckoff_strings = eval(point_df["0"][num])
    else:
        wyckoff_strings = eval(wyckoff_df["0"][num])

    return wyckoff_strings


if __name__ == '__main__':
    # print(get_wyckoffs_strings(225))

    _gcw = GenCrystalWyckPos(
        225, ['Ca', 'S'], [4, 4], [['a'], ['b']],
        flexible_site=True,
    )
    wyckoffs_strings = get_wyckoffs_strings(225)
    wyckoffs_strings_len = len(wyckoffs_strings)
    _gcw.wyckoff_string = wyckoffs_strings
    _gcw.wyckoff_string_len = wyckoffs_strings_len
    _wp = _gcw.get_WyckPos_combination()
    print(_wp)
