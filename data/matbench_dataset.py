# -*- coding: utf-8 -*-
# ========================================================================
#  2022/1/13 18:22
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

from data.dataset_base import DatasetBase


class MatBenchDataset(DatasetBase):
    def get_dataset(self):
        from matminer.datasets import load_dataset

        df = load_dataset("matbench_mp_e_form")  # pandas --> DataFrame,,,,,*.json
        df = df[:100]
        # print(df)
        # df = load_dataset("matbench_expt_gap")  # pandas --> DataFrame
        structures, targets = df['structure'], df['e_form']

        return structures, targets


