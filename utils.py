# python libraries import
import glob
import os

import numpy as np
import pandas as pd
import torch
import zuko


def standardize(input_tensor, conditions_tensor, path):
    # Now, the standartization -> The arrays are the same ones using during training
    input_mean_for_std = torch.tensor(np.load(path + "input_means.npy"))
    input_std_for_std = torch.tensor(np.load(path + "input_std.npy"))
    condition_mean_for_std = torch.tensor(np.load(path + "conditions_means.npy"))
    condition_std_for_std = torch.tensor(np.load(path + "conditions_std.npy"))
    input_tensor = (input_tensor - input_mean_for_std) / input_std_for_std
    conditions_tensor[:, :-1] = (
        conditions_tensor[:, :-1] - condition_mean_for_std
    ) / condition_std_for_std
    return (
        input_tensor,
        conditions_tensor,
        input_mean_for_std,
        input_std_for_std,
        condition_mean_for_std,
        condition_std_for_std,
    )
