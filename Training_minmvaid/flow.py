import glob
import os

import hist
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
import torch.utils.data as t_data
import utils as utlis
import yaml
import zuko
from sklearn.model_selection import train_test_split
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.utils.data import DataLoader
from yaml import Loader

# Set the mplhep style to CMS for plots
hep.style.use("CMS")


# %% get data and simulation
path_df = "/net/scratch_cms3a/daumann/normalizing_flows_project/script_to_prepare_samples_for_paper/splited_parquet/Diphoton_samples/"
df_data = pd.read_parquet(path_df + "Data_postEE.parquet")
df_Diphoton = pd.read_parquet(path_df + "Diphoton_postEE.parquet")
df_GJEt = pd.read_parquet(path_df + "GJEt_postEE.parquet")

path = "/home/home1/institut_3a/jaensch/Documents/BA/BA/Diphoton/Training_minmvaid/"

# %% plot histograms
var_list = [
    "r9",
    "etaWidth",
    "phiWidth",
    "s4",
    "sieie",
    "sieip",
]

conditions_list = ["pt", "ScEta", "phi", "fixedGridRhoAll", "mass"]

# %%
df_Diphoton["origin"] = 1
df_GJEt["origin"] = 0
df_data["origin"] = 2

# Combine the diphoton and g_jet dataframes
mc_df = pd.concat([df_Diphoton, df_GJEt])
mc_df["is_data"] = 0  # Add a new column 'is_data' that is 0 for the simulation

# Add a new column 'is_data' that is 1 for the data
df_data["is_data"] = 1

# Combine the simulation and data dataframes
combined_df = pd.concat([mc_df, df_data])
selection_mass = (combined_df["mass"] > 100) & (combined_df["mass"] < 180)
combined_df = combined_df[selection_mass]
# Define the variables you want to select
other_vars = [
    "energyRaw",
    "hoe",
    "ecalPFClusterIso",
    "hcalPFClusterIso",
    "trkSumPtHollowConeDR03",
    "trkSumPtSolidConeDR04",
    "pfChargedIso",
    "pfChargedIsoWorstVtx",
    "esEffSigmaRR",
    "esEnergyOverRawE",
]
variables = var_list + conditions_list + other_vars

variables_no_prefix = ["mass", "fixedGridRhoAll"]

# For each variable, add a new column to the DataFrame that contains the value of the variable with the minimum mvaID
for var in variables:
    if var in variables_no_prefix:
        combined_df[var + "_minmvaID"] = combined_df[var]
    else:
        combined_df[var + "_minmvaID"] = np.where(
            combined_df["lead_mvaID"] < combined_df["sublead_mvaID"],
            combined_df["lead_" + var],
            combined_df["sublead_" + var],
        )

input_list = [var + "_minmvaID" for var in var_list]
conditions_list = [var + "_minmvaID" for var in conditions_list]


# Split the combined dataframe into inputs and conditions
inputs = combined_df[input_list + ["weight"]]
conditions = combined_df[
    conditions_list + ["is_data"]
]  # Include the 'is_data' column in the conditions

# Identify indices of rows with nan in either DataFrame
nan_rows = inputs.isnull().any(axis=1) | conditions.isnull().any(axis=1)

# Drop these rows from both DataFrames
inputs = inputs.loc[~nan_rows]
conditions = conditions.loc[~nan_rows]
all_info = combined_df.loc[~nan_rows]

# Split the inputs and conditions into training, testing, and validation sets
(
    inputs_train,
    inputs_test,
    conditions_train,
    conditions_test,
    all_info_train,
    all_info_test,
) = train_test_split(inputs, conditions, all_info, test_size=0.2, random_state=42)
(
    inputs_train,
    inputs_val,
    conditions_train,
    conditions_val,
    all_info_train,
    all_info_val,
) = train_test_split(
    inputs_train, conditions_train, all_info_train, test_size=0.25, random_state=42
)

# From numpy to pytorch tensors
training_inputs = torch.Tensor(np.array(inputs_train))
validation_inputs = torch.Tensor(np.array(inputs_val))
test_inputs = torch.Tensor(np.array(inputs_test))

training_conditions = torch.Tensor(np.array(conditions_train))
validation_conditions = torch.Tensor(np.array(conditions_val))
test_conditions = torch.Tensor(np.array(conditions_test))

all_info_test.to_csv(path + "all_info_test.csv", index=False)


# %%
def normalize_weights(inputs, conditions):
    # Extract "is_data" column from conditions
    is_data_column = conditions[:, -1]  # Assuming "is_data" is the last column

    # Extract "weights" from inputs
    weights = inputs[:, -1]  # Assuming "weights" is the last column

    # Normalize weights separately for "data" and "not data" categories
    data_indices = is_data_column.nonzero().squeeze()  # Indices where is_data is True
    not_data_indices = (
        (1 - is_data_column).nonzero().squeeze()
    )  # Indices where is_data is False

    scaling_factors = {}

    # Normalize weights for "data" if there are data samples
    if len(data_indices) > 0:
        data_weights = weights[data_indices]
        scaling_factors["data"] = sum(data_weights)
        normalized_data_weights = data_weights / scaling_factors["data"]
        inputs[data_indices, -1] = normalized_data_weights

    # Normalize weights for "not data" if there are not data samples
    if len(not_data_indices) > 0:
        not_data_weights = weights[not_data_indices]
        scaling_factors["not_data"] = sum(not_data_weights)
        normalized_not_data_weights = not_data_weights / scaling_factors["not_data"]
        inputs[not_data_indices, -1] = normalized_not_data_weights

    return inputs, scaling_factors


def apply_scaling(inputs, conditions, scaling_factors):
    # Extract "is_data" column from conditions
    is_data_column = conditions[:, -1]  # Assuming "is_data" is the last column

    # Extract "weights" from inputs
    weights = inputs[:, -1]  # Assuming "weights" is the last column

    # Apply scaling factors separately for "data" and "not data" categories
    data_indices = is_data_column.nonzero().squeeze()  # Indices where is_data is True
    not_data_indices = (
        (1 - is_data_column).nonzero().squeeze()
    )  # Indices where is_data is False

    # Apply scaling factor for "data" if there are data samples
    if len(data_indices) > 0:
        data_weights = weights[data_indices]
        inputs[data_indices, -1] = data_weights / scaling_factors["data"]

    # Apply scaling factor for "not data" if there are not data samples
    if len(not_data_indices) > 0:
        not_data_weights = weights[not_data_indices]
        inputs[not_data_indices, -1] = not_data_weights / scaling_factors["not_data"]

    return inputs


# Normalize the weights for the training set and get the scaling factors
training_inputs, scaling_factors = normalize_weights(
    training_inputs, training_conditions
)

# Apply the same scaling factors to the validation and test sets
validation_inputs = apply_scaling(
    validation_inputs, validation_conditions, scaling_factors
)
test_inputs = apply_scaling(test_inputs, test_conditions, scaling_factors)

# Separate the weights and meta data from the rest of the data
training_inputs, training_weights = (
    training_inputs[:, :-1],
    training_inputs[:, -1],
)
validation_inputs, validation_weights = (
    validation_inputs[:, :-1],
    validation_inputs[:, -1],
)
test_inputs, test_weights = (
    test_inputs[:, :-1],
    test_inputs[:, -1],
)
# %%
# standardize the data

np.save(
    path + "input_means.npy",
    training_inputs.mean(dim=0).numpy(),
)
np.save(
    path + "input_std.npy",
    training_inputs.std(dim=0).numpy(),
)
np.save(
    path + "conditions_means.npy",
    training_conditions[:, :-1].mean(dim=0).numpy(),
)
np.save(
    path + "conditions_std.npy",
    training_conditions[:, :-1].std(dim=0).numpy(),
)

(
    training_inputs,
    training_conditions,
    input_mean_for_std,
    input_std_for_std,
    condition_mean_for_std,
    condition_std_for_std,
) = utlis.standardize(
    training_inputs,
    training_conditions,
    path=path,
)

(
    test_inputs,
    test_conditions,
    _,
    _,
    _,
    _,
) = utlis.standardize(
    test_inputs,
    test_conditions,
    path=path,
)

(
    validation_inputs,
    validation_conditions,
    _,
    _,
    _,
    _,
) = utlis.standardize(
    validation_inputs,
    validation_conditions,
    path=path,
)

torch.save(
    test_inputs,
    path + "test_inputs.pt",
)
torch.save(
    test_conditions,
    path + "test_conditions.pt",
)

torch.save(
    test_weights,
    path + "test_weights.pt",
)

torch.save(
    training_inputs,
    path + "training_inputs.pt",
)
torch.save(
    training_conditions,
    path + "training_conditions.pt",
)

torch.save(
    training_weights,
    path + "training_weights.pt",
)


# %% flow training

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

stream = open(path + "flow_config.yaml", "r")
dictionary = yaml.load(stream, Loader)

for key in dictionary:

    # network configurations
    n_transforms = dictionary[key]["n_transforms"]  # number of transformation
    aux_nodes = dictionary[key]["aux_nodes"]  # number of nodes in the auxiliary network
    aux_layers = dictionary[key][
        "aux_layers"
    ]  # number of auxiliary layers in each flow transformation
    n_splines_bins = dictionary[key][
        "n_splines_bins"
    ]  # Number of rationale quadratic spline flows bins

    # Some general training parameters
    max_epoch_number = dictionary[key]["max_epochs"]
    initial_lr = dictionary[key]["initial_lr"]
    batch_size = dictionary[key]["batch_size"]
# %%
# flow training
flow = zuko.flows.NSF(
    test_inputs.size()[1],
    context=test_conditions.size()[1],
    bins=n_splines_bins,
    transforms=n_transforms,
    hidden_features=[aux_nodes] * aux_layers,
)
flow.to(device)

optimizer = torch.optim.AdamW(flow.parameters(), lr=initial_lr, weight_decay=1e-6)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=10)

training_inputs = training_inputs.to(device)
training_conditions = training_conditions.to(device)

validation_inputs = validation_inputs.to(device)
validation_conditions = validation_conditions.to(device)

training_loss_array = []
validation_loss_array = []

training_inputs = training_inputs.type(dtype=training_conditions.dtype)
flow = flow.type(training_inputs.dtype)

# training_weights = training_weights / torch.sum(training_weights)
# validation_weights = validation_weights / torch.sum(validation_weights)

print(training_weights)
print(validation_weights)

print("Start training")
save_dir = path + "results/saved_states"
os.makedirs(save_dir, exist_ok=True)
for epoch in range(999):
    epoch_loss = 0.0
    epoch_validation_loss = 0.0
    for batch in range(250):

        optimizer.zero_grad()

        idxs = torch.randint(low=0, high=training_inputs.size()[0], size=(batch_size,))

        loss = training_weights[idxs].to(device) * (
            -flow(training_conditions[idxs]).log_prob(training_inputs[idxs])
        )
        loss = loss.mean()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(flow.parameters(), 1.0e-1)

        optimizer.step()

        epoch_loss += loss.item()

        # Validation loss for each batch
        with torch.no_grad():
            idxs = torch.randint(
                low=0, high=validation_inputs.size()[0], size=(batch_size,)
            )

            validation_loss = validation_weights[idxs].to(device) * (
                -flow(validation_conditions[idxs]).log_prob(validation_inputs[idxs])
            )
            validation_loss = validation_loss.mean()

            epoch_validation_loss += validation_loss.item()

    epoch_loss /= 250  # average training loss for the epoch
    epoch_validation_loss /= 250  # average validation loss for the epoch

    torch.save(
        flow.state_dict(),
        save_dir + "/epoch_" + str(epoch) + ".pth",
    )

    training_loss_array.append(float(1e6 * epoch_loss))
    validation_loss_array.append(float(1e6 * epoch_validation_loss))

    scheduler.step(epoch_validation_loss)

    print(
        "Epoch: ",
        epoch,
        " Training loss: ",
        float(1e6 * epoch_loss),
        " Validation loss: ",
        float(1e6 * epoch_validation_loss),
    )

    # TODO: Early stopping and save the model
    if epoch > max_epoch_number:

        print(
            "Best epoch loss: ",
            np.min(np.array(validation_loss_array)),
            " at epoch: ",
            np.argmin(np.array(np.array(validation_loss_array))),
        )

        # Lets select the model with the best validation loss
        flow.load_state_dict(
            torch.load(
                save_dir
                + "/epoch_"
                + str(np.argmin(np.array(np.array(validation_loss_array))))
                + ".pth"
            )
        )
        torch.save(flow.state_dict(), save_dir + "/best_model_.pth")

        break

# plot the training and validation loss
utlis.plot_loss_cruve(
    training_loss_array,
    validation_loss_array,
    path,
)
