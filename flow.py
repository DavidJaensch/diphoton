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

mva_id_mask = True
scaled = True
stacked = True


def create_directory(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


# %% get data and simulation
if scaled == False:
    path_df = "/net/scratch_cms3a/daumann/normalizing_flows_project/script_to_prepare_samples_for_paper/splited_parquet/Diphoton_samples/"
    df_data = pd.read_parquet(path_df + "Data_postEE.parquet")
    df_Diphoton = pd.read_parquet(path_df + "Diphoton_postEE.parquet")
    df_GJEt = pd.read_parquet(path_df + "GJEt_postEE.parquet")

    path = "/home/home1/institut_3a/jaensch/Documents/BA/BA/Diphoton/unscaled/"
    path_base = "/home/home1/institut_3a/jaensch/Documents/BA/BA/Diphoton/"

elif scaled == True and stacked == True:
    path_df = "/net/scratch_cms3a/daumann/normalizing_flows_project/script_to_prepare_samples_for_paper/diphoton_samples/Scale_diphoton_and_GJet/"
    df_data = pd.read_parquet(path_df + "Data_postEE.parquet")
    df_Diphoton = pd.read_parquet(path_df + "Diphoton_postEE.parquet")
    df_GJEt = pd.read_parquet(path_df + "GJEt_postEE.parquet")

    path = "/home/home1/institut_3a/jaensch/Documents/BA/BA/Diphoton/scaled/"
    path_base = "/home/home1/institut_3a/jaensch/Documents/BA/BA/Diphoton/"

elif scaled == True and stacked == False:
    path_df = "/net/scratch_cms3a/daumann/normalizing_flows_project/script_to_prepare_samples_for_paper/diphoton_samples/Scale_only_diphoton/"
    df_data = pd.read_parquet(path_df + "Data_postEE.parquet")
    df_Diphoton = pd.read_parquet(path_df + "Diphoton_postEE.parquet")
    df_GJEt = pd.read_parquet(path_df + "GJEt_postEE.parquet")

    path = (
        "/home/home1/institut_3a/jaensch/Documents/BA/BA/Diphoton/scaled_only_diphoton/"
    )
    path_base = "/home/home1/institut_3a/jaensch/Documents/BA/BA/Diphoton/"


create_directory(path)

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

plot_list = ["lead_" + s for s in var_list]

plot_list = ["lead_" + s for s in var_list]
plot_list.append("mass")
plot_list.append("lead_pt")

input_list_lead = ["lead_" + s for s in var_list]
n_input = len(
    input_list_lead
)  # so i can add other information i need to the input list, but only use the here defined as input
n_conditions = len(conditions_list)
conditions_list_lead = ["lead_" + s for s in conditions_list[:-2]]
conditions_list_lead.append("fixedGridRhoAll")
conditions_list_lead.append("mass")

"""
if scaled == False:
    plot_list = ["lead_" + s for s in var_list]
    plot_list.append("mass")
    plot_list.append("lead_pt")

    input_list_lead = ["lead_" + s for s in var_list]
    n_input = len(
        input_list_lead
    )  # so i can add other information i need to the input list, but only use the here defined as input
    n_conditions = len(conditions_list)
    conditions_list_lead = ["lead_" + s for s in conditions_list[:-2]]
    conditions_list_lead.append("fixedGridRhoAll")
    conditions_list_lead.append("mass")
    df_data["lead_corr_mvaID_run3"] = df_data["lead_mvaID"]
    df_Diphoton["lead_corr_mvaID_run3"] = df_Diphoton["lead_mvaID"]
    df_GJEt["lead_corr_mvaID_run3"] = df_GJEt["lead_mvaID"]

elif scaled == True and stacked == True:
    plot_list = ["lead_corr_" + s for s in var_list]
    plot_list.append("mass")
    plot_list.append("lead_pt")

    input_list_lead = ["lead_corr_" + s for s in var_list]
    n_input = len(
        input_list_lead
    )  # so i can add other information i need to the input list, but only use the here defined as input
    n_conditions = len(conditions_list)
    conditions_list_lead = ["lead_" + s for s in conditions_list[:-2]]
    conditions_list_lead.append("fixedGridRhoAll")
    conditions_list_lead.append("mass")

    variables_no_prefix = ["mass", "fixedGridRhoAll"]

    for var in var_list:
        if var in variables_no_prefix:
            df_data[var] = df_data[var]
        else:
            df_data["lead_corr_" + var] = df_data["lead_" + var]
            df_data["lead_corr_mvaID_run3"] = df_data["lead_mvaID"]

elif scaled == True and stacked == False:
    plot_list = ["lead_corr_" + s for s in var_list]
    plot_list.append("mass")
    plot_list.append("lead_pt")

    input_list_lead = ["lead_corr_" + s for s in var_list]
    n_input = len(
        input_list_lead
    )  # so i can add other information i need to the input list, but only use the here defined as input
    n_conditions = len(conditions_list)
    conditions_list_lead = ["lead_" + s for s in conditions_list[:-2]]
    conditions_list_lead.append("fixedGridRhoAll")
    conditions_list_lead.append("mass")

    variables_no_prefix = ["mass", "fixedGridRhoAll"]

    for var in var_list:
        df_data["lead_corr_" + var] = df_data["lead_" + var]
        df_data["lead_corr_mvaID_run3"] = df_data["lead_mvaID"]
        df_GJEt["lead_corr_" + var] = df_GJEt["lead_" + var]
        df_data["lead_corr_mvaID_run3"] = df_data["lead_mvaID"]

"""


def plot_hist(df_diphoton, df_g_jet, df_data, var):
    plt.clf()
    plt.style.use(hep.style.CMS)

    # Calculate the mean and standard deviation of the data
    mean = np.mean(np.concatenate([df_diphoton[var], df_g_jet[var], df_data[var]]))
    std = np.std(np.concatenate([df_diphoton[var], df_g_jet[var], df_data[var]]))

    # Define the bins for the histogram
    if var == "mass":
        min_value = 100
        max_value = 180
        num_bins = 50
    else:
        num_bins = 50
        min_value = mean - 3 * std
        max_value = mean + 3 * std

    # Create the diphoton and g_jet histograms and stack them
    hist_diphoton = (
        hist.Hist.new.Reg(num_bins, min_value, max_value)
        .Weight()
        .fill(
            df_diphoton.loc[df_diphoton["lead_mvaID"] > 0.5, var],
            weight=df_diphoton.loc[df_diphoton["lead_mvaID"] > 0.5, "weight"],
        )
    )
    hist_g_jet = (
        hist.Hist.new.Reg(num_bins, min_value, max_value)
        .Weight()
        .fill(
            df_g_jet.loc[df_g_jet["lead_mvaID"] > 0.5, var],
            weight=df_g_jet.loc[df_g_jet["lead_mvaID"] > 0.5, "weight"],
        )
    )
    sum_mc = hist_diphoton.values().sum() + hist_g_jet.values().sum()
    bin_width = (max_value - min_value) / num_bins
    hist_diphoton = hist_diphoton / (sum_mc * bin_width)
    hist_g_jet = hist_g_jet / (sum_mc * bin_width)

    hep.histplot(
        [hist_diphoton, hist_g_jet],
        stack=True,
        histtype="fill",
        label=["Diphoton", "G Jet"],
    )

    # Create the data histogram and plot it with only the top marks
    hist_data = (
        hist.Hist.new.Reg(num_bins, min_value, max_value)
        .Weight()
        .fill(
            df_data.loc[df_data["lead_mvaID"] > 0.5, var],
            weight=df_data.loc[df_data["lead_mvaID"] > 0.5, "weight"],
        )
    )
    hist_data = hist_data / (hist_data.values().sum() * bin_width)
    bin_edges = hist_data.axes[0].edges
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    plt.errorbar(bin_centers, hist_data.values(), fmt="x", color="black", label="Data")

    plt.xlabel(var)
    plt.ylabel("Events/Bin")
    plt.legend()
    hep.cms.label("Work in Progress", data=False)
    plt.savefig(path + f"hist_{var}.png")
    plt.show()


for var in plot_list:
    plot_hist(df_Diphoton, df_GJEt, df_data, var)

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
if scaled == False:
    selection_mass = (
        (combined_df["mass"] > 100)
        & (combined_df["mass"] < 180)
        & ((combined_df["lead_mvaID"] > 0.5) if mva_id_mask else True)
    )
else:
    selection_mass = (
        (combined_df["mass"] > 100)
        & (combined_df["mass"] < 180)
        & ((combined_df["lead_mvaID"] > 0.5) if mva_id_mask else True)
    )
combined_df = combined_df[selection_mass]


input_list_lead = input_list_lead + ["weight"]

# Split the combined dataframe into inputs and conditions
inputs = combined_df[input_list_lead]
conditions = combined_df[
    conditions_list_lead + ["is_data"]
]  # Include the 'is_data' column in the conditions
# """
# Identify indices of rows with nan in either DataFrame
nan_rows = inputs.isnull().any(axis=1) | conditions.isnull().any(axis=1)

# Drop these rows from both DataFrames
inputs = inputs.loc[~nan_rows]
conditions = conditions.loc[~nan_rows]
all_info = combined_df.loc[~nan_rows]
# """
# all_info = combined_df
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
    training_inputs[:, :n_input],
    training_inputs[:, -1],
)
validation_inputs, validation_weights = (
    validation_inputs[:, :n_input],
    validation_inputs[:, -1],
)
test_inputs, test_weights = (
    test_inputs[:, :n_input],
    test_inputs[:, -1],
)
# %%
# standardize the data
"""
np.save(
    path + "input_means.npy",
    training_inputs.mean(dim=0).numpy(),
)
np.save(
    path + "input_std.npy",
    training_inputs.std(dim=0).numpy() * 2,
)
np.save(
    path + "conditions_means.npy",
    training_conditions[:, :-1].mean(dim=0).numpy(),
)
np.save(
    path + "conditions_std.npy",
    training_conditions[:, :-1].std(dim=0).numpy() * 2,
)
"""
np.save(
    path + "input_means.npy",
    np.average(training_inputs, weights=training_weights, axis=0),
)
np.save(
    path + "input_std.npy",
    np.sqrt(
        np.average(
            (
                training_inputs
                - np.average(training_inputs, weights=training_weights, axis=0)
            )
            ** 2,
            weights=training_weights,
            axis=0,
        )
    )
    * 2,
)
np.save(
    path + "conditions_means.npy",
    np.average(training_conditions[:, :-1], weights=training_weights, axis=0),
)
np.save(
    path + "conditions_std.npy",
    np.sqrt(
        np.average(
            (
                training_conditions[:, :-1]
                - np.average(
                    training_conditions[:, :-1], weights=training_weights, axis=0
                )
            )
            ** 2,
            weights=training_weights,
            axis=0,
        )
    )
    * 2,
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

# Convert tensor to numpy array
zmmg_inputs_np = training_inputs.numpy()


# Convert numpy array to DataFrame
zmmg_inputs_df = pd.DataFrame(zmmg_inputs_np)

# Plot histograms
zmmg_inputs_df.hist(bins=50, figsize=(20, 15))
plt.tight_layout()
plt.savefig(path + "hist_inputs.png")
plt.show()

zmmg_inputs_np = training_conditions.numpy()

# Convert numpy array to DataFrame
zmmg_inputs_df = pd.DataFrame(zmmg_inputs_np)

# Plot histograms
zmmg_inputs_df.hist(bins=50, figsize=(20, 15))
plt.tight_layout()
plt.savefig(path + "hist_conditions.png")
plt.show()

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

stream = open(path_base + "flow_config.yaml", "r")
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
        if mva_id_mask == True:
            torch.save(flow.state_dict(), save_dir + "/best_model_mva_id_mask.pth")
        else:
            torch.save(flow.state_dict(), save_dir + "/best_model_.pth")

        break

# plot the training and validation loss
utlis.plot_loss_cruve(
    training_loss_array,
    validation_loss_array,
    path,
)

# %%
