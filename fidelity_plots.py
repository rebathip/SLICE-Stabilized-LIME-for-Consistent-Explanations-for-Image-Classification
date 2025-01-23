import os
import pickle
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import pingouin as pg


metric_key = 'aopc_del' # 'ins', 'del', 'aopc_del', 'aopc_ins' as per which metric u want to analyze

# Define results directory
results_dir = "C:/Users/revotipb/Projects_Notebooks/bayslice/notebooks/fidelity_results/" + metric_key + '/'

# Initialize dictionary to store AOPC results
aopc_dict = {}

# Process each file in the directory
files = os.listdir(results_dir)

# Specify your dataset and model names
dataset_name = "oxpets"
model_name = "inceptionv3"

# Filter files that contain both dataset_name and model_name
#files = [file for file in files if dataset_name in file and model_name in file]
print(files)
for file in files:
    with open(results_dir + file, 'rb') as f:
        data = pickle.load(f)

    img_keys = data.keys()
    aopcs = []
    for img_key in img_keys:
        img_res = data[img_key]

        if isinstance(img_res, np.ndarray):  # Check if img_res is a numpy array
            # Compute mean of the array if it's a numpy array
            aopcs.append(np.mean(img_res[0]))

        elif isinstance(img_res, list) and all(isinstance(item, dict) for item in img_res):
            mean_pos_neg = [np.mean(entry['pos']) for entry in img_res]
            aopcs.append(np.mean(mean_pos_neg))
        else:
            print(f"Unexpected data format in {file} for key {img_key}")

    # Extract parts of the filename to construct dictionary keys
    parts = file.split('_')
    aopc_dict_key = '_'.join(parts[:3])
    key_parts = aopc_dict_key.split('_')
    aopc_dict_key = '_'.join([key_parts[0], key_parts[2], key_parts[1]])

    # Store AOPC values in the dictionary
    aopc_dict[aopc_dict_key] = np.array(aopcs)


# Set matplotlib parameters
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.size'] = 15
mpl.rcParams['font.weight'] = 'bold'

# Define the size for each subplot
subplot_size = 3  # Adjust for the size of each subplot

# Create a new figure with 2x2 subplots
fig, axes = plt.subplots(2, 2, figsize=(2 * subplot_size, 2 * subplot_size))

# Define titles and keys for each subplot
title_index = ['Oxford-IIIT Pets:InceptionV3', 'Oxford-IIIT Pets:ResNet50', 'Pascal VOC2007:Inception V3', 'Pascal VOC2007:ResNet50']
slice_keys = ['oxpets_inceptionv3_sliceblurfe', 'oxpets_resnet50_sliceblurfe', 'pvoc_inceptionv3_sliceblurfe', 'pvoc_resnet50_sliceblurfe']
baylime_keys = ['oxpets_inceptionv3_baylime', 'oxpets_resnet50_baylime', 'pvoc_inceptionv3_baylime', 'pvoc_resnet50_baylime']
lime_keys = ['oxpets_inceptionv3_lime', 'oxpets_resnet50_lime', 'pvoc_inceptionv3_lime', 'pvoc_resnet50_lime']

# Assuming `aopc_dict` contains your data
dict_analysis = aopc_dict
key_index = 0
handles_list = []
labels_list = []

# Define a function to replace NaN values with the median of the data
def replace_nan_with_median(data):
    if np.isnan(data).any():
        median_value = np.nanmedian(data)
        data = np.where(np.isnan(data), median_value, data)
    return data

# Define a capping function that applies stricter capping for ResNet50 models
def cap_values(data, lower_percentile, upper_percentile):
    lower_cap = np.percentile(data, lower_percentile)
    upper_cap = np.percentile(data, upper_percentile)
    return np.clip(data, lower_cap, upper_cap)

# Function to perform Wilcoxon Signed-Rank test with effect size on original data
def perform_wilcoxon_test(data1, data2, alternative):
    result = pg.wilcoxon(data1, data2, alternative=alternative)
    w_value = result['W-val'][0]
    p_value = result['p-val'][0]
    effect_size_rbc = result['RBC'][0]  # Rank-biserial correlation effect size
    effect_size_cles = result['CLES'][0]  # Common Language Effect Size
    return w_value, p_value, effect_size_rbc, effect_size_cles

# Plot ECDF
for ax in axes.ravel():
    # Replace NaNs in the original data with the median (for testing) and in capped data (for visualization)
    slice_data_original = replace_nan_with_median(np.array(dict_analysis[slice_keys[key_index]]))
    baylime_data_original = replace_nan_with_median(np.array(dict_analysis[baylime_keys[key_index]]))
    lime_data_original = replace_nan_with_median(np.array(dict_analysis[lime_keys[key_index]]))

    # Apply capping for visualization
    if 'ResNet50' in title_index[key_index]:
        lower_cap, upper_cap = (0, 100)  # Adjust these values as needed
    else:
        lower_cap, upper_cap = (0, 100)  # Adjust these values as needed

    # Capped data for ECDF visualization
    slice_data = cap_values(slice_data_original, lower_cap, upper_cap)
    baylime_data = cap_values(baylime_data_original, lower_cap, upper_cap)
    lime_data = cap_values(lime_data_original, lower_cap, upper_cap)

    # ECDF plots for each explanation method
    sns.ecdfplot(slice_data, color='blue', linewidth=1.5, ax=ax, label='SLICE')
    sns.ecdfplot(baylime_data, color='teal', linewidth=1.5, ax=ax, label='BayLIME')
    sns.ecdfplot(lime_data, color='purple', linewidth=1.5, ax=ax, label='LIME')

    # Set the title for each subplot
    ax.set_title(f"{title_index[key_index]}", fontsize=12, fontweight='bold')
    ax.set_xlabel("AOPC Insertion Scores", fontsize=12, fontweight='bold')
    ax.set_ylabel("Cumulative Probability", fontsize=12, fontweight='bold')

    # Collect handles and labels for creating a single legend later
    handles, labels = ax.get_legend_handles_labels()
    handles_list.extend(handles)
    labels_list.extend(labels)

    key_index += 1

# Display a unique legend for all subplots
unique = [(h, l) for i, (h, l) in enumerate(zip(handles_list, labels_list)) if l not in labels_list[:i]]
fig.legend(*zip(*unique), loc='upper center', bbox_to_anchor=(0.5, 1.05), fontsize=12, ncol=4)

# Adjust layout and save the figure
plt.tight_layout()
plt.savefig('plots/aopc_del.pdf', format='pdf', dpi=300, bbox_inches='tight')
plt.show()