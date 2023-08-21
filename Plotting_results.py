import matplotlib.pyplot as plt
import pandas as pd
import glob
import numpy as np


def shorten_label(labels, max_length=13):
    adjusted_labels = []
    for label in labels:
        if len(label) > max_length:
            adjusted_labels.append(str(label[:max_length-3]) + "...")
        else:
            adjusted_labels.append(label)
    return adjusted_labels


Results_directory = './datasets/Itemset_importance/'
Top_important_features = 20

f, axs = plt.subplots(3, 3, figsize=[30, 20])

all_files = glob.glob(Results_directory + "*.csv")

row = 0
col = 0
for filename in all_files:
    data_name = filename.split('\\')[-1].split('_')[0]
    importance_type = filename.split('\\')[-1].split('_')[1].split('.')[0]
    if importance_type != 'LI':
        continue
    LI_df = pd.read_csv(filename)
    LI_df = LI_df.iloc[:, :Top_important_features]
    EI_df = pd.read_csv(Results_directory + data_name + '_EI.csv')
    EI_df = EI_df[LI_df.columns.tolist()]
    EI_df = EI_df.reindex(LI_df.columns.tolist(), axis=1)
    # all_data = pd.concat([LI_df, EI_df])

    LI_values = []
    EI_values = []
    for fe in LI_df.columns:
        LI_values.append(LI_df[fe].tolist())
        EI_values.append(EI_df[fe].tolist())

    li_plot = axs[row, col].boxplot(LI_values, positions=np.array(np.arange(len(LI_values))) * 2.0 - 0.35,
                                    widths=0.6, showfliers=False,
                                    vert=False, patch_artist=True, boxprops=dict(facecolor="green"),
                                    medianprops=dict(color="#00C957"))
    ei_plot = axs[row, col].boxplot(EI_values, positions=np.array(np.arange(len(EI_values))) * 2.0 + 0.35,
                                    widths=0.6, showfliers=False,
                                    vert=False, patch_artist=True, boxprops=dict(facecolor="blue"),
                                    medianprops=dict(color="#1E90FF"))

    # Set y ticks and labels
    Adjusted_labels = shorten_label(LI_df.columns.tolist())
    axs[row, col].set_yticks(np.arange(0, len(LI_df.columns.tolist()) * 2, 2), Adjusted_labels)
    axs[row, col].set_title(data_name)
    # axs[row, col].set_yticklabels(LI_df.columns.tolist())

    if col < 2:
        col += 1
    else:
        col = 0
        row += 1

# Set common x and y labels
f.text(0.5, 0.08, 'Decrease in f1-score', ha='center', fontsize=14)
f.text(0.07, 0.5, 'Itemsets', va='center', rotation='vertical', fontsize=14)
f.legend([li_plot["boxes"][0], ei_plot["boxes"][0]], ['Itemsets location importance', 'Itemsets existence importance'],
         loc='upper center', ncol=2, fontsize=14)
plt.savefig(Results_directory + "Location_Existence_Importance.png")
#
f.show()


