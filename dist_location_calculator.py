import pandas as pd
from xgboost import XGBClassifier
# from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score
from tools import DataManager
import matplotlib.pyplot as plt
from statistics import mode


K_fold = 5
Important_number= 10
min_prefix_length = 2
L_max_perc = 0.8
max_prefix_length = None
constrain = True
Binary_Existence = True
Multi_activity = False
case_id = "case:concept:name"
activity = "concept:name"
time_col = "time:timestamp"
outcome = "label"
dataset_address = "./datasets/BPIC11_f1_trunc36.csv"
results_address = './datasets/BPIC2011_f1'
location_importance_address = '/Multi/location_importance_BPIC11_f1_trunc36_Multi.csv'

folder_name = results_address.split('/')[-1]
data_name = dataset_address.split('/')[-1].split('.')[0]
data_manager = DataManager(dataset_address, min_prefix_length,
                           max_prefix_length, L_max_perc=L_max_perc)


## load data set and preprocess
df = pd.read_csv(dataset_address, sep=",")
df[case_id] = df[case_id].astype(str)
df[activity] = df[activity].str.lower()
if 'lifecycle:transition' in df.columns:
    df[activity] = df[activity].map(lambda x: x.split('-')[0])
    df = df[df['lifecycle:transition'] == 'COMPLETE']
    df.sort_values([case_id, 'event_nr'], ascending=[True, True], inplace=True)
    df['event_nr'] = df.groupby([case_id]).cumcount() + 1
df[activity] = df[activity].str.replace(" ", "")
df[activity] = df[activity].str.replace("-", "")
df[activity] = df[activity].str.replace("_", "")
df[time_col] = df[time_col].str.replace("/", "-")
# df[time_col] = pd.to_datetime(df[time_col],
#                               dayfirst=True).map(lambda x: x.strftime("%Y.%m.%d %H:%M:%S"))
# df.sort_values([case_id, time_col], ascending=[True, True], inplace=True)
df.loc[df[outcome] == "deviant", outcome] = 1
df.loc[df[outcome] == "regular", outcome] = 0

location_importance = pd.read_csv(results_address + location_importance_address)
most_important_items = location_importance.columns.tolist()[:Important_number]

######## for multi activity ##########
for fe in most_important_items:
    filtered_cases = []
    fe = set(eval(fe.strip("'")))
    event_counts = pd.DataFrame(columns=['label', 'locations'])
    for case in df[case_id].unique():
        trace_case = df[df[case_id] == case].sort_values(by=['event_nr'])
        trace = trace_case[activity].tolist()
        if fe.issubset(set(trace)):
            if len(fe) == len(trace):
                continue
            filtered_cases.append(case)

    filtered_df = df[df[case_id].isin(filtered_cases)]
    filtered_df = filtered_df[filtered_df[activity].isin(fe)]

    All_locations = []
    Labels = []
    for f_case in filtered_cases:
        locations = []
        for act in fe:
            locations.append(filtered_df[(filtered_df[activity] == act) & (filtered_df[case_id] == f_case)]['event_nr'].to_list()[0])
        All_locations.append(str(locations))
        Labels.append(filtered_df[(filtered_df[case_id] == f_case)].iloc[0]['label'])

    event_counts['locations'] = All_locations
    event_counts['label'] = Labels
    # event_counts = filtered_df.groupby(['label', 'event_nr']).size().reset_index(name='count')
    # Pivot the data to have 'event_nr' as columns
    # pivot_table = event_counts.pivot(index='event_nr', columns='label', values='count')
    # pivot_table['frac'] = pivot_table[1] / (pivot_table[0] + pivot_table[1])
    # # keep only two digits after the decimal point
    # pivot_table['frac'] = pivot_table['frac'].map(lambda x: round(x, 2))
    # pivot_table = pivot_table.fillna(0)
    # Plot the distribution
    event_counts = event_counts.groupby(['locations', 'label']).size().reset_index(name='count')
    event_counts['fristloc'] = event_counts['locations'].map(lambda x: int(x.split(',')[0].split('[')[1]))
    event_counts_2 = event_counts.groupby(['fristloc', 'label'], as_index=False)['count'].sum()
    pivot_table = event_counts_2.pivot(index='fristloc', columns='label', values='count')
    pivot_table = pivot_table.fillna(0)
    pivot_table.plot(kind='bar', stacked=True)
    plt.xlabel('Locations')
    # remove the x tickes
    plt.ylabel('Count')
    plt.title('Itemset %s' % fe)
    plt.legend(labels=['positive', 'negative'])
    plt.savefig(results_address + '/dist_Loc/dist_%s_0002.png' % fe)
    plt.show()

# for fe in most_important_items:
#     filtered_df = df[df[activity] == fe]
#     event_counts = filtered_df.groupby(['label', 'event_nr']).size().reset_index(name='count')
#     # Pivot the data to have 'event_nr' as columns
#     pivot_table = event_counts.pivot(index='event_nr', columns='label', values='count')
#     # pivot_table['frac'] = pivot_table[1] / (pivot_table[0] + pivot_table[1])
#     # # keep only two digits after the decimal point
#     # pivot_table['frac'] = pivot_table['frac'].map(lambda x: round(x, 2))
#     pivot_table = pivot_table.fillna(0)
#     # Plot the distribution
#     pivot_table.plot(kind='bar', stacked=True)
#     plt.xlabel('Locations')
#     plt.ylabel('Count')
#     plt.title('Activity %s' % fe)
#     plt.legend(labels=['positive', 'negative'])
#     plt.savefig(results_address + '/dist_Loc/dist_%s.png' % fe)
#     plt.show()

