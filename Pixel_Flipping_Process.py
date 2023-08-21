# import numpy as np
import pandas as pd
from xgboost import XGBClassifier
# from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score
from tools import DataManager
import matplotlib.pyplot as plt
from statistics import mode

if __name__ == '__main__':
    K_fold = 5
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
    dataset_address = "./datasets/BPIC11_f2_trunc40.csv"
    results_address = './datasets/bpic2011_f2'
    location_importance_address = '/location_importance_BPIC11_f2_trunc40_Prefixes.csv'
    Shap_address = '/Importances_Classical_SHAP_BPIC11_f2_trunc40_Prefix.csv'
    classical_Permutation_address = '/Importances_Classical_Permutation_BPIC11_f2_trunc40_Prefix.csv'

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

    df[activity] = df[activity].str.replace(" ", "")
    df[activity] = df[activity].str.replace("-", "")
    df[activity] = df[activity].str.replace("_", "")
    df[time_col] = df[time_col].str.replace("/", "-")
    df[time_col] = pd.to_datetime(df[time_col],
                                  dayfirst=True).map(lambda x: x.strftime("%Y.%m.%d %H:%M:%S"))

    df.sort_values([case_id, time_col], ascending=[True, True], inplace=True)

    df.loc[df[outcome] == "deviant", outcome] = 1
    df.loc[df[outcome] == "regular", outcome] = 0

    ## importing the itemset with the most location importance from the previous step
    location_importance = pd.read_csv(results_address + location_importance_address)
    most_important_items = location_importance.columns.tolist()

    itemsets_neutral_location = dict()
    observed_locations = dict()
    for item in most_important_items:
        if Multi_activity:
            set_of_items = item
            if type(set_of_items) is str:
                set_of_items = eval(item.strip("'"))

            itemsets_neutral_location[item] = dict()
            # find rows in df that contain the item
            # TODO: find a way to do this without looping over the df
            for act in set_of_items:
                observed_locations[act] = []
            # find cases with all acts in item
            for case in df[case_id].unique():
                if set(set_of_items).issubset(set(df[df[case_id] == case][activity].tolist())):
                    for act in set_of_items:
                        observed_locations[act].append(
                            df[(df[case_id] == case) & (df[activity] == act)]['event_nr'].tolist()[0])

            for act in set_of_items:
                itemsets_neutral_location[item][act] = mode(observed_locations[act])

        else:
            itemsets_neutral_location[item] = mode(df[df[activity] == item]['event_nr'].tolist())

    prefixed_data = data_manager.prefix_generator()
    encoded_data = data_manager.index_encoding(prefixed_data)
    train_list, test_list = data_manager.cross_split_test_train(K_fold)

    Pixel_Flipping = []

    if Multi_activity:
        candidate_activities_dict = {most_important_items[i]: i for i in range(len(most_important_items))}
        reverse_candidate_activities = {i: most_important_items[i] for i in range(len(most_important_items))}

        prefixed_data = data_manager.prefix_generator()
        if Binary_Existence:
            tabular_encoded_data = data_manager.binary_encoding(prefixed_data)
        else:
            tabular_encoded_data = data_manager.frequency_encoding(prefixed_data)

        # data_manager.data['trace'] = data_manager.data.groupby(data_manager.case_id)[data_manager.activity].transform(
        #     lambda x: '->'.join(x))
        # data_manager.data.drop_duplicates(subset=data_manager.case_id, keep='first', inplace=True)
        # tabular_encoded_data = []
        # for case in data_manager.data[data_manager.case_id]:
        #     case_data = data_manager.data[data_manager.data[data_manager.case_id] == case]
        #     encoded_case = {data_manager.case_id: case,
        #                     data_manager.outcome: case_data[data_manager.outcome].tolist()[0]}
        #
        #     trace = case_data['trace'].tolist()[0]
        #     for item in most_important_items:
        #         set_of_items = set(eval(item.strip("'")))
        #         item_freq = min(trace.count(act) for act in set_of_items)
        #         encoded_case[candidate_activities_dict[item]] = item_freq
        #
        #     tabular_encoded_data.append(encoded_case)
        #
        # tabular_encoded_data = pd.DataFrame(tabular_encoded_data)

    else:
        prefixed_data = data_manager.prefix_generator()
        if Binary_Existence:
            tabular_encoded_data = data_manager.binary_encoding(prefixed_data)
        else:
            tabular_encoded_data = data_manager.frequency_encoding(prefixed_data)

    # import the results of classical permutation importance
    classic_permutation_importance = pd.read_csv(results_address + classical_Permutation_address)
    Tabular_Flipping = []
    # import the results of SHAP importance
    SHAP_permutation_importance = pd.read_csv(results_address + Shap_address)
    Tabular_Flipping_SHAP = []

    # for item in most_important_items:
    for i in range(K_fold):
        # pixel flipping for tabular data
        t_train_x = tabular_encoded_data[tabular_encoded_data[data_manager.case_id].isin(train_list[i])]
        t_train_y = t_train_x[data_manager.outcome].tolist()
        t_train_x.drop(
            [data_manager.case_id, data_manager.outcome],
            axis=1, inplace=True)

        t_test_x = tabular_encoded_data[tabular_encoded_data[data_manager.case_id].isin(test_list[i])]
        t_test_y = t_test_x[data_manager.outcome].tolist()
        t_test_x.drop(
            [data_manager.case_id, data_manager.outcome],
            axis=1, inplace=True)

        T_model = XGBClassifier()
        T_model.fit(t_train_x, t_train_y)

        # pixel flipping for location data
        train_x = encoded_data[encoded_data[data_manager.case_id].isin(train_list[i])]
        train_y = train_x[data_manager.outcome].tolist()
        train_x.drop([data_manager.case_id, data_manager.outcome],
                     axis=1, inplace=True)

        test_x = encoded_data[encoded_data[data_manager.case_id].isin(test_list[i])]
        test_y = test_x[data_manager.outcome].tolist()
        test_x.drop([data_manager.case_id, data_manager.outcome],
                    axis=1, inplace=True)

        model = XGBClassifier()
        model.fit(train_x, train_y)

        for item in most_important_items:
            location_peritem = {'rank': location_importance.columns.to_list().index(item),
                                'itemset': item}

            shap_peritem = {'rank': SHAP_permutation_importance.columns.to_list().index(item),
                            'itemset': item}

            permutation_peritem = {'rank': classic_permutation_importance.columns.to_list().index(item),
                                   'itemset': item}

            # put the mode of value in each column of tabular data
            t_train_x_Corrupted = t_train_x.copy()
            if Multi_activity:
                set_of_items = item
                if type(set_of_items) is str:
                    set_of_items = eval(item.strip("'"))
                sub_corrupted_data = t_train_x_Corrupted[t_train_x_Corrupted[set_of_items] > 0]
                for act in set_of_items:
                    sub_corrupted_data[act] = mode(t_train_x_Corrupted[act])
            else:
                t_train_x_Corrupted[item] = mode(t_train_x_Corrupted[item])

            t_corrupted_predicted = T_model.predict(t_train_x_Corrupted)
            t_corrupted_f1 = f1_score(t_train_y, t_corrupted_predicted, average='weighted')
            permutation_peritem['fscore'] = t_corrupted_f1
            shap_peritem['fscore'] = t_corrupted_f1

            # mode location data
            sub_data = df[df[case_id].isin(train_list[i])]
            shuffled_cases = []
            for case in train_list[i]:
                trace_case = sub_data[sub_data[case_id] == case].sort_values(by=[time_col])
                trace = trace_case[activity].tolist()
                # check if the itemset is in the trace
                if Multi_activity:
                    set_of_items = item
                    if type(set_of_items) is str:
                        set_of_items = set(eval(item.strip("'")))
                    elif type(set_of_items) is list:
                        set_of_items = set(item)
                else:
                    set_of_items = {item}

                if set_of_items.issubset(set(trace)):
                    if len(set_of_items) == len(trace):
                        continue
                    shuffled_cases.append(case)
                    shuffled_sequence = trace.copy()
                    if Multi_activity:
                        for act in set_of_items:
                            shuffled_sequence.insert(itemsets_neutral_location[item][act] - 1,
                                                     shuffled_sequence.pop(trace.index(act)))
                    else:
                        shuffled_sequence.insert(itemsets_neutral_location[item] - 1,
                                                 shuffled_sequence.pop(trace.index(item)))

                    sub_data.loc[sub_data[case_id] == case, activity] = shuffled_sequence

            # encoded only the shuffled cases
            if len(shuffled_cases) > 0:
                X_corrupt = data_manager.index_encoding(sub_data[sub_data[case_id].isin(shuffled_cases)])
                zero_features = [x for x in train_x.columns if x not in X_corrupt.columns]
                X_corrupt[zero_features] = 0
                X_corrupt.drop([case_id, outcome], axis=1, inplace=True)
                X_corrupt = X_corrupt[train_x.columns]
                Shuffled_X = train_x.copy()
                Shuffled_X.loc[Shuffled_X.index.isin(X_corrupt.index)] = X_corrupt
                # calculate the importance
                Corrupt_predicted = model.predict(Shuffled_X)
                Corrupt_f1 = f1_score(train_y, Corrupt_predicted, average='weighted')
                location_peritem['fscore'] = Corrupt_f1
            else:
                predicted = model.predict(train_x)
                location_peritem['fscore'] = f1_score(train_y, predicted, average='weighted')
                print(item, ': nothing shuffled')
            # importance = baseline - Corrupt_f1


            # Pixel_Flipping.append({'itemset': item, i+1: Corrupt_f1})
            # Pixel_Flipping.loc[Pixel_Flipping['itemset'] == item, i + 1] = Corrupt_f1

            Tabular_Flipping.append(permutation_peritem)
            Tabular_Flipping_SHAP.append(shap_peritem)

            Pixel_Flipping.append(location_peritem)

    # Calculate average and standard deviation
    Pixel_Flipping = pd.DataFrame(Pixel_Flipping)
    Pixel_Flipping = Pixel_Flipping.groupby(['itemset', 'rank']).mean()
    Pixel_Flipping = Pixel_Flipping.reset_index()
    # Pixel_Flipping['Average'] = Pixel_Flipping.iloc[:, 1:].mean(axis=1)
    # Pixel_Flipping['Std'] = Pixel_Flipping.iloc[:, 1:].std(axis=1)
    Pixel_Flipping.sort_values(by=['rank'], ascending=True, inplace=True)

    if Multi_activity:
        Pixel_Flipping.to_csv(results_address + "/location_pixel_fliping_%s_Multi.csv" % folder_name, index=False)
    else:
        Pixel_Flipping.to_csv(results_address + "/location_pixel_fliping_%s_Prefix.csv" % folder_name, index=False)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(Pixel_Flipping['rank'], Pixel_Flipping['fscore'], color='blue', linewidth=2,
             label='Location importance')
    # if not Multi_activity:
    Tabular_Flipping = pd.DataFrame(Tabular_Flipping)
    Tabular_Flipping = Tabular_Flipping.groupby(['itemset', 'rank']).mean()
    Tabular_Flipping = Tabular_Flipping.reset_index()
    # Tabular_Flipping['Average'] = Tabular_Flipping.iloc[:, 1:].mean(axis=1)
    Tabular_Flipping.sort_values(by=['rank'], ascending=True, inplace=True)
    plt.plot(Tabular_Flipping['rank'], Tabular_Flipping['fscore'], color='red', linewidth=2,
             label='Permutation existence importance')

    Tabular_Flipping_SHAP = pd.DataFrame(Tabular_Flipping_SHAP)
    Tabular_Flipping_SHAP = Tabular_Flipping_SHAP.groupby(['itemset', 'rank']).mean()
    Tabular_Flipping_SHAP = Tabular_Flipping_SHAP.reset_index()
    # Tabular_Flipping_SHAP['Average'] = Tabular_Flipping_SHAP.iloc[:, 1:].mean(axis=1)
    Tabular_Flipping_SHAP.sort_values(by=['rank'], ascending=True, inplace=True)
    plt.plot(Tabular_Flipping_SHAP['rank'], Tabular_Flipping_SHAP['fscore'], color='green', linewidth=2,
             label='SHAP existence importance')

    if Multi_activity:
        Tabular_Flipping.to_csv(results_address + "/tabular_pixel_flipping_%s_Multi.csv" % folder_name, index=False)
        Tabular_Flipping_SHAP.to_csv(results_address + "/tabular_pixel_flipping_SHAP_%s_Multi.csv" % folder_name, index=False)
    else:
        Tabular_Flipping.to_csv(results_address + "/tabular_pixel_flipping_%s_Prefix.csv" % folder_name, index=False)
        Tabular_Flipping_SHAP.to_csv(results_address + "/tabular_pixel_flipping_SHAP_%s_Prefix.csv" % folder_name, index=False)

    # plt.xticks(Tabular_Flipping_SHAP['rank'])
    plt.ylabel('Average Prediction Score')
    plt.xlabel('Feature Importance Ranking')
    plt.title('Pixel Flipping _ All Prefixes_%s' % folder_name)
    plt.legend()
    # save the figure of the pixel flipping
    if Multi_activity:
        plt.savefig(results_address + '/Pixel_Flipping_%s_Multi.png' % folder_name)
    else:
        plt.savefig(results_address + '/Pixel_Flipping_%s_Prefix.png' % folder_name)
    plt.show()
