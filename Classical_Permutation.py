import pandas as pd
# import numpy as np
from xgboost import XGBClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score
from tools import DataManager
import matplotlib.pyplot as plt
# import seaborn as sns
import shap

if __name__ == '__main__':
    K_fold = 5
    min_prefix_length = 2
    All_prefixes = False
    max_prefix_length = None
    Binary_Existence = True
    Multi_activity = True
    frequency_threshold = 1
    results_address = "./datasets/bpic2012_3"
    address = "./datasets/bpic2012_O_DECLINED_Trunc40.csv"
    frequent_itemsets_address = "/location_importance_bpic2012_O_DECLINED_Trunc40_Multi.csv"

    folder_name = results_address.split('/')[-1]
    data_name = address.split('/')[-1].split('.')[0]
    data_manager = DataManager(address, min_prefix_length,
                               max_prefix_length, frequency_threshold)

    if Multi_activity:
        candidate_activities_list = pd.read_csv(results_address + frequent_itemsets_address).columns.tolist()
        candidate_activities_dict = {candidate_activities_list[i]: i for i in range(len(candidate_activities_list))}
        reverse_candidate_activities = {i: candidate_activities_list[i] for i in range(len(candidate_activities_list))}
        # if All_prefixes:
        #     prefixed_data = data_manager.prefix_generator()
        #     if Binary_Existence:
        #         encoded_data = data_manager.binary_encoding(prefixed_data)
        #     else:
        #         encoded_data = data_manager.frequency_encoding(prefixed_data)
        # else:
        #     if Binary_Existence:
        #         encoded_data = data_manager.binary_encoding(data_manager.data)
        #     else:
        #         encoded_data = data_manager.frequency_encoding(data_manager.data)

        data_manager.data['trace'] = data_manager.data.groupby(data_manager.case_id)[data_manager.activity].transform(
            lambda x: '->'.join(x))
        data_manager.data.drop_duplicates(subset=data_manager.case_id, keep='first', inplace=True)

        encoded_data = []
        for case in data_manager.data[data_manager.case_id]:
            case_data = data_manager.data[data_manager.data[data_manager.case_id] == case]
            encoded_case = {data_manager.case_id: case,
                            data_manager.outcome: case_data[data_manager.outcome].tolist()[0]}

            trace = case_data['trace'].tolist()[0]
            for item in candidate_activities_list:
                set_of_items = set(eval(item.strip("'")))
                item_freq = min(trace.count(act) for act in set_of_items)
                encoded_case[candidate_activities_dict[item]] = item_freq

            encoded_data.append(encoded_case)

        encoded_data = pd.DataFrame(encoded_data)

    else:
        if All_prefixes:
            prefixed_data = data_manager.prefix_generator()
            if Binary_Existence:
                encoded_data = data_manager.binary_encoding(prefixed_data)
            else:
                encoded_data = data_manager.frequency_encoding(prefixed_data)
        else:
            if Binary_Existence:
                encoded_data = data_manager.binary_encoding(data_manager.data)
            else:
                encoded_data = data_manager.frequency_encoding(data_manager.data)


    train_list, test_list = data_manager.cross_split_test_train(K_fold)
    permutation_importance_results = dict()
    SHAP_values = dict()
    for i in range(K_fold):
        train_x = encoded_data[encoded_data[data_manager.case_id].isin(train_list[i])]
        train_y = train_x[data_manager.outcome].tolist()
        train_x.drop([data_manager.case_id, data_manager.outcome], axis=1, inplace=True)

        test_x = encoded_data[encoded_data[data_manager.case_id].isin(test_list[i])]
        test_y = test_x[data_manager.outcome].tolist()
        test_x.drop([data_manager.case_id, data_manager.outcome], axis=1, inplace=True)


        model = XGBClassifier()
        model.fit(train_x, train_y)
        predicted = model.predict(test_x)
        # probabilities = np.mean(model.predict_proba(test_x)[:, 1])

        print("f1-score test: %.3f" % f1_score(test_y, predicted, average='weighted'))

        # calculate permutation importance for training data
        SHAP_values[i] = shap.TreeExplainer(model).shap_values(train_x)
        # mean absolute value of the SHAP values for each feature
        # SHAP_values[i] = SHAP_values[i]
        SHAP_values[i] = pd.DataFrame(SHAP_values[i])
        SHAP_values[i].columns = train_x.columns
        SHAP_values[i] = SHAP_values[i].abs()
        SHAP_values[i] = SHAP_values[i].mean().sort_values(ascending=False)
        SHAP_values[i] = pd.DataFrame(SHAP_values[i]).transpose()

        result_train = permutation_importance(
            model, train_x, train_y, n_repeats=20, random_state=42, n_jobs=2)

        # sorted_importances_idx_train = result_train.importances_mean.argsort()
        permutation_importance_results[i] = pd.DataFrame(
            result_train.importances.T,
            columns=train_x.columns,
        )

    permutation_importance_all = pd.concat(
        [permutation_importance_results[p] for p in permutation_importance_results])

    permutation_importance_all = permutation_importance_all.reindex(permutation_importance_all.mean().sort_values(
        ascending=False).index, axis=1)
    if Multi_activity:
        permutation_importance_all.rename(columns={x: reverse_candidate_activities[x] for x in reverse_candidate_activities},
                                          inplace=True)

    SHAP_values_all = pd.concat([SHAP_values[p] for p in SHAP_values])
    SHAP_values_all = SHAP_values_all.reindex(SHAP_values_all.mean().sort_values(ascending=False).index, axis=1)
    if Multi_activity:
        SHAP_values_all.rename(columns={x: reverse_candidate_activities[x] for x in reverse_candidate_activities},
                               inplace=True)
    # save the importance
    if Multi_activity:
        permutation_importance_all.to_csv(results_address + '/Importances_Classical_Permutation_Multi_%s.csv' % data_name, index=False)
        SHAP_values_all.to_csv(results_address + '/Importances_Classical_SHAP_Multi_%s.csv' % data_name, index=False)
    else:
        permutation_importance_all.to_csv(results_address + '/Importances_Classical_Permutation_%s.csv' % data_name, index=False)
        SHAP_values_all.to_csv(results_address + '/Importances_Classical_SHAP_%s_Prefix.csv' % data_name, index=False)

    f, axs = plt.subplots(1, 1, figsize=(15, 10))
    permutation_importance_all.plot.box(vert=False, whis=10, ax=axs)
    axs.set_title("Classical Permutation Importance_Prefix_Binary Encoding (%s_fold cross validation)" % K_fold)
    axs.axvline(x=0, color="k", linestyle="--")
    axs.set_xlabel("Decrease in f1 score")
    axs.figure.tight_layout()
    if Multi_activity:
        plt.savefig(results_address + '/Importance_Classical_Permutation_Multi_%s.png' % data_name)
    else:
        plt.savefig(results_address + '/Importance_Classical_Permutation_%s.png' % data_name)

    f.show()
