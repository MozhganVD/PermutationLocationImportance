import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score, accuracy_score
from tools import DataManager
import matplotlib.pyplot as plt


if __name__ == '__main__':
    min_prefix_length = 2
    max_prefix_length = None
    constrain = False
    data_manager = DataManager("./datasets/AllEpisode_1517Patients_08012023_preprocessed_V2.csv", min_prefix_length, max_prefix_length)

    encoded_data = data_manager.frequency_encoding()
    # remove columns with same value for all rows
    ToRemoveActs = list(set(encoded_data.columns).difference(
        set(encoded_data.loc[:, (encoded_data != encoded_data.iloc[0]).any()].columns)))

    # calculate the correlation matrix for the encoded data
    corr_matrix = encoded_data.corr()
    HighCorrelActs = dict()
    for act in corr_matrix:
        high_correl_list = corr_matrix.loc[corr_matrix[act] > 0.7, act].index.tolist()
        if act in high_correl_list:
            high_correl_list.remove(act)
        if len(high_correl_list) > 0:
            HighCorrelActs[act] = high_correl_list

    to_keep_acts = []
    indexing_info = data_manager.indexing_info
    for act in HighCorrelActs:
        correlated_list = [act]
        correlated_list.extend(HighCorrelActs[act])
        indexStd_max = indexing_info[indexing_info[data_manager.activity].isin(correlated_list)]['std'].idxmax()
        to_keep_acts.append(indexing_info.at[indexStd_max, data_manager.activity])

    to_keep_acts = set(np.unique(to_keep_acts))
    all_highcorrelacts = set(HighCorrelActs.keys())
    ToRemoveActs.extend(list(all_highcorrelacts.difference(to_keep_acts)))

    # remove rows with a certain value in activity column
    data_manager.data = data_manager.data[~data_manager.data[
        data_manager.activity].isin(ToRemoveActs)]

    encoded_data = data_manager.index_encoding(data_manager.data)

    train_list, test_list = data_manager.split_test_train(0.8)
    train_x = encoded_data[encoded_data[data_manager.case_id].isin(train_list)]
    train_y = train_x[data_manager.outcome].tolist()
    train_x.drop([data_manager.case_id, data_manager.activity, data_manager.time_col, data_manager.outcome],
                 axis=1, inplace=True)

    test_x = encoded_data[encoded_data[data_manager.case_id].isin(test_list)]
    test_y = test_x[data_manager.outcome].tolist()
    test_x.drop([data_manager.case_id, data_manager.activity, data_manager.time_col, data_manager.outcome],
                axis=1, inplace=True)

    model = XGBClassifier()
    model.fit(train_x, train_y)
    predicted = model.predict(test_x)

    print("AUC test: %.3f" % accuracy_score(test_y, predicted))

    location_importance_test = data_manager.trace_permutation_importance(model, test_x, test_y, test_list,
                                                                         constrain=constrain)
    # save the results to a csv file
    location_importance_test.to_csv('location_importance_test.csv', index=False)

    location_importance_train = data_manager.trace_permutation_importance(model, train_x, train_y, train_list,
                                                                          constrain=constrain)
    # save the results to a csv file
    location_importance_train.to_csv('location_importance_train.csv', index=False)

    # plot the results
    f, axs = plt.subplots(1, 2, figsize=(15, 5))
    location_importance_test.plot.box(vert=False, whis=10, ax=axs[0])
    axs[0].set_title("Permutation Importances (test set)")
    axs[0].axvline(x=0, color="k", linestyle="--")
    axs[0].set_xlabel("Decrease in accuracy score")
    axs[0].figure.tight_layout()

    location_importance_train.plot.box(vert=False, whis=10, ax=axs[1])
    axs[1].set_title("Permutation Importances (train set)")
    axs[1].axvline(x=0, color="k", linestyle="--")
    axs[1].set_xlabel("Decrease in accuracy score")
    axs[1].figure.tight_layout()

    f.show()


    # # remove columns with same value in all rows in location_importance
    # location_importance_test = location_importance_test.loc[:,
    #                            (location_importance_test != location_importance_test.iloc[0]).any()]
    #
    # location_importance_train = location_importance_train.loc[:,
    #                            (location_importance_train != location_importance_train.iloc[0]).any()]
    #
    # # plot the results
    # f, axs = plt.subplots(1, 2, figsize=(15, 5))
    # location_importance_test.plot.box(vert=False, whis=10, ax=axs[0])
    # axs[0].set_title("Permutation Importances (test set)")
    # axs[0].axvline(x=0, color="k", linestyle="--")
    # axs[0].set_xlabel("Decrease in accuracy score")
    # axs[0].figure.tight_layout()
    #
    # location_importance_train.plot.box(vert=False, whis=10, ax=axs[1])
    # axs[1].set_title("Permutation Importances (train set)")
    # axs[1].axvline(x=0, color="k", linestyle="--")
    # axs[1].set_xlabel("Decrease in accuracy score")
    # axs[1].figure.tight_layout()
    #
    # f.show()

    print('done!')
