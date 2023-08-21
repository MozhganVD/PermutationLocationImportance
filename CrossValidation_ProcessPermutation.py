import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
from tools import DataManager
import matplotlib.pyplot as plt
import argparse


parser = argparse.ArgumentParser(description="Location Permutation Importance")

parser.add_argument("--address",
                    type=str,
                    default="./datasets/bpic2012_1_trunc40.csv",
                    help="path to the dataset")

parser.add_argument("--constrain",
                    type=bool,
                    default=True,
                    help="True if you want to avoid generating unrealistic trace after permutation")

parser.add_argument("--Multi_activity",
                    type=bool,
                    default=True,
                    help="True if you want to consider multi activity in itemsets, "
                         "False if you want to consider single activity")

parser.add_argument('--top_k', default=10, type=float)
parser.add_argument('--min_support', default=0.5, type=float)

args = parser.parse_args()

if __name__ == '__main__':
    min_support = args.min_support
    top_k = args.top_k
    Multi_activity = args.Multi_activity
    All_prefixes = False
    candidate_selection_method = 'apriori' # optimized , apriori
    L_max_perc = 0.8
    K_fold = 5
    min_prefix_length = 2
    max_prefix_length = None
    constrain = args.constrain
    address = args.address
    data_name = address.split('/')[-1].split('.')[0]
    data_manager = DataManager(address, min_prefix_length,
                               max_prefix_length, L_max_perc=L_max_perc)

    if Multi_activity:
        candidate_activities_list = None
        if candidate_selection_method == 'apriori':
            candidate_activities_list, itemsets = data_manager.frequent_activity_sets(min_support, top_k)
            print('%s frequent itemsets found' % len(candidate_activities_list))
            # create a dictionary assigning a code to each itemset of frequent_itemsets
        elif candidate_selection_method == 'optimized':
            candidate_activities_list = data_manager.optimized_activity_sets()
            print('%s Effective itemsets found' % len(candidate_activities_list))

        if candidate_activities_list is None:
            print('No candidate activities found')
            exit(0)
        candidate_itemsets = {}
        for i, itemset in enumerate(candidate_activities_list):
            candidate_itemsets[int(i)] = itemset

    if All_prefixes:
        prefixed_data = data_manager.prefix_generator()
        encoded_data = data_manager.index_encoding(prefixed_data)
    else:
        encoded_data = data_manager.index_encoding(data_manager.data)

    train_list, test_list = data_manager.cross_split_test_train(K_fold)

    permutation_importance_test_all = dict()
    permutation_importance_train_all = dict()

    for i in range(K_fold):
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
        predicted = model.predict(test_x)
        # probabilities = np.mean(model.predict_proba(test_x)[:, 1])

        print("f1-score test: %.3f" % f1_score(test_y, predicted, average='weighted'))

        if Multi_activity:
            permutation_importance_train_all[i] = data_manager.itemset_permutation_importance(model, train_x, train_y,
                                                                                              train_list[i],
                                                                                              candidate_itemsets,
                                                                                              constrain=constrain)


        else:
            permutation_importance_train_all[i] = data_manager.trace_permutation_importance(model, train_x, train_y,
                                                                                            train_list[i],
                                                                                            constrain=constrain)



    location_importance_train = pd.concat(
        [permutation_importance_train_all[p] for p in permutation_importance_train_all])
    location_importance_train = location_importance_train.reindex(location_importance_train.mean().sort_values(
        ascending=False).index, axis=1)

    if Multi_activity:
        yticklabels = []
        for label in location_importance_train.columns.tolist():
            yticklabels.append(candidate_itemsets[label])
            location_importance_train.rename(columns={label: str(candidate_itemsets[label])}, inplace=True)


    if Multi_activity:
        location_importance_train.to_csv('location_importance_%s_Multi.csv' % data_name, index=False)
    else:
        location_importance_train.to_csv('location_importance_%s.csv' % data_name, index=False)

    f, axs = plt.subplots(1, 1, figsize=(15, 10))
    location_importance_train.plot.box(vert=False, whis=10, ax=axs)
    axs.set_title("Location Permutation Importances (%s-fold cross-validation)" % K_fold)
    axs.axvline(x=0, color="k", linestyle="--")
    axs.set_xlabel("Decrease in f1 score")
    # set y tick labels

    if Multi_activity:
        axs.set_yticklabels(yticklabels)
    else:
        axs.set_yticklabels(location_importance_train.columns.tolist())

    axs.figure.tight_layout()
    # save figure
    if Multi_activity:
        plt.savefig('location_importance_%s_Multi.png' % data_name)
    else:
        plt.savefig('location_importance_%s.png' % data_name)
    f.show()
