import itertools
# from pm4py.objects.log.obj import EventLog
# import pm4py
# from matplotlib import pyplot as plt
# from pm4py.algo.filtering.log.variants import variants_filter
# from sklearn_genetic.plots import plot_fitness_evolution
# from xgboost import XGBClassifier
# import random
# from itertools import combinations
import numpy as np
import pandas as pd
# import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from mlxtend.frequent_patterns import apriori
# from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder
import itertools
# import datetime
# import time
# from sklearn_genetic import GAFeatureSelectionCV


class DataManager:
    def __init__(self, address, min_length, max_length=None, frq_threshold=2, L_max_perc=0.8):
        self.case_id = "case:concept:name"
        self.activity = "concept:name"
        self.time_col = "time:timestamp"
        self.outcome = "label"
        self.frq_threshold = frq_threshold
        self.L_max_perc = L_max_perc
        self.data, self.indexing_info = self._load_df(address)
        self.min_length = min_length
        self.max_length = max_length

    def _load_df(self, filepath):
        df = pd.read_csv(filepath, sep=",")
        df[self.case_id] = df[self.case_id].astype(str)
        df[self.activity] = df[self.activity].str.lower()

        if 'lifecycle:transition' in df.columns:
            df[self.activity] = df[self.activity].map(lambda x: x.split('-')[0])
            df = df[df['lifecycle:transition'] == 'COMPLETE']
            df.drop(['lifecycle:transition'], axis=1, inplace=True)
            df.sort_values([self.case_id, 'event_nr'], ascending=[True, True], inplace=True)
            df['event_nr'] = df.groupby([self.case_id]).cumcount() + 1

        df[self.activity] = df[self.activity].str.replace(" ", "")
        df[self.activity] = df[self.activity].str.replace("-", "")
        df[self.activity] = df[self.activity].str.replace("_", "")
        # df[self.time_col] = df[self.time_col].str.replace("/", "-")
        # df[self.time_col] = pd.to_datetime(df[self.time_col],
        #                                    dayfirst=True).map(lambda x: x.strftime("%Y.%m.%d %H:%M:%S"))

        df.sort_values([self.case_id, 'event_nr'], ascending=[True, True], inplace=True)

        df.loc[df[self.outcome] == "deviant", self.outcome] = 1
        df.loc[df[self.outcome] == "regular", self.outcome] = 0

        # remove cases contain very low frequent activities
        act_frequency = df.groupby(by=self.activity).agg({self.case_id: 'count'})
        low_frequent_acts = act_frequency[act_frequency[self.case_id] < self.frq_threshold].index.tolist()
        to_remove_cases = df.loc[df[self.activity].isin(low_frequent_acts), self.case_id].unique().tolist()
        df = df[~df[self.case_id].isin(to_remove_cases)]

        indexing_info = df.groupby(self.activity).agg({'event_nr': ['mean', 'std']})
        indexing_info.columns = ['mean', 'std']
        indexing_info.reset_index(inplace=True)

        self.Allowed_locations = dict()
        for act in df[self.activity].unique():
            self.Allowed_locations[act] = df.loc[df[self.activity] == act, 'event_nr'].unique().tolist()
            self.Allowed_locations[act].sort()

        self.L_max = np.quantile(df['event_nr'], self.L_max_perc)
        df = df[[self.case_id, self.activity, self.time_col, self.outcome, 'event_nr']]

        return df, indexing_info


    def prefix_generator(self):
        # generate prefixes for each case in the log
        prefixes = self.data.copy()
        for case in self.data[self.case_id].unique():
            case_data = self.data[self.data[self.case_id] == case].sort_values(by='event_nr')
            for i in range(2, len(self.data[self.data[self.case_id] == case])):
                case_prefix = pd.DataFrame({ self.case_id: str(case) + '_%s' % str(i),
                                             self.activity: case_data[self.activity].iloc[:i],
                                             self.outcome: case_data[self.outcome].iloc[0],
                                             self.time_col: case_data[self.time_col].iloc[0],
                                             'event_nr': case_data['event_nr'].iloc[:i]})
                prefixes = prefixes.append(case_prefix)

        prefixes = prefixes.reset_index(drop=True)
        return prefixes

    def optimized_activity_sets(self):
        pass
        # main_data = pm4py.format_dataframe(self.data, case_id=self.case_id,
        #                                    activity_key=self.activity, timestamp_key=self.time_col)
        # main_log = pm4py.convert_to_event_log(main_data)
        # variants = variants_filter.get_variants(main_log)
        # number_of_variants = len(variants)
        # pp_log = EventLog()
        # pp_log._attributes = main_log.attributes
        # for i, k in enumerate(variants):
        #     pp_log.append(variants[k][0])
        #
        # selected_variants = pm4py.convert_to_dataframe(pp_log)
        # n_grams = set()
        # selected_variants['case_length'] = selected_variants.groupby(self.case_id)[self.activity].transform(len)
        # for case in selected_variants[self.case_id].unique():
        #     window_length = int(
        #         min(selected_variants.loc[selected_variants[self.case_id] == case, 'case_length'].values[0], self.L_max))
        #     act = selected_variants[selected_variants[self.case_id] == case][self.activity].to_list()
        #     for i in range(0, len(act)):
        #         n_grams.add(frozenset({act[i]}))
        #         for j in range(i + 1, window_length + 1):
        #             itemset = set(act[i:j])
        #             n_grams.add(frozenset(itemset))
        #
        # training_data = self.data[[self.case_id, self.outcome]]
        # training_data.drop_duplicates([self.case_id], inplace=True)
        # training_data[self.case_id] = training_data[self.case_id].transform(str)
        # for n in n_grams:
        #     training_data[str(set(n))] = 0
        #
        # # for case in training_data[self.case_id]:
        # #     trace = set(main_data.loc[main_data[self.case_id] == case, self.activity].tolist())
        # #     training_data.loc[training_data[self.case_id] == case, 'trace'] = trace
        # for case in training_data[self.case_id]:
        #     trace = set(main_data.loc[main_data[self.case_id] == case, self.activity].tolist())
        #     for item in n_grams:
        #         if item.issubset(trace):
        #             training_data.loc[training_data[self.case_id] == case, str(set(item))] = 1
        #
        # y = training_data[self.outcome]
        # X = training_data.drop([self.outcome, self.case_id], axis=1)
        # clf = XGBClassifier()
        #
        # evolved_estimator = GAFeatureSelectionCV(
        #     estimator=clf,
        #     cv=5,
        #     scoring="accuracy",
        #     population_size=3,
        #     generations=5,
        #     n_jobs=1,
        #     verbose=True,
        #     keep_top_k=4,
        #     elitism=True,
        # )
        #
        # evolved_estimator.fit(X, y)
        # Selected_features = evolved_estimator.best_features_
        # plot = plot_fitness_evolution(evolved_estimator, metric="fitness")
        # plt.show()
        #
        # return X.columns[Selected_features].tolist()

    def frequent_activity_sets(self, min_support, top_k):
        df_2 = self.data[[self.case_id, self.activity]]
        grouped = df_2.groupby(self.case_id)[self.activity].apply(list)
        transactions = list(grouped)
        tr = TransactionEncoder()
        tr_arr = tr.fit(transactions).transform(transactions)
        df_2 = pd.DataFrame(tr_arr, columns=tr.columns_)
        frequent_itemsets = apriori(df_2, min_support=min_support, use_colnames=True)
        frequent_itemsets = frequent_itemsets.sort_values(['support'], ascending=False).head(
            top_k + len(self.data[self.activity].unique()))

        frequent_itemsets['item_size'] = frequent_itemsets.itemsets.apply(lambda x: len(list(x)))
        frequent_itemsets = frequent_itemsets[frequent_itemsets['item_size'] > 1]

        selected_itemsets = frequent_itemsets.sort_values(['support'], ascending=False).head(top_k)
        frequent_activities_list = selected_itemsets.itemsets.apply(lambda x: list(x)).to_list()

        return frequent_activities_list, selected_itemsets

    def generate_prefix_data(self, data):
        # generate prefix data (each possible prefix becomes a trace)
        data['case_length'] = data.groupby(self.case_id)[self.activity].transform(len)
        data['base_case_id'] = data[self.case_id]
        dt_prefixes = data[data['case_length'] >= self.min_length].groupby(self.case_id).head(self.min_length)
        for nr_events in range(self.min_length + 1, self.max_length + 1):
            tmp = data[data['case_length'] >= nr_events].groupby(self.case_id).head(nr_events)
            tmp[self.case_id] = tmp[self.case_id].apply(lambda x: "%s_%s" % (x, nr_events))
            dt_prefixes = pd.concat([dt_prefixes, tmp], axis=0)

        dt_prefixes['case_length'] = dt_prefixes.groupby(self.case_id)[self.activity].transform(len)
        dt_prefixes = dt_prefixes.reset_index(drop=True)

        return dt_prefixes

    def split_test_train(self, train_ratio):
        Cases = []
        Prefixes = []
        df = self.data
        case_id_col = self.case_id
        for case in df[case_id_col].unique():
            Cases.append(case)
            Prefixes.append(len(df[df[case_id_col] == case]))

        if self.max_length is None:
            self.max_length = max(Prefixes)

        Case_indexes = pd.DataFrame({case_id_col: Cases, "prefix length": Prefixes})
        train, test = train_test_split(Case_indexes, train_size=train_ratio, random_state=142)

        train_list = train[case_id_col]
        test_list = test[case_id_col]

        return train_list, test_list

    def cross_split_test_train(self, K_fold, regression=False):
        Cases = []
        Outcome = []
        df = self.data
        case_id_col = self.case_id
        for case in df[case_id_col].unique():
            Cases.append(case)
            Outcome.append(df[df[case_id_col] == case][self.outcome].unique().tolist()[0])

        if self.max_length is None:
            self.max_length = max(Outcome)

        Case_indexes = pd.DataFrame({case_id_col: Cases, "Outcome": Outcome})

        if regression:
            skf = KFold(n_splits=K_fold, shuffle=True)
        else:
            skf = StratifiedKFold(n_splits=K_fold, shuffle=True)

        train_dict = dict()
        test_dict = dict()
        for i, (train_index, test_index) in enumerate(skf.split(Case_indexes,
                                                                Case_indexes["Outcome"])):
            train_dict[i] = Case_indexes.iloc[train_index, 0]
            test_dict[i] = Case_indexes.iloc[test_index, 0]

        # train_list = train[case_id_col]
        # test_list = test[case_id_col]

        # return train_list, test_list
        return train_dict, test_dict

    def get_act_dict(self):
        dictionary_acts = dict()
        activities = self.data[self.activity].unique()
        self.num_acts = len(activities)
        for idx, act in enumerate(activities):
            dictionary_acts[act] = idx
        return dictionary_acts

    # def frequency_encoding(self):
    #     encoded_data = self.data.copy()
    #     encoded_data[[self.data[self.activity].unique().tolist()]] = 0
    #
    #     for case in encoded_data[self.case_id].unique():
    #         trace = encoded_data[encoded_data[self.case_id] == case][self.activity].tolist()
    #         for act in trace:
    #             encoded_data.loc[encoded_data[self.case_id] == case, act] = trace.count(act)
    #
    #     encoded_data.drop_duplicates(subset=[self.case_id], keep='first', inplace=True)
    #     encoded_data.replace(np.nan, 0, inplace=True)
    #
    #     return encoded_data

    def frequency_encoding(self, data):
        encoded_data = data[[self.case_id, self.outcome]]
        encoded_data.drop_duplicates(subset=[self.case_id], keep='first', inplace=True)
        activity_columns = data[self.activity].unique()

        # Create activity columns with zeros
        encoded_data[activity_columns] = 0

        # Count activity frequencies within each case
        for case in encoded_data[self.case_id]:
            trace = data[data[self.case_id] == case][self.activity].tolist()
            value_counts = pd.value_counts(trace)
            encoded_data.loc[encoded_data[self.case_id] == case, value_counts.index] = value_counts.values

        # encoded_data.replace(np.nan, 0, inplace=True)

        return encoded_data

    def binary_encoding(self, data):
        encoded_data = data[[self.case_id, self.outcome]]
        encoded_data.drop_duplicates(subset=[self.case_id], keep='first', inplace=True)
        activity_columns = data[self.activity].unique()

        # Create activity columns with zeros
        encoded_data[activity_columns] = 0

        # Count activity frequencies within each case
        for case in encoded_data[self.case_id]:
            trace = data[data[self.case_id] == case][self.activity].unique().tolist()
            # value_counts = pd.value_counts(trace)
            encoded_data.loc[encoded_data[self.case_id] == case, trace] = 1

        # encoded_data.replace(np.nan, 0, inplace=True)

        return encoded_data

    # def index_encoding(self, data):
    #     max_length = max(self.data.groupby(self.case_id)[self.activity].transform(len))
    #
    #     indexed_col = ['e%s' % i for i in range(1, max_length + 1)]
    #
    #     encoded_data = pd.concat([data, pd.DataFrame(columns=indexed_col)])
    #     encoded_data.drop_duplicates(subset=[self.case_id], keep='first', inplace=True)
    #
    #     for case in encoded_data[self.case_id]:
    #         trace_case = \
    #             data[data[self.case_id] == case].sort_values(by=[self.time_col])
    #         trace = trace_case[self.activity].tolist()
    #         trace_length = len(trace)
    #         for i in range(1, trace_length + 1):
    #             encoded_data.loc[encoded_data[self.case_id] == case, 'e%s' % i] = trace[i - 1]
    #
    #     encoded_data.replace(np.nan, 0, inplace=True)
    #     for col in indexed_col:
    #         one_hot = pd.get_dummies(encoded_data[col], prefix=col)
    #         encoded_data[one_hot.columns] = one_hot
    #
    #     # indexed_col.extend(['e%s_0' % i for i in range(min_length + 1, max_length + 1)])
    #     encoded_data.drop(indexed_col, axis=1, inplace=True)
    #
    #     AllActs = set(self.data[self.activity].unique().tolist())
    #     for e in indexed_col:
    #         Colsfor_e = [col for col in encoded_data.columns if col.split("_")[0] == e]
    #         Actsfor_e = {a.split("_")[1] for a in Colsfor_e}
    #         missedActs = AllActs.difference(Actsfor_e)
    #         new_cols = [e + "_" + a for a in missedActs]
    #         encoded_data[new_cols] = 0
    #
    #     return encoded_data

    def index_encoding(self, data):
        indexed_data = data.sort_values([self.case_id, 'event_nr'])
        max_length = int(indexed_data.groupby(self.case_id)[self.activity].transform('size').max())
        indexed_col = ['e%s' % i for i in range(1, max_length + 1)]

        # Pivot the DataFrame to create one-hot encoded columns
        ID_Labels = indexed_data[[self.case_id, self.outcome]].drop_duplicates(subset=[self.case_id], keep='first')
        encoded_data = pd.pivot_table(indexed_data, index=self.case_id, columns='event_nr',
                                      values=self.activity, aggfunc=lambda x: x, fill_value=0)
        encoded_data.columns = indexed_col
        ID_Labels.index = ID_Labels[self.case_id]
        encoded_data[self.case_id] = ID_Labels[self.case_id]
        encoded_data[self.outcome] = ID_Labels[self.outcome]

        for col in indexed_col:
            one_hot = pd.get_dummies(encoded_data[col], prefix=col)
            encoded_data[one_hot.columns] = one_hot

        encoded_data.drop(indexed_col, axis=1, inplace=True)

        # Add missing activity columns
        AllActs = set(self.data[self.activity].unique().tolist())
        missing_cols = [str(e) + "_" + str(act) for (e, act) in itertools.product(indexed_col, AllActs)
                        if str(e) + "_" + str(act) not in encoded_data.columns]
        encoded_data[missing_cols] = 0

        return encoded_data

    def trace_permutation_importance(self, model, X, y, case_list, constrain=False, n_repeats=10, random_state=2023):
        np.random.seed(random_state)
        # calculate the baseline accuracy
        predicted = model.predict(X)
        baseline = f1_score(y, predicted, average='weighted')

        # shuffle the location of each activity in the trace
        sub_data = self.data[self.data[self.case_id].isin(case_list)]
        # permutation_importance = pd.DataFrame(columns=['activity', 'iter', 'importance'])
        permutation_importance = []
        for act in self.data[self.activity].unique():
            for r in range(n_repeats):
                if len(self.Allowed_locations[act]) < 2:
                    permutation_importance.append({'activity': act,
                                                   'iter': r,
                                                   'importance': 0})

                    # print("No alternative location. skipped!")
                    continue

                shuffled_cases = []
                for case in case_list:
                    trace_case = sub_data[sub_data[self.case_id] == case].sort_values(by=['event_nr'])
                    trace = trace_case[self.activity].tolist()
                    if act in trace:
                        shuffled_cases.append(case)
                        # find the index of all occurrences of act
                        act_indexes = [i for i, x in enumerate(trace) if x == act]
                        for index in act_indexes:
                            if constrain:
                                # select a random index form self.AllowedLocations
                                adjusted_allowed = [i for i in self.Allowed_locations[act] if i < len(trace) + 1]
                                random_index = np.random.choice(adjusted_allowed, 1)[0] - 1
                            else:
                                # select a random index
                                random_index = np.random.choice(len(trace), 1)[0]

                            # move nth occurrence of act to random index and shifting other activities
                            trace.insert(random_index, trace.pop(index))
                            sub_data.loc[sub_data[self.case_id] == case, self.activity] = trace

                # encode shuffled cases only
                if len(shuffled_cases) < 3:
                    permutation_importance.append({'activity': act,
                                                   'iter': r,
                                                   'importance': 0})
                    print("No shuffled cases. skipped!")
                    continue
                else:
                    X_corrupt = self.index_encoding(sub_data[sub_data[self.case_id].isin(shuffled_cases)])
                    zero_features = [x for x in X.columns if x not in X_corrupt.columns]
                    X_corrupt[zero_features] = 0
                    # X = encoded_x[encoded_x[self.case_id].isin(case_list)]
                    X_corrupt.drop([self.case_id, self.outcome], axis=1, inplace=True)
                    X_corrupt = X_corrupt[X.columns]
                    # replace correpsonding rows in X with the shuffled cases in X_corrupt
                    Shuffled_X = X.copy()
                    Shuffled_X[Shuffled_X.index.isin(X_corrupt.index)] = X_corrupt
                    # calculate the importance
                    predicted = model.predict(Shuffled_X)
                    importance = baseline - f1_score(y, predicted, average='weighted')
                    permutation_importance.append({'activity': act,
                                                   'iter': r,
                                                   'importance': importance})

        # calculate mean importance for each activity
        permutation_importance = pd.DataFrame(permutation_importance)
        permutation_results = permutation_importance.groupby('activity').agg({'importance': 'mean'})
        permutation_results.columns = ['mean']
        permutation_results.reset_index(inplace=True)
        # sort the results
        permutation_results.sort_values(by=['mean'], ascending=True, inplace=True)
        permutation_results.drop(['mean'], axis=1, inplace=True)
        # transpose the results
        permutation_results = permutation_results.transpose()
        permutation_results.columns = permutation_results.iloc[0]
        permutation_results = permutation_results.iloc[1:]
        # add list of permutation importance for each activity as a column
        for act in self.data[self.activity].unique():
            permutation_results[act] = permutation_importance[
                permutation_importance['activity'] == act]['importance'].tolist()

        return permutation_results

    def shuffle_sequence(self, sequence, item_set):
        occurrences_indexes = self.find_itemset_indexes(sequence, item_set)
        shuffled_sequence = sequence.copy()
        locations = {index: index for index, value in enumerate(sequence)}
        for IDX in occurrences_indexes:
            subset = [sequence[i] for i in IDX]
            previous_index = -1
            max_possibles = dict()
            max_location = 10000
            for i, act in enumerate(subset[::-1]):
                # find the max possible location for each activity
                max_possibles[act] = max([a for a in self.Allowed_locations[act] if a < max_location])
                max_location = max_possibles[act]

            for i, act in enumerate(subset):
                adjusted_allowed = [index for index in self.Allowed_locations[act] if
                                    previous_index < index <= max_possibles[act]]
                random_index = np.random.choice(adjusted_allowed, 1)[0]
                previous_index = random_index
                current_act_index = locations[IDX[subset.index(act)]]
                shuffled_sequence.insert(random_index - 1, shuffled_sequence.pop(current_act_index))
                locations[current_act_index] = min(random_index - 1, len(sequence) - 1)
                for loc in locations:
                    if current_act_index < loc < random_index - 1:
                        locations[loc] -= 1

        return shuffled_sequence

    def find_itemset_indexes(self, sequence, itemset):
        itemset_length = len(itemset)
        indexes = []
        combinations_list = list(itertools.combinations(range(len(sequence)), itemset_length))
        for combination in combinations_list:
            subset = [sequence[i] for i in combination]
            if set(subset) == set(itemset):
                indexes.append(combination)

        remaining_indexes = []
        for t in indexes:
            non_overlapped = True
            for r_idx in remaining_indexes:
                if len(set(t).intersection(set(r_idx))) > 0:
                    non_overlapped = False
            if non_overlapped:
                remaining_indexes.append(t)

        return remaining_indexes

    def itemset_permutation_importance(self, model, X, y, case_list, frequent_itemsets,
                                       constrain=False, n_repeats=10, random_state=2023):

        np.random.seed(random_state)
        # calculate the baseline accuracy
        predicted = model.predict(X)
        baseline = f1_score(y, predicted, average='weighted')

        # shuffle the location of each itemset in the trace while keeping the order of activities same
        sub_data = self.data[self.data[self.case_id].isin(case_list)]
        permutation_importance = []
        for itemset in frequent_itemsets:
            for r in range(n_repeats):
                shuffled_cases = []
                for case in case_list:
                    trace_case = sub_data[sub_data[self.case_id] == case].sort_values(by=['event_nr'])
                    trace = trace_case[self.activity].tolist()
                    # check if the itemset is in the trace
                    set_of_items = frequent_itemsets[itemset]
                    if type(set_of_items) is str:
                        set_of_items = eval(frequent_itemsets[itemset].strip("'"))
                    elif type(set_of_items) is list:
                        set_of_items = set(frequent_itemsets[itemset])
                    if set_of_items.issubset(set(trace)):
                        if len(set_of_items) == len(trace):
                            continue
                        shuffled_cases.append(case)
                        shuffled_trace = self.shuffle_sequence(trace, set_of_items)
                        sub_data.loc[sub_data[self.case_id] == case, self.activity] = shuffled_trace

                # encode shuffled cases only
                X_corrupt = self.index_encoding(sub_data[sub_data[self.case_id].isin(shuffled_cases)])
                zero_features = [x for x in X.columns if x not in X_corrupt.columns]
                X_corrupt[zero_features] = 0
                # X = encoded_x[encoded_x[self.case_id].isin(case_list)]
                X_corrupt.drop([self.case_id, self.outcome], axis=1, inplace=True)
                X_corrupt = X_corrupt[X.columns]
                Shuffled_X = X.copy()
                Shuffled_X[Shuffled_X.index.isin(X_corrupt.index)] = X_corrupt

                predicted = model.predict(Shuffled_X)
                importance = baseline - f1_score(y, predicted, average='weighted')
                permutation_importance.append({'itemset': itemset,
                                               'iter': r,
                                               'importance': importance})

                print('Itemset: %s, Iteration: %s, Importance: %s' % (frequent_itemsets[itemset], r, importance))

        permutation_importance = pd.DataFrame(permutation_importance)
        permutation_results = pd.DataFrame(columns=permutation_importance["itemset"].unique())
        for itemset in frequent_itemsets:
            permutation_results[itemset] = permutation_importance[
                permutation_importance['itemset'] == itemset]['importance'].tolist()

        return permutation_results

    def PDP_process_location(self, model, activity, train_X, case_list, regressor=False):
        PD_activity = pd.DataFrame(columns=['location', 'Probability'])
        PD_activity['location'] = self.Allowed_locations[activity]
        # PD_activity['location'].sort_values(inplace=True)
        sub_data = self.data[self.data[self.case_id].isin(case_list)]

        for location in PD_activity['location']:
            shuffled_case = []
            for case in case_list:
                trace_case = sub_data[sub_data[self.case_id] == case].sort_values(by=['event_nr'])
                trace = trace_case[self.activity].tolist()
                if activity in trace:
                    shuffled_case.append(case)
                    # TODO: check how many times the activity is in the trace
                    trace.remove(activity)
                    trace.insert(location - 1, activity)
                    sub_data.loc[sub_data[self.case_id] == case, self.activity] = trace

            # encode shuffled cases only
            X_corrupt = self.index_encoding(sub_data[sub_data[self.case_id].isin(shuffled_case)])
            zero_features = [x for x in train_X.columns if x not in X_corrupt.columns]
            X_corrupt[zero_features] = 0
            X_corrupt.drop([self.case_id, self.activity, self.time_col, self.outcome], axis=1, inplace=True)
            X_corrupt = X_corrupt[train_X.columns]
            # replace correpsonding rows in X with the shuffled cases in X_corrupt
            Shuffled_X = train_X.copy()
            Shuffled_X[Shuffled_X.index.isin(X_corrupt.index)] = X_corrupt

            if regressor:
                predicted_proba = np.mean(model.predict(Shuffled_X))
            else:
                predicted_proba = np.mean(model.predict_proba(Shuffled_X)[:, 1])

            PD_activity.loc[PD_activity['location'] == location, 'Probability'] = predicted_proba

        return PD_activity
