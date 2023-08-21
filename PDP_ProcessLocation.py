import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, mean_squared_error
from tools import DataManager
import matplotlib.pyplot as plt


if __name__ == '__main__':
    K_fold = 5
    min_prefix_length = 2
    max_prefix_length = None
    Regressor = False
    data_manager = DataManager("./datasets/bpic2012_1_trunc40.csv", min_prefix_length, max_prefix_length)
    encoded_data = data_manager.index_encoding(data_manager.data)

    train_list, test_list = data_manager.cross_split_test_train(K_fold, Regressor)

    for ACT in data_manager.data[data_manager.activity].unique():
        PDP_ACT = dict()
        for i in range(K_fold):
            train_x = encoded_data[encoded_data[data_manager.case_id].isin(train_list[i])]
            train_y = train_x[data_manager.outcome].tolist()
            train_x.drop([data_manager.case_id, data_manager.activity, data_manager.time_col, data_manager.outcome],
                         axis=1, inplace=True)

            test_x = encoded_data[encoded_data[data_manager.case_id].isin(test_list[i])]
            test_y = test_x[data_manager.outcome].tolist()
            test_x.drop([data_manager.case_id, data_manager.activity, data_manager.time_col, data_manager.outcome],
                        axis=1, inplace=True)

            if Regressor:
                model = XGBRegressor()
            else:
                model = XGBClassifier()

            model.fit(train_x, train_y)

            if Regressor:
                predicted = model.predict(test_x)
                average_prediction = np.mean(predicted)
                print('regression mean squared error: %.3f' % mean_squared_error(test_y, predicted))

            else:
                predicted = model.predict(test_x)
                probabilities_positive = np.mean(model.predict_proba(test_x)[:, 1])

                print("f1-score test: %.3f" % f1_score(test_y, predicted, average='weighted'))

            PDP_ACT[i] = data_manager.PDP_process_location(model, ACT, train_x, train_list[i], regressor=Regressor)

        merged_df = pd.DataFrame()  # Initialize an empty dataframe to store the merged result
        dataframes = [PDP_ACT[p] for p in PDP_ACT.keys()]
        for i, df in enumerate(dataframes):
            df.rename(columns={'Probability': 'Probability_%s' % (i + 1)}, inplace=True)  # Rename 'Probability' column

            if i == 0:
                merged_df = df.copy()
            else:
                merged_df = pd.merge(merged_df, df[['location', 'Probability_%s' % (i + 1)]], on='location',
                                     how='outer')

        # Calculate average and standard deviation
        merged_df['Average'] = merged_df.iloc[:, 1:].mean(axis=1)
        merged_df['Std'] = merged_df.iloc[:, 1:].std(axis=1)

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(merged_df['location'], merged_df['Average'], color='blue', linewidth=2)
        plt.fill_between(merged_df['location'], merged_df['Average'] - merged_df['Std'],
                         merged_df['Average'] + merged_df['Std'], color='lightblue')

        plt.xlabel('Location')
        plt.xticks(data_manager.Allowed_locations[ACT])
        if Regressor:
            plt.ylabel('Prediction values')
            plt.title('Average Prediction with Variation for %s' % ACT)
        else:
            plt.ylabel('Probability values')
            plt.title('Average Probability for positive class with Variation for %s' % ACT)

        # save the figure
        plt.savefig('./datasets/bpic2012_1/PDP/PDP_%s.png' % ACT)
        # plt.show()

