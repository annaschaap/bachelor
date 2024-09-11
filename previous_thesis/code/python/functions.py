from typing import List, Optional, Tuple
import pandas as pd
import os, joblib
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

def get_feature_list(path:str, file_name:str, n_folds: Optional[int]=5) -> List[List[str]] :
    """Takes a file containing cross-validated feature selection. 
    Extracts selected features and returns a list of lists with the features

    Args:
        path (str): path to folder with file
        file_name (str): file name (including .csv)
        n_folds (int, optional): number of cross-validation folds. Defaults to 5.

    Returns:
        List[List[str]]: List of n_folds feature lists
    """    
    feature_df = pd.read_csv(os.path.join(path, file_name), index_col = 0)
    feature_list = []
    for i in range(n_folds):
        single_set = feature_df.loc[feature_df['fold'] == i+1]["features"].tolist()
        feature_list.append(single_set)
    return feature_list

# Defining ml_train
def train_tweak_hyper(train:pd.DataFrame, 
             train_name:str, 
             feature_lists:List[List[str]], 
             kernel:str, 
             save:bool,
             gamma:str='scale'):
    """This function is for training SVM with specified kernel and tweaking hyperparameters on the validation set. 

    Args:
        train (pd.DataFrame): Full train data set with all features and a column indicating cross-validation folds
        train_name (str): String to use for folder name and file pre-name
        feature_lists (List[List[str]]): List of feature lists with strings indicating selected features
        kernel (str): Takes values 'linear', 'poly', 'rbf', 'sigmoid' or 'precomputed'
        save (bool): True or False - whether or not to save classification reports, model predictions and confusion matrices

    Returns:
        tuple: containing validation classification reports, validation confusion matrices, validation model predictions
    """    
    # Empty lists for appending
    validation_classification_reports = []
    validation_confusion_matrices = []
    validation_model_predictions = []

    # Model specifications
    model = SVC(kernel=kernel, class_weight = 'balanced', gamma=gamma) #default gamma is 'scaled', other possibility is auto

    # Creating a list of numbers from 1 to number of feature lists
    index_list = list(range(1,len(feature_lists)+1))

    # Loop that trains and validates
    for n in index_list:
        # For feature set 1 model, subset training data to only include fold 2,3,4,5. Etc.
        train_subset = train.loc[train['.folds'] != n]

        # Defining validation set
        validation = train.loc[train['.folds'] == n]

        # Dividing 'train' and 'validation' up into predictor variables (x) and what should be predicted (y)
        trainX = train_subset.loc[ : , feature_lists[n-1]] #n-1 because of 0-indexing
        trainY = train_subset.loc[ : , ['Diagnosis', 'ID']] #Include ID to be able to differentiate same participants new task and new participants new task
        validationX = validation.loc[ : , feature_lists[n-1]]
        validationY = validation.loc[ : , ['Diagnosis', 'ID']]
        
        # Fit model to training data and save the model
        model = model.fit(trainX, trainY["Diagnosis"])
        if not os.path.exists(f'../data/models/{train_name}'):
            os.makedirs(f'../data/models/{train_name}')
        joblib.dump(model, f'../data/models/{train_name}/{kernel}_{n}.pkl')

        # Predict validation set with model
        validation_predictions = model.predict(validationX)

        # Retrieving performance measures
        validation_classification_report = pd.DataFrame(classification_report(validationY["Diagnosis"], validation_predictions, output_dict = True))
        validation_confusion_matrix = pd.DataFrame(confusion_matrix(validationY["Diagnosis"], validation_predictions))

        # Loading the performance into the empty lists
        validation_classification_reports.append(validation_classification_report)
        validation_confusion_matrices.append(validation_confusion_matrix)

        # Retrieving true diagnosis and model predictions and load it into dataframe    
        model_predictions = pd.DataFrame({f"fold_{str(n)}_true_diagnosis": validationY["Diagnosis"], 
                                          f"fold_{str(n)}_predicted_diagnosis": validation_predictions,
                                          f"ID_{str(n)}": validationY["ID"]})
        model_predictions["Correct"] = list(model_predictions[f"fold_{str(n)}_true_diagnosis"] == model_predictions[f"fold_{str(n)}_predicted_diagnosis"])
        validation_model_predictions.append(model_predictions)

    if save:
        if not os.path.exists("results"):
            os.makedirs("results")
        if not os.path.exists("results/validation"):
            os.makedirs("results/validation")
        if not os.path.exists(f"results/validation/{train_name}"):
            os.makedirs(f"results/validation/{train_name}")
        for n in index_list:
            # Go through each index in the list of data frames with diagnosis predictions and true diagnosis - save them
            pd.DataFrame(validation_model_predictions[n-1]).to_csv(os.path.join("results", "validation", train_name, f"{train_name}_{kernel}_model_predictions_{n}.csv"), sep=',', index = True)
            
            # Go through each index in the list of classification reports  - save them
            pd.DataFrame(validation_classification_reports[n-1]).to_csv(os.path.join("results", "validation", train_name, f"{train_name}_{kernel}_classification_report_{n}.csv"), sep=',', index = True)

            # Go through each index in the list of confusion matrices  - save them
            pd.DataFrame(validation_confusion_matrices[n-1]).to_csv(os.path.join("results", "validation", train_name, f"{train_name}_{kernel}_confusion_matrix_{n}.csv"), sep=',', index = True)
    
    # Return the kernel + the 4 lists
    return validation_classification_reports, validation_confusion_matrices, validation_model_predictions

# Defining ml_test
def test_model(train_name:str, 
               test_name:str, 
               kernel:str, 
               save:bool, 
               test:pd.DataFrame, 
               feature_lists:List[List[str]]):
    """This function tests a trained model on a hold-out set to get out-of-sample performance

    Args:
        train_name (str): string to use for folder name and file pre-name - should match existing folder and file
        test_name (str): string to use for test folder name and file pre-name when saving performance
        kernel (str): string specifying kernel used for training model - used to extract correct model
        save (bool): whether or not to save performance metrics
        test (pd.DataFrame): dataframe with the hold out data
        feature_lists (List[List[str]]): list of lists of features selected

    Returns:
        tuple: tuple containing classification reports, confusion matrices, model predictions, ensemble classification_report, ensemble confusion_matrix
    """    

    # Empty lists for appending
    classification_reports = []
    confusion_matrices = []
    model_predictions = pd.DataFrame({"true_diagnosis": test["Diagnosis"]})

    # Creating a list of numbers from 1 to number of feature lists
    index_list = list(range(1,len(feature_lists)+1))

    # Loop that trains and validates
    for n in index_list:
        
        # Divide up the test set into predictor variables (testX) and what should be predicted (testY)
        testX = test.loc[ : , feature_lists[n-1]]
        testY = test.loc[ : , 'Diagnosis']

        # Predict test set with saved model
        predictions = joblib.load(f'../data/models/{train_name}/{kernel}_{n}.pkl').predict(testX)

        # Retrieving performance measures
        classif_report = pd.DataFrame(classification_report(testY, predictions, output_dict = True))
        conf_matrix = pd.DataFrame(confusion_matrix(testY, predictions))

        # Loading the performance into the empty lists
        classification_reports.append(classif_report)
        confusion_matrices.append(conf_matrix)

        # Retrieving true diagnosis and model predictions and load it into dataframe    
        model_predictions[f"model_{str(n)}_predicted_diagnosis"] = predictions

    # Getting majority decision of the 5 models and appending it to the df "model_predictions" 
    prediction_list = [f'model_{n}_predicted_diagnosis' for n in index_list] 
    
    ensemble_predictions = model_predictions[prediction_list].mode(axis = 1)

    # Appending new column with ensemble predictions
    model_predictions["ensemble_predictions"] = ensemble_predictions.iloc[:,0]
    model_predictions["ID"] = test.loc[ : , 'ID']
    model_predictions["Correct"] = list(model_predictions["true_diagnosis"] == model_predictions["ensemble_predictions"])
    
    # Getting the classification report + confusion matrix for the ensemble model. Both sexes.
    ensemble_classification_report = pd.DataFrame(classification_report(testY, ensemble_predictions.iloc[:,0], output_dict = True))
    ensemble_confusion_matrix = pd.DataFrame(confusion_matrix(testY, ensemble_predictions.iloc[:,0]))

    # Saving output
    if save == True:
        if not os.path.exists("results"):
            os.makedirs("results")
        if not os.path.exists("results/test"):
            os.makedirs("results/test")
        if not os.path.exists(f"results/test/{train_name}"):
            os.makedirs(f"results/test/{train_name}")
        # Save the predictions
    
        model_predictions.to_csv(os.path.join("results", "test", train_name, f"{train_name}_tested_on_{test_name}_{kernel}_model_predictions.csv"), sep=',', index = True)

        # Save the ensemble classification report + confusion matrix - both sexes
        pd.DataFrame(ensemble_classification_report).to_csv(os.path.join("results", "test", train_name, f"{train_name}_tested_on_{test_name}_{kernel}_classification_report_ensemble.csv"), sep=',', index = True)
        pd.DataFrame(ensemble_confusion_matrix).to_csv(os.path.join("results", "test", train_name, f"{train_name}_tested_on_{test_name}_{kernel}_confusion_matrix_ensemble.csv"), sep=',', index = True)

        # Save the individual model classification reports + confusion matrices
        for n in index_list:
            # Go through each index in the list of classification reports  - save them
            pd.DataFrame(classification_reports[n-1]).to_csv(os.path.join("results", "test", train_name, f"{train_name}_tested_on_{test_name}_{kernel}_classification_report_{n}.csv"), sep=',', index = True)

            # Go through each index in the list of confusion matrices  - save them
            pd.DataFrame(confusion_matrices[n-1]).to_csv(os.path.join("results", "test", train_name, f"{train_name}_tested_on_{test_name}_{kernel}_confusion_matrix_{n}.csv"), sep=',', index = True)

    return classification_reports, confusion_matrices, model_predictions, ensemble_classification_report, ensemble_confusion_matrix


def test_same_new_split(test_to_split: pd.DataFrame, df_train:pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Takes a non-holdout test set (same language, new task) and splits participants based on whether they were also included in train data.

    Args:
        test_to_split (pd.DataFrame): the test dataframe that should be split
        df_train (pd.DataFrame): the train data where hold-out IDs are not included

    Returns:
        tuple: test set with same participants as in train data, test set with new participants
    """    
    same = test_to_split[test_to_split["ID"].isin(list(df_train["ID"].unique()))]
    new = test_to_split[-test_to_split["ID"].isin(list(df_train["ID"].unique()))]
    return same, new