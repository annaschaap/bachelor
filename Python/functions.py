import pandas as pd
import numpy as np
import os, random, joblib, statistics
import sklearn as sk
import optuna
import xgboost as xgb
import plotly
import json
import math
import matplotlib.pyplot as plt
from optuna.visualization import plot_optimization_history, plot_param_importances
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import GroupKFold
from typing import List, Optional, Tuple

random.seed(1213870)

# OBS: XGBoost does not allow these characters in column names, so this function is there to replace them with _
def clean_features(data, feature_list = True):
    """Replaces all invalid characters in the data

    Args:
        data (pd.DataFrame): either the training data (where column names are cleaned) or feature list (where feature columns are cleaned)
        feature_lists (bool): whether the data is the feature list or not

    Returns:
        Cleaned data
    """ 

    # Characters to replace with '_'
    invalid_chars = ['[', ']', '<', '>']
    
    if feature_list:
        # Replace invalid characters in the 'features' column
        data['features'] = data['features'].apply(lambda x: ''.join('_' if char in invalid_chars else char for char in x))

    else:
        data.columns = data.columns.str.replace(r'[\[\]<>]', '_', regex=True)
    
    return data


# parameter tuning definition for xg boost and svm (defined by model_type)  

def parametertuner(model_type: str,
            train:pd.DataFrame,  
             feature_lists:pd.DataFrame,
             train_name:str,
             trials: int,
             jobs: int):
    
    """Tunes parameters of the xgboost on training data

    Args:
        model_type: xgb or svm
        train (pd.DataFrame): the training data
        feature_lists (pd.DataFrame): the df of which features are used in which fold
        train_name: string used to save parameters in correct folder
        trials: number of trials used to optimize
        jobs: number of trials running simultaneously 

    Returns:
        Optimal parameters for the model.
    """ 

    # specifying which parameters should be tuned as well as the model
    def objective(trial):
        f1_scores = []

        count = 0
        if model_type == 'xgb':
            params = {
                'learning_rate' : trial.suggest_float('learning_rate', 0.01, 0.3),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 25, log=True),
                'max_depth' : trial.suggest_int('max_depth', 3, 10),
                'min_child_weight' : trial.suggest_int('min_child_weight',1, 100),
                'gamma' : trial.suggest_float('gamma', 0, 5),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 25, log=True)
            }
            
            model = xgb.XGBClassifier(
                learning_rate=params['learning_rate'], 
                gamma=params['gamma'], 
                max_depth=params['max_depth'], 
                min_child_weight=params['min_child_weight'], 
                reg_lambda=params['reg_lambda'], 
                reg_alpha=params['reg_alpha'],
                scale_pos_weight = 0.78
            )
        
        if model_type == 'svm':
            params = {
                "C" : trial.suggest_float("C", 0.001, 100),
                "gamma" : trial.suggest_float("gamma", 0.001, 100)
            }
            
            model = sk.svm.SVC(
                C = params['C'], 
                gamma = params['gamma'], 
                kernel = "rbf",
                class_weight = 'balanced' 
            )

        # Creating a list of numbers from 1 to number of feature lists. THe rest of the loop sort of corresponds to a manual way of cross_val_scores()
        index_list = list(range(1,max(feature_lists['fold']) + 1))

        for n in index_list:
            count = count + 1
            print(count)
            # extracting correct features:
            feature_list = feature_lists[feature_lists['fold'] == n]['features'].tolist()

            # For feature set 1 model, subset training data to only include fold 2,3,4,5. Etc.
            train_subset = train.loc[train['.folds'] != n]

            # Defining validation set
            validation = train.loc[train['.folds'] == n]

            # Dividing 'train' and 'validation' up into predictor variables (x) and what should be predicted (y)
            trainX = train_subset.loc[ : , feature_list]
            trainY = train_subset.loc[ : , ['Diagnosis']].values.ravel()
            validationX = validation.loc[ : , feature_list]
            validationY = validation.loc[ : , ['Diagnosis']].values.ravel()

            # use model
            model.fit(trainX, trainY)

            # Predict validation set with model
            predY = model.predict(validationX)

            # calculating F1 score (metric to optimize for tuning the parameters, weighted to make it optimize f1 for both 0 and 1)
            f1 = f1_score(validationY, predY, average='weighted')

            # appending the f1 score of current iteration
            f1_scores.append(f1)
        
        # calculating the mean (as hyperparameters should be optimized across all folds)
        mean = np.mean(f1_scores)

        return mean
        
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=trials, n_jobs=jobs)

    # extracting the tuned parameters
    best_params = study.best_params

    # saving optimization plot and most important parameters
    plot_optimization_history(study).write_html(f"optimization_history_{model_type}_{train_name}.html")
    plot_param_importances(study).write_html(f"param_importances_{model_type}_{train_name}.html")


    # saving bestparams
    save_dir = f'/work/bachelor/Python/results/{model_type}/params/{train_name}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(f'{save_dir}/best_params.json', 'w') as f:
        json.dump(best_params, f)
    
    return best_params

# little function to load already tuned parameters into wd instead of running function above again
def extract_bestparams(train_name, model_type):
    bestparams = pd.read_json(f'/work/bachelor/Python/results/{model_type}/params/{train_name}/best_params.json', typ='series')
    bestparams = bestparams.to_dict()
    if model_type == 'xgb':
        bestparams['max_depth'] = int(bestparams['max_depth'])
        bestparams['min_child_weight'] = int(bestparams['min_child_weight'])

    return bestparams

def modeltraining(train:pd.DataFrame,
             feature_lists:pd.DataFrame,
             bestparams:dict,
             model_type: str,
             train_name:str,
             save: bool):

    """Takes a df containing crossvalidated feature selection and fits model with tuned hyperparameters

    Args:
        train (pd.DataFrame): training data set
        feature_lists (pd.DataFrame): dataframe with features selected for each fold
        bestparams (dict): output of parametertuner (the tuned hyperparameters)
        model_type (str): xgb or svm
        train_name (str): string containing name of training set - just for saving the models under the right name
        save (bool): whether model results should be saved or not

    Returns:
        Tuple: containing validation classification reports, validation confusion matrices, validation model predictions
    """ 

    # Empty lists for appending
    validation_classification_reports = []
    validation_confusion_matrices = []
    validation_model_predictions = []

    # specifying the model
    if model_type == 'xgb':
        model = xgb.XGBClassifier(
            learning_rate=bestparams['learning_rate'], 
            gamma=bestparams['gamma'], 
            max_depth=bestparams['max_depth'], 
            min_child_weight=bestparams['min_child_weight'], 
            reg_lambda=bestparams['reg_lambda'], 
            reg_alpha=bestparams['reg_lambda'],
            scale_pos_weight = 0.78
        ) 
    if model_type == 'svm':
        model = sk.svm.SVC(
            C = bestparams['C'], 
            gamma = bestparams['gamma'],
            kernel = 'rbf', 
            probability=True, 
            class_weight = 'balanced') 

    index_list = list(range(1, max(feature_lists['fold']) + 1))

       # Creating a list of numbers from 1 to number of feature lists. THe rest of the loop sort of corresponds to a manual way of cross_val_scores()
    index_list = list(range(1,max(feature_lists['fold']) + 1))

    for n in index_list:
        # extracting correct features:
        feature_list = feature_lists[feature_lists['fold'] == n]['features'].tolist()

        # For feature set 1 model, subset training data to only include fold 2,3,4,5. Etc.
        train_subset = train.loc[train['.folds'] != n]

        # Defining validation set
        validation = train.loc[train['.folds'] == n]

        # Dividing 'train' and 'validation' up into predictor variables (x) and what should be predicted (y)
        trainX = train_subset.loc[ : , feature_list]
        trainY = train_subset.loc[ : , ['Diagnosis', 'ID']] 
        validationX = validation.loc[ : , feature_list]
        validationY = validation.loc[ : , ['Diagnosis', 'ID']]

        # use model
        model.fit(trainX, trainY['Diagnosis'])

        # Predict validation set with model
        predY = model.predict(validationX)
        
        # saving the model
        if not os.path.exists(f'../data/models/{train_name}_{model_type}'):
            os.makedirs(f'../data/models/{train_name}_{model_type}')
        joblib.dump(model, f'../data/models/{train_name}_{model_type}/{n}.pkl')

        # Retrieving performance measures
        validation_classification_report = pd.DataFrame(classification_report(validationY['Diagnosis'], predY, output_dict = True))
        validation_confusion_matrix = pd.DataFrame(confusion_matrix(validationY['Diagnosis'], predY))

        # Loading the performance into the empty lists
        validation_classification_reports.append(validation_classification_report)
        validation_confusion_matrices.append(validation_confusion_matrix)

        # Retrieving true diagnosis and model predictions and load it into dataframe    
        model_predictions = pd.DataFrame({f"fold_{str(n)}_true_diagnosis": validationY['Diagnosis'], 
                                        f"fold_{str(n)}_predicted_diagnosis": predY,
                                        f"ID_{str(n)}": validationY["ID"]})
        model_predictions["Correct"] = list(model_predictions[f"fold_{str(n)}_true_diagnosis"] == model_predictions[f"fold_{str(n)}_predicted_diagnosis"])
        validation_model_predictions.append(model_predictions)

    if save:
        # Define base directory path
        base_dir = f"results/{model_type}/validation/{train_name}"
        
        # List of directories to create
        directories = [
            f"results/{model_type}",
            f"results/{model_type}/validation",
            base_dir
        ]
        
        # Create directories if they do not exist
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        
        # Save the files
        for n in index_list:
            # Prepare file paths
            model_predictions_path = os.path.join(base_dir, f"{train_name}_model_predictions_{n}.csv")
            classification_report_path = os.path.join(base_dir, f"{train_name}_classification_report_{n}.csv")
            confusion_matrix_path = os.path.join(base_dir, f"{train_name}_confusion_matrix_{n}.csv")
            
            # Save model predictions, classification reports, and confusion matrices
            pd.DataFrame(validation_model_predictions[n-1]).to_csv(model_predictions_path, sep=',', index=True)
            pd.DataFrame(validation_classification_reports[n-1]).to_csv(classification_report_path, sep=',', index=True)
            pd.DataFrame(validation_confusion_matrices[n-1]).to_csv(confusion_matrix_path, sep=',', index=True)

    # Return the lists with model performance
    return validation_classification_reports, validation_confusion_matrices, validation_model_predictions

# diagnosis on participant level
def modeltraining2(train:pd.DataFrame,
             feature_lists:pd.DataFrame,
             bestparams:dict,
             model_type: str,
             train_name:str,
             save: bool):

    """Takes a df containing crossvalidated feature selection and fits model with tuned hyperparameters. Aggregation per participant version

    Args:
        train (pd.DataFrame): training data set
        feature_lists (pd.DataFrame): dataframe with features selected for each fold
        bestparams (dict): output of parametertuner (the tuned hyperparameters)
        model_type (str): xgb or svm
        train_name (str): string containing name of training set - just for saving the models under the right name
        save (bool): whether model results should be saved or not

    Returns:
        Tuple: containing validation classification reports, validation confusion matrices, validation model predictions
    """ 

    # Empty lists for appending
    validation_classification_reports = []
    validation_confusion_matrices = []
    validation_model_predictions = []

    # specifying the model
    if model_type == 'xgb':
        model = xgb.XGBClassifier(
            learning_rate=bestparams['learning_rate'], 
            gamma=bestparams['gamma'], 
            max_depth=bestparams['max_depth'], 
            min_child_weight=bestparams['min_child_weight'], 
            reg_lambda=bestparams['reg_lambda'], 
            reg_alpha=bestparams['reg_lambda'],
            scale_pos_weight = 0.78
        ) 
    if model_type == 'svm':
        model = sk.svm.SVC(
            C = bestparams['C'], 
            gamma = bestparams['gamma'],
            kernel = 'rbf', 
            probability=True, 
            class_weight = 'balanced') 

    index_list = list(range(1, max(feature_lists['fold']) + 1))

    # Creating a list of numbers from 1 to number of feature lists. THe rest of the loop sort of corresponds to a manual way of cross_val_scores()

    # DataFrame to accumulate predictions for aggregation
    all_predictions = pd.DataFrame(columns=['ID', 'true_diagnosis', 'predicted_diagnosis'])

    for n in index_list:
        # extracting correct features:
        feature_list = feature_lists[feature_lists['fold'] == n]['features'].tolist()

        # For feature set 1 model, subset training data to only include fold 2,3,4,5. Etc.
        train_subset = train.loc[train['.folds'] != n]

        # Defining validation set
        validation = train.loc[train['.folds'] == n]

        # Dividing 'train' and 'validation' up into predictor variables (x) and what should be predicted (y)
        trainX = train_subset.loc[ : , feature_list]
        trainY = train_subset.loc[ : , ['Diagnosis', 'ID']] 
        validationX = validation.loc[ : , feature_list]
        validationY = validation.loc[ : , ['Diagnosis', 'ID']]

        # use model
        model.fit(trainX, trainY['Diagnosis'])

        # Predict validation set with model
        predY = model.predict(validationX)
        
        # Save predictions and IDs for this fold
        fold_predictions = pd.DataFrame({
            'ID': validationY['ID'].values,
            'true_diagnosis': validationY['Diagnosis'].values,
            'predicted_diagnosis': predY
        })
        
        # saving the model
        if not os.path.exists(f'../data/models/{train_name}_{model_type}'):
            os.makedirs(f'../data/models/{train_name}_{model_type}')
        joblib.dump(model, f'../data/models/{train_name}_{model_type}/{n}.pkl')

        # Aggregate predictions at the participant level
        participant_predictions = (
            fold_predictions.groupby('ID')
            .agg({
                'true_diagnosis': 'first',  # Assuming all entries for a participant have the same diagnosis
                'predicted_diagnosis': lambda x: x.value_counts().idxmax()  # Majority vote
            })
            .reset_index()
            .rename(columns={'predicted_diagnosis': 'final_predicted_diagnosis'})
        )
        # ensuring they are actual numbers
        participant_predictions['true_diagnosis'] = participant_predictions['true_diagnosis'].astype(int)
        participant_predictions['final_predicted_diagnosis'] = participant_predictions['final_predicted_diagnosis'].astype(int)

        # Calculate participant-level metrics
        participant_classification_report = classification_report(
            participant_predictions['true_diagnosis'],
            participant_predictions['final_predicted_diagnosis'],
            output_dict=True
        )
        participant_confusion_matrix = confusion_matrix(
            participant_predictions['true_diagnosis'],
            participant_predictions['final_predicted_diagnosis']
        )

        # Append metrics for return and optional saving
        validation_classification_reports.append(pd.DataFrame(participant_classification_report))
        validation_confusion_matrices.append(pd.DataFrame(participant_confusion_matrix))
        validation_model_predictions.append(participant_predictions)

    if save:
        # Define base directory path
        base_dir = f"results/{model_type}/validation/{train_name}"
        
        # List of directories to create
        directories = [
            f"results/{model_type}",
            f"results/{model_type}/validation",
            base_dir
        ]
        
        # Create directories if they do not exist
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        
        # Save the files in a loop
        for n in index_list:
            # Prepare file paths
            model_predictions_path = os.path.join(base_dir, f"{train_name}_model_predictions_{n}.csv")
            classification_report_path = os.path.join(base_dir, f"{train_name}_classification_report_{n}.csv")
            confusion_matrix_path = os.path.join(base_dir, f"{train_name}_confusion_matrix_{n}.csv")
            
            # Save model predictions, classification reports, and confusion matrices
            pd.DataFrame(validation_model_predictions[n-1]).to_csv(model_predictions_path, sep=',', index=True)
            pd.DataFrame(validation_classification_reports[n-1]).to_csv(classification_report_path, sep=',', index=True)
            pd.DataFrame(validation_confusion_matrices[n-1]).to_csv(confusion_matrix_path, sep=',', index=True)

    # Return the lists of model performance
    return validation_classification_reports, validation_confusion_matrices, validation_model_predictions

# diagnosis per utterance
def modeltesting(
             feature_lists: pd.DataFrame,
             test: pd.DataFrame,
             model_type: str,
             save: bool,
             test_name: str,
             train_name: str):

    """This function tests a trained model on a hold-out set to get out-of-sample performance

    Args:
        train_name (str): string to use for folder name and file pre-name - should match existing folder and file
        test_name (str): string to use for test folder name and file pre-name when saving performance
        model_type (str): xgb or svm
        save (bool): whether or not to save performance metrics
        test (pd.DataFrame): dataframe with the hold out data
        feature_lists (pd.DataFrame): DataFrame containing features used in each fold

    Returns:
        tuple: tuple containing classification reports, confusion matrices, model predictions, ensemble classification_report, ensemble confusion_matrix
    """  
    # Empty lists for appending
    classification_reports = []
    confusion_matrices = []
    model_predictions = pd.DataFrame({"true_diagnosis": test["Diagnosis"]})

    # Creating a list of numbers from 1 to number of feature lists. 
    index_list = list(range(1, max(feature_lists['fold']) + 1))

    # Store probabilities of all models for soft voting
    all_probs = []

    for n in index_list:
        # Extracting correct features for each fold
        feature_list = feature_lists[feature_lists['fold'] == n]['features'].tolist()

        # Dividing test set into predictor and predicted variables
        testX = test.loc[:, feature_list]
        testY = test.loc[:, ['Diagnosis']]

        # Load model and get probability predictions (not class predictions)
        model = joblib.load(f'../data/models/{train_name}_{model_type}/{n}.pkl')
        pred_proba = model.predict_proba(testX)  # Predict probabilities (not the class label)

        # Collecting the probabilities for soft voting
        all_probs.append(pred_proba)

        # Retrieving performance measures for the current model
        classif_report = classification_report(testY, pred_proba.argmax(axis=1), output_dict=True)
        conf_matrix = pd.DataFrame(confusion_matrix(testY, pred_proba.argmax(axis=1)))

        # Loading the performance into the empty lists
        classification_reports.append(classif_report)
        confusion_matrices.append(conf_matrix)

        # Storing individual model's predicted probabilities for later
        model_predictions[f"model_{str(n)}_predicted_probabilities"] = list(pred_proba)

    # Soft voting: Stack the individual model probabilities
    all_probs = np.stack(all_probs, axis=0)  # Shape should be (num_models, num_samples, num_classes)

    # Compute mean probabilities across all models (axis=0)
    mean_probs = np.mean(all_probs, axis=0)  # Shape will be (num_samples, num_classes)

    # Get final predictions: class with the highest mean probability
    ensemble_predictions = mean_probs.argmax(axis=1)  # Final predictions

    # Adding the ensemble predictions to the dataframe
    model_predictions["ensemble_predictions"] = ensemble_predictions
    model_predictions["ID"] = test.loc[:, 'ID']
    model_predictions["Correct"] = (model_predictions["true_diagnosis"] == model_predictions["ensemble_predictions"])

    # Getting the classification report + confusion matrix for the ensemble model
    ensemble_classification_report = pd.DataFrame(classification_report(testY, ensemble_predictions, output_dict=True))
    ensemble_confusion_matrix = pd.DataFrame(confusion_matrix(testY, ensemble_predictions))

    # Optionally, save the results if needed
    if save:
        base_dir = f"results/{model_type}/test/{train_name}"
        os.makedirs(base_dir, exist_ok=True) 

        # Define file paths
        predictions_path = os.path.join(base_dir, f'{test_name}_predictions.csv')
        classification_report_path = os.path.join(base_dir, f'{test_name}_ensemble_classification_report.csv')
        confusion_matrix_path = os.path.join(base_dir, f'{test_name}_ensemble_confusion_matrix.csv')

        # Save the files
        pd.DataFrame(model_predictions).to_csv(predictions_path, index=False)
        ensemble_classification_report.to_csv(classification_report_path, index=False)
        ensemble_confusion_matrix.to_csv(confusion_matrix_path, index=False)

    return classification_reports, confusion_matrices, model_predictions, ensemble_classification_report, ensemble_confusion_matrix

# diagnosis per participant
def modeltesting2(
             feature_lists: pd.DataFrame,
             test: pd.DataFrame,
             model_type: str,
             save: bool,
             test_name: str,
             train_name: str):

    """This function tests a trained model on a hold-out set to get out-of-sample performance, aggregated by ID.

    Args:
        train_name (str): string to use for folder name and file pre-name - should match existing folder and file
        test_name (str): string to use for test folder name and file pre-name when saving performance
        model_type (str): xgb or svm
        save (bool): whether or not to save performance metrics
        test (pd.DataFrame): dataframe with the hold out data
        feature_lists (pd.DataFrame): DataFrame containing features used in each fold

    Returns:
        tuple: tuple containing classification reports, confusion matrices, model predictions, ensemble classification_report, ensemble confusion_matrix
    """  
    # Empty lists for appending
    classification_reports = []
    confusion_matrices = []
    model_predictions = pd.DataFrame({"ID": test["ID"].unique()})

    # Creating a list of numbers from 1 to the number of feature lists. 
    index_list = list(range(1, max(feature_lists['fold']) + 1))

    # Store probabilities of all models for soft voting
    all_probs = {n: [] for n in index_list}

    for n in index_list:
        # Extracting correct features for each fold
        feature_list = feature_lists[feature_lists['fold'] == n]['features'].tolist()

        # Dividing test set into predictor and predicted variables
        testX = test.loc[:, feature_list]
        testY = test.loc[:, ['Diagnosis']]

        # Load model and get probability predictions (not class predictions)
        model = joblib.load(f'../data/models/{train_name}_{model_type}/{n}.pkl')
        pred_proba = model.predict_proba(testX)  # Predict probabilities

        # Append probabilities for each ID
        test["model_probabilities"] = list(pred_proba)
        grouped = test.groupby("ID")["model_probabilities"].apply(
            lambda x: np.mean(np.stack(x), axis=0)
        )
        all_probs[n] = grouped

    # Stack all model probabilities for each ID and compute mean probabilities across models
    stacked_probs = np.stack([np.vstack(all_probs[n]) for n in index_list], axis=0)
    mean_probs = np.mean(stacked_probs, axis=0)  # Average across models

    # Get final predictions: class with the highest mean probability
    ensemble_predictions = mean_probs.argmax(axis=1)  # Final predictions
    model_predictions["ensemble_predictions"] = ensemble_predictions

    # True diagnosis per ID
    true_diagnosis = test.groupby("ID")["Diagnosis"].first()
    model_predictions["true_diagnosis"] = true_diagnosis.values

    # Check correctness of predictions
    model_predictions["Correct"] = model_predictions["true_diagnosis"] == model_predictions["ensemble_predictions"]

    # Getting classification report and confusion matrix for the ensemble model
    ensemble_classification_report = pd.DataFrame(
        classification_report(true_diagnosis, ensemble_predictions, output_dict=True)
    )
    ensemble_confusion_matrix = pd.DataFrame(
        confusion_matrix(true_diagnosis, ensemble_predictions)
    )

    # Optionally, save the results
    if save:
        save_path = f"results/{model_type}/test/{train_name}"
        os.makedirs(save_path, exist_ok=True)
        model_predictions.to_csv(os.path.join(save_path, f'{test_name}_participant_predictions.csv'), index=False)
        ensemble_classification_report.to_csv(os.path.join(save_path, f'{test_name}_participant_classification_report.csv'), index=False)
        ensemble_confusion_matrix.to_csv(os.path.join(save_path, f'{test_name}_participant_confusion_matrix.csv'), index=False)

    return classification_reports, confusion_matrices, model_predictions, ensemble_classification_report, ensemble_confusion_matrix


# extracting performance estimates for plots
def plot_estimates(train_name, order):
    """
    This function extracts all test results from a model, which are afterward used to visualize the results.

    Args:
        train_name (str): Name of the training set.
        order: order of the tests on the x axis

    Returns:
        pd.DataFrame: Dataframe containing F1 scores, error intervals, model type, and test name for each test set the model was tested on.
    """ 
    # Define folder paths for XGB and SVM results and collecting files in them
    folder_path_xgb = f"./results/xgb/test/{train_name}"
    folder_path_svm = f"./results/svm/test/{train_name}"

    files_xgb = [
        os.path.join(folder_path_xgb, file)
        for file in os.listdir(folder_path_xgb)
        if '_classification_report' in file
    ]
    files_svm = [
        os.path.join(folder_path_svm, file)
        for file in os.listdir(folder_path_svm)
        if '_classification_report' in file
    ]

    # Initialize lists to store results
    f1_scores_td = []
    f1_scores_asd = []
    test_types = []
    model_types = []
    eb_td_scores = []
    eb_asd_scores = []

    # Process files from both models
    for file_path in files_xgb + files_svm:
        # Determine model type and test type
        model_type = 'xgb' if folder_path_xgb in file_path else 'svm'
        model_types.append(model_type)

        filename = os.path.basename(file_path)
        name_suffix = filename.find('_participant_classification_report')
        test_type = filename[:name_suffix]
        test_types.append(test_type)

        # Load classification report
        classification_report = pd.read_csv(file_path)

        # Extract F1 scores
        f1_td = classification_report.iloc[2, 0]
        f1_asd = classification_report.iloc[2, 1]
        f1_scores_td.append(f1_td)
        f1_scores_asd.append(f1_asd)

        # Load predictions to calculate sample size
        pred_file_path = os.path.join(
            folder_path_xgb if model_type == 'xgb' else folder_path_svm,
            f'{test_type}_participant_predictions.csv'
        )
        predictions = pd.read_csv(pred_file_path)
        n = predictions.shape[0]  # Sample size

        # Calculate error bars
        eb_td = ((f1_td * (1 - f1_td)) / n) ** 0.5
        eb_asd = ((f1_asd * (1 - f1_asd)) / n) ** 0.5
        eb_td_scores.append(eb_td)
        eb_asd_scores.append(eb_asd)

    # Combine results into a DataFrame
    results = pd.DataFrame({
        'test_type': test_types,
        'model_type': model_types,
        'F1_score_td': f1_scores_td,
        'F1_score_asd': f1_scores_asd,
        'eb_td': eb_td_scores,
        'eb_asd': eb_asd_scores
    })

    # Define the desired order for test types: OBS RIGHT NOW A LITTLE TIDEOUS TO PUT IN
    desired_order = order

    # Ensure the test_type column follows this order
    results['test_type'] = pd.Categorical(
        results['test_type'], 
        categories=desired_order, 
        ordered=True
    )

    # Sort the DataFrame by test_type
    results = results.sort_values('test_type')

    return results

def plot_results(train_name: str, plot_values: pd.DataFrame):
    """This function visualizes the model performance - both svm and xgb is displayed in the same plot for comparison.
    Args:
        train_name(str): name of the model.
        plot_values (pd.DataFrame): extracted model performance values from different test sets

    Returns:
        A plot showing model performance across different test sets and different models
    """ 
    # Splitting results into model type
    xgb = plot_values[plot_values['model_type'] == 'xgb']
    svm = plot_values[plot_values['model_type'] == 'svm']

    # Plotting results
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])  # Axes are as wide and high as the figure

    # Cool colours
    colours1 = ['#8B0000', '#D97D3A']  # XGB
    colours2 = ['#228B22', '#8A9A5B']  # SVM

    # Add TD error bars (no connecting line)
    ax.errorbar( # XGB
        xgb['test_type'], 
        xgb['F1_score_td'],
        yerr=xgb['eb_td'], 
        capsize=5, 
        color=colours1[0], 
        fmt='none'
    )

    ax.errorbar( # SVM
        svm['test_type'], 
        svm['F1_score_td'], 
        yerr=svm['eb_td'], 
        capsize=5, 
        color=colours2[0], 
        fmt='none'
    )

    # Add ASD error bars
    ax.errorbar( # XGB
        xgb['test_type'], 
        xgb['F1_score_asd'], 
        yerr=xgb['eb_asd'], 
        capsize=5, 
        color=colours1[1], 
        fmt='none'
    )

    ax.errorbar( # SVM
        svm['test_type'], 
        svm['F1_score_asd'], 
        yerr=svm['eb_asd'], 
        capsize=5, 
        color=colours2[1], 
        fmt='none'
    )

    # Plot TD line - XGB
    ax.plot(
        xgb['test_type'], 
        xgb['F1_score_td'], 
        'o-', 
        label='XGB: TD', 
        color=colours1[0], 
        linewidth=2
    )

    # Plot ASD line - XGB
    ax.plot(
        xgb['test_type'], 
        xgb['F1_score_asd'], 
        'o:', 
        label='XGB: ASD', 
        color=colours1[1], 
        linewidth=2
    )
    # Plot TD line - SVM
    ax.plot(
        svm['test_type'], 
        svm['F1_score_td'], 
        'o-', 
        label='SVM: TD', 
        color=colours2[0], 
        linewidth=2
    )
    # Plot ASD line - SVM
    ax.plot(
        svm['test_type'], 
        svm['F1_score_asd'], 
        'o:', 
        label='SVM: ASD', 
        color=colours2[1], 
        linewidth=2
    )

    # Labels and title
    ax.set_xlabel('Test set')
    ax.set_ylabel('F1 Score')
    ax.set_title(f'{train_name} Model Performance')

    # Show legend
    ax.legend()

    # Display plot
    plt.show()
