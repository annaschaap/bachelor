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
import textwrap
import seaborn as sns
from optuna.visualization import plot_optimization_history, plot_param_importances
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import GroupKFold
from typing import List, Optional, Tuple
from tabulate import tabulate
from matplotlib import colors as mcolors

random.seed(1213870)

def parametertuner(train:pd.DataFrame,  
             feature_lists:pd.DataFrame,
             train_name:str,
             trials: int,
             jobs: int):
    
    """Tunes parameters of the xgboost on training data

    Args:
        train (pd.DataFrame): the training data
        feature_lists (pd.DataFrame): the df of which features are used in each fold
        train_name: string used to save parameters in correct folder
        trials: number of trials used to optimize
        jobs: number of trials running simultaneously 

    Returns:
        Optimal parameters for the model.
    """ 

    # specifying the type of model as well as which parameters should be tuned. The intervals defined for the parameters are chosen based on a litterature search.
    def objective(trial):
        f1_scores = []
        count = 0

        params = {
            'learning_rate' : trial.suggest_float('learning_rate', 0.01, 0.3),
            'gamma' : trial.suggest_float('gamma', 0, 5),
            'max_depth' : trial.suggest_int('max_depth', 3, 10),
            'min_child_weight' : trial.suggest_int('min_child_weight',1, 100),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 25, log=True),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 25, log=True)
        }
            
        model = xgb.XGBClassifier(
            learning_rate=params['learning_rate'], 
            gamma=params['gamma'], 
            max_depth=params['max_depth'], 
            min_child_weight=params['min_child_weight'], 
            reg_lambda=params['reg_lambda'], 
            reg_alpha=params['reg_alpha']
            #scale_pos_weight = 0.85 # to account for class imbalance
        )

        # Creating a list of numbers: from 1 to number of feature lists. The loop sort of corresponds to a manual way of using cross_val_scores()
        index_list = list(range(1,max(feature_lists['fold']) + 1))

        for n in index_list:
            count = count + 1
            print(count)
            # extracting features of current iteration
            feature_list = feature_lists[feature_lists['fold'] == n]['features'].tolist()

            # For first interation of cross validation, training data includes fold 2,3,4,5 and fold 1 is then used for validation. Etc.
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

            # Predict validation set
            predY = model.predict(validationX)

            # calculating F1 score (metric to optimize for tuning the parameters: weighted f1 to make it optimize f1 score for both class 0 and 1)
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

    # Defining save directory
    save_dir = f'/work/bachelor/Python/results/params/{train_name}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Save plots inside save_dir
    plot_optimization_history(study).write_html(f"{save_dir}/optimization_history_{train_name}.html")
    plot_param_importances(study).write_html(f"{save_dir}/param_importances_{train_name}.html")

    # Save best parameters
    with open(f'{save_dir}/best_params.json', 'w') as f:
        json.dump(best_params, f)

    return best_params


def extract_bestparams(train_name):
    """ Little function to load already tuned parameters into wd instead of running parametertuner function again

    Args:
        train_name: string that should match train_name in parametertuner

    Returns:
    Loads in tuned hyperparameters
    """
    
    bestparams = pd.read_json(f'/work/bachelor/Python/results/params/{train_name}/best_params.json', typ='series')
    bestparams = bestparams.to_dict()

    for key, value in bestparams.items():
        if key in ['max_depth', 'min_child_weight']: 
            bestparams[key] = int(value)

    return bestparams

def modeltraining(train:pd.DataFrame,
             feature_lists:pd.DataFrame,
             bestparams:dict,
             train_name:str,
             save: bool):

    """Takes a df containing crossvalidated feature selection and fits model with tuned hyperparameters. Aggregates per participant

    Args:
        train (pd.DataFrame): training data set
        feature_lists (pd.DataFrame): dataframe with features selected for each fold
        bestparams (dict): output of parametertuner (the tuned hyperparameters)
        train_name (str): string containing name of training set - just for saving the models under the right name
        save (bool): whether model results should be saved or not

    Returns:
        Tuple: containing training classification reports, training confusion matrices, training model predictions
    """ 

    # Empty lists for appending
    training_classification_reports = []
    training_confusion_matrices = []
    training_model_predictions = []

    # specifying the model
    model = xgb.XGBClassifier(
        learning_rate=bestparams['learning_rate'], 
        gamma=bestparams['gamma'], 
        max_depth=bestparams['max_depth'], 
        min_child_weight=bestparams['min_child_weight'], 
        reg_lambda=bestparams['reg_lambda'], 
        reg_alpha=bestparams['reg_alpha']
        #scale_pos_weight = 0.85
    )

    # Creating a list of numbers from 1 to number of feature lists. THe rest of the loop sort of corresponds to a manual way of cross_val_scores()
    index_list = list(range(1, max(feature_lists['fold']) + 1))

    # DataFrame to store all predictions (to allow aggregation afterwards)
    all_predictions = pd.DataFrame(columns=['ID', 'true_diagnosis', 'predicted_diagnosis'])

    for n in index_list:
        # extracting correct features:
        feature_list = feature_lists[feature_lists['fold'] == n]['features'].tolist()

        # For first iteration of cross validation, training data includes fold 2,3,4,5 and fold 1 is then used for validation. Etc.
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

        # Predict validation set
        predY = model.predict(validationX)
        
        # Save predictions and IDs for this fold
        fold_predictions = pd.DataFrame({
            'ID': validationY['ID'].values,
            'true_diagnosis': validationY['Diagnosis'].values,
            'predicted_diagnosis': predY
        })
        
        # saving the model
        if not os.path.exists(f'../data/models/{train_name}'):
            os.makedirs(f'../data/models/{train_name}')
        joblib.dump(model, f'../data/models/{train_name}/{n}.pkl')

        # Aggregate predictions at participant level
        participant_predictions = (
            fold_predictions.groupby('ID')
            .agg({
                'true_diagnosis': 'first', # just taking the true diagnosis of the first row, as diagnosis is the same within each participant
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

        # Append metrics
        training_classification_reports.append(pd.DataFrame(participant_classification_report))
        training_confusion_matrices.append(pd.DataFrame(participant_confusion_matrix))
        training_model_predictions.append(participant_predictions)

    # optional saving
    if save:
        base_dir = f"results/training/{train_name}"
        directories = [
            f"results/training",
            base_dir
        ]
        
        # creating the directory if it does not already exist
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        
        # Save the files in a loop
        for n in index_list:
            model_predictions_path = os.path.join(base_dir, f"{train_name}_model_predictions_{n}.csv")
            classification_report_path = os.path.join(base_dir, f"{train_name}_classification_report_{n}.csv")
            confusion_matrix_path = os.path.join(base_dir, f"{train_name}_confusion_matrix_{n}.csv")
            
            pd.DataFrame(training_model_predictions[n-1]).to_csv(model_predictions_path, sep=',', index=True)
            pd.DataFrame(training_classification_reports[n-1]).to_csv(classification_report_path, sep=',', index=True)
            pd.DataFrame(training_confusion_matrices[n-1]).to_csv(confusion_matrix_path, sep=',', index=True)

    # Return the lists of model performance
    return training_classification_reports, training_confusion_matrices, training_model_predictions

def feature_importance(train_name, feature_lists: pd.DataFrame, save: bool):
    """
    Calculates and plots feature importance based on pretrained models from cross-validation.

    Args:
        train_name (str): name of training set
        feature_lists (pd.DataFrame): DataFrame containing features used for each fold.
        save (bool): True or False (saves the plot)

    Returns:
        pd.DataFrame: DataFrame of average feature importance across folds.
        plot visualizing feature importance for each model
    """
    # Empty list to store importance data for each fold
    fold_importances = []

    # List of fold indices
    index_list = list(range(1, max(feature_lists['fold']) + 1))

    for fold in index_list:
        # Path to the saved model for the current fold
        model_path = f'../data/models/{train_name}/{fold}.pkl'

        # Loading model
        model = joblib.load(model_path)

        # Loading features for the current fold
        feature_list = feature_lists[feature_lists['fold'] == fold]['features'].tolist()

        # Extracting feature importance and store it
        importance = model.feature_importances_
        fold_importances.append(pd.DataFrame({
            'Feature': feature_list,
            'Importance': importance
        }))

    # Combine importance data from all folds
    combined_importance = pd.concat(fold_importances)
    avg_importance = (
        combined_importance.groupby('Feature')['Importance']
        .mean()
        .reset_index()
        .sort_values(by='Importance', ascending=False)
    )

    # Select only the top 20 features
    top_20_importance = avg_importance.head(20)

    # Plot average feature importance for top 20 features
    plt.figure(figsize=(10, 6))
    plt.barh(top_20_importance['Feature'], top_20_importance['Importance'], color='skyblue')
    plt.xlabel('Average Importance')
    plt.ylabel('Feature')
    plt.title(f'Top 20 Feature Importance Across Folds ({train_name})')
    plt.gca().invert_yaxis()  # Invert y-axis to display the most important feature at the top
    plt.tight_layout()

    # Optionally, save the results
    if save:
        save_path = f"results/feature_importance/{train_name}"
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, f"feature_importance_{train_name}.png"))

    plt.show()

    return top_20_importance

def modeltesting(
             feature_lists: pd.DataFrame,
             test: pd.DataFrame,
             save: bool,
             test_name: str,
             train_name: str):

    """This function tests a trained model on a hold-out set to get out-of-sample performance. Aggregates by ID.

    Args:
        train_name (str): string to use for folder name and file pre-name - should match existing folder and file
        test_name (str): string to use for test folder name and file pre-name when saving performance
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

    # Store probabilities of all training (sub)models for soft voting
    all_probs = {n: [] for n in index_list}

    for n in index_list:
        # Extracting correct features for each fold
        feature_list = feature_lists[feature_lists['fold'] == n]['features'].tolist()

        # Dividing test set into predictor and predicted variables
        testX = test.loc[:, feature_list]
        testY = test.loc[:, ['Diagnosis']]

        # Load model and get probability predictions (not class predictions)
        model = joblib.load(f'../data/models/{train_name}/{n}.pkl')
        pred_proba = model.predict_proba(testX)  # Predict probabilities

        # Append probabilities for each ID
        #test["model_probabilities"] = list(pred_proba)
        test["model_probabilities"] = pred_proba.tolist()
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
        save_path = f"results/test/{train_name}"
        os.makedirs(save_path, exist_ok=True)
        model_predictions.to_csv(os.path.join(save_path, f'{test_name}_participant_predictions.csv'), index=False)
        ensemble_classification_report.to_csv(os.path.join(save_path, f'{test_name}_participant_classification_report.csv'), index=False)
        ensemble_confusion_matrix.to_csv(os.path.join(save_path, f'{test_name}_participant_confusion_matrix.csv'), index=False)

    return classification_reports, confusion_matrices, model_predictions, ensemble_classification_report, ensemble_confusion_matrix


def class_preparation(train_name, order: dict, new_names: dict):
    """
    This function extracts all test results from a model, which are afterward used to visualize the results.

    Args:
        train_name (str): Name of the training set.
        order: order of the tests on the x axis
        new_names (str): names (test conditions) used on the x-axis

    Returns:
        pd.DataFrame: Dataframe containing F1 scores, error intervals, model type, and test name for each test set the model was tested on.
    """ 
    # Define folder path
    folder_path = f"./results/test/{train_name}"

    files = [
        os.path.join(folder_path, file)
        for file in os.listdir(folder_path)
        if '1_participant_classification_report' in file
    ]

    # Initialize lists to store results
    sensitivities = []
    specificities = []
    test_types = []
    eb_sensitivities = [] # error bars
    eb_specificities = [] # error bars

    # Process files from both models
    for file_path in files:
        # extracting test type based on filename
        filename = os.path.basename(file_path)
        name_suffix = filename.find('_participant_classification_report.csv')
        test_type = filename[:name_suffix]
        test_types.append(test_type)

        # Load classification report
        conf_matrix = pd.read_csv(file_path)

        # Extract F1 scores for each class
        sensitivity = conf_matrix.iloc[0, 0]/(conf_matrix.iloc[0, 0]+conf_matrix.iloc[0, 1])
        specificity = conf_matrix.iloc[1, 1]/(conf_matrix.iloc[1, 1]+conf_matrix.iloc[1, 0])
        sensitivities.append(sensitivity)
        specificities.append(specificity)

        # Load predictions to calculate sample size
        pred_file_path = os.path.join(
            folder_path,
            f'{test_type}_participant_predictions.csv'
        )

        predictions = pd.read_csv(pred_file_path)
        n = predictions.shape[0]  # Sample size

        # Calculate error bars
        eb_sensitivity = ((sensitivity * (1 - sensitivity)) / n) ** 0.5
        eb_specificity = ((specificity * (1 - specificity)) / n) ** 0.5
        eb_sensitivities.append(eb_sensitivity)
        eb_specificities.append(eb_specificity)

    # Combine results into a DataFrame
    results = pd.DataFrame({
        'test_type': test_types,
        'Sensitivity': sensitivities,
        'Specificity': specificities,
        'eb_sensitivity': eb_sensitivities,
        'eb_specificity': eb_specificities
    })

    # Define the desired order for test types:
    desired_order = order

    # Ensure the test_type column follows this order
    results['test_type'] = pd.Categorical(
        results['test_type'], 
        categories=desired_order, 
        ordered=True
    )

    # Sort the DataFrame by test_type
    results = results.sort_values('test_type')

    # Rename test sets using new_names dictionary
    results['test_type'] = results['test_type'].replace(new_names)

    return results

def class_plot(train_name: str, plot_values: pd.DataFrame):
    """
    Visualizes the model performance.

    Args:
        train_name (str): Name of the model
        plot_values (pd.DataFrame): extracted model performance values from different test sets

    Returns:
        A plot showing model performance across different test sets and different models
    """

    # Extract data for plotting
    test_types = plot_values['test_type']
    sensitivity = plot_values['Sensitivity']
    specificity = plot_values['Specificity']
    error_bars_sens = plot_values['eb_sensitivity']
    error_bars_spes = plot_values['eb_specificity']

    # Initialize plot
    fig, ax = plt.subplots(figsize=(10, 10))  # Adjustable figure size

    # Colors for the plot
    colours = ['#ef8a62', '#67a9cf']  # Sensitivity and Specificity colors

    # Add error bars (no connecting line)
    ax.errorbar(test_types, sensitivity, yerr=error_bars_sens, capsize=5, color=colours[0], fmt='none')
    ax.errorbar(test_types, specificity, yerr=error_bars_spes, capsize=5, color=colours[1], fmt='none')

    # Plot Sens/spes lines with markers
    ax.plot(test_types, sensitivity, 'o-', label='Sensitivity', color=colours[0], linewidth=2)
    ax.plot(test_types, specificity, 'o-', label='Specificity', color=colours[1], linewidth=2)

    # Labels, title, and legend
    #ax.set_ylabel('F1 Score', fontsize=14)
    ax.set_title(train_name, fontsize=16)
    ax.legend(fontsize=12)

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right', fontsize=12)

    # Adjust layout to prevent clipping
    plt.tight_layout()

    # Display the plot
    plt.show()

def dimension_plot(dimension, plot=True):
    """
    Processes classification report files, computes F1 scores and error bars, 
    and optionally plots the results.

    Args:
        dimension (str): Dimension to analyze ('familiarity' or other).
        plot (bool): Whether to generate plots. Default is True.

    Returns:
        str: Tabulated results as a string.
        (Optional) A plot of the results if `plot=True`.
    """
    folder_path = "./results/xgb/test"

    # File selection based on the dimension
    if dimension == 'familiarity':
        files = [
            os.path.join(root, file)
            for root, _, files in os.walk(folder_path)
            for file in files
            if '_participant_classification_report' in file and 'MG' not in file
        ]
    else:
        files = [
            os.path.join(root, file)
            for root, _, files in os.walk(folder_path)
            for file in files
            if '_participant_classification_report' in file and 'Qunfam' not in file
        ]

    # lists to store results
    f1_scores_td = []
    f1_scores_asd = []
    f1_averages = []
    f1_averages_ebs = []
    test_names = []
    test_types = []
    model_types = []

    # loop to process files
    for file_path in files:
        filename = os.path.basename(file_path)
        name_suffix = filename.find('_participant_classification_report')
        test_name = filename[:name_suffix]
        print(test_name)
        test_names.append(test_name)

        if dimension == 'familiarity':
            if "Qunfam" in test_name and "Qfam" in test_name and '80' in test_name:
                test_type = 'Different Familiarity, Same Participants'
            elif "Qunfam" in test_name and "Qfam" in test_name and '20' in test_name:
                test_type = 'Different Familiarity, Different Participants'
            else:
                test_type = 'Within'
            
            model_type = 'Convo With Experimenter' if test_name[:2] == 'Qu' else 'Convo With Caregiver'

        else:
            if "MG" in test_name and "Qfam" in test_name and '80' in test_name:
                test_type = 'Different Task, Same Participants'
            elif "MG" in test_name and "Qfam" in test_name and '20' in test_name:
                test_type = 'Different Task, Different Participants'
            else:
                test_type = 'Within'
            
            model_type = 'Matching Game' if test_name[0] == 'M' else 'Convo With Caregiver'

        test_types.append(test_type)
        model_types.append(model_type)

        # Load classification report
        classification_report = pd.read_csv(file_path)

        # F1 scores from classification reports
        f1_td = classification_report.iloc[2, 0]
        f1_asd = classification_report.iloc[2, 1]
        f1_average = classification_report.iloc[2, 4]
        f1_scores_td.append(f1_td)
        f1_scores_asd.append(f1_asd)
        f1_averages.append(f1_average)

        # Load predictions to calculate sample size
        pred_file_path = [
            os.path.join(root, file)
            for root, _, files in os.walk(folder_path)
            for file in files
            if file == f'{test_name}_participant_predictions.csv'
        ]
        predictions = pd.read_csv(pred_file_path[0])
        n = predictions.shape[0]  # Sample size

        # Calculate F1 error bars
        f1_average_eb = ((f1_average * (1 - f1_average)) / n) ** 0.5
        f1_averages_ebs.append(f1_average_eb)

    # Combine results into a DataFrame
    results = pd.DataFrame({
        'test_type': test_types,
        'model_type': model_types,
        'test_name': test_names,
        'F1_score_td': f1_scores_td,
        'F1_score_asd': f1_scores_asd,
        'f1_average': f1_averages,
        'f1_average_eb': f1_averages_ebs
    })

    # Sorting the DataFrame by test_type and splitting it
    if dimension == 'familiarity':
        test_type_order = ['Within', 'Different Familiarity, Same Participants', 'Different Familiarity, Different Participants']
        results['test_type'] = pd.Categorical(results['test_type'], categories=test_type_order, ordered=True)
        results = results.sort_values('test_type')
        data1 = results[results['model_type'] == 'Convo With Experimenter']  # Qunfam
        data2 = results[results['model_type'] == 'Convo With Caregiver']    # Qfam
    else:
        test_type_order = ['Within', 'Different Task, Same Participants', 'Different Task, Different Participants']
        results['test_type'] = pd.Categorical(results['test_type'], categories=test_type_order, ordered=True)
        results = results.sort_values('test_type')
        data1 = results[results['model_type'] == 'Matching Game']          # MG
        data2 = results[results['model_type'] == 'Convo With Caregiver']   # Qfam

    # Optionally plot results
    if plot:
        labels = test_type_order
        colors = ['#ef8a62', '#67a9cf']
        ylabel = 'F1 Score'
        ylim = (0.3, 0.95)
        legend_labels = [data1['model_type'].iloc[0], data2['model_type'].iloc[0]]

        # Plotting logic integrated here
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.errorbar(data1['test_type'], data1['f1_average'], yerr=data1['f1_average_eb'], capsize=5, color=colors[0], fmt='none')
        ax.errorbar(data2['test_type'], data2['f1_average'], yerr=data2['f1_average_eb'], capsize=5, color=colors[1], fmt='none')
        ax.plot(data1['test_type'], data1['f1_average'], 'o-', label=legend_labels[0], color=colors[0], linewidth=2)
        ax.plot(data2['test_type'], data2['f1_average'], 'o-', label=legend_labels[1], color=colors[1], linewidth=2)

        ax.set_ylabel(ylabel, fontsize=16)
        ax.set_ylim(*ylim)

        # Rotate x-axis labels
        wrapped_labels = [textwrap.fill(label, width=20) for label in labels]
        plt.xticks(ticks=range(len(labels)), labels=wrapped_labels, fontsize=14, rotation=45)  # Change 45 to any angle you want

        ax.legend(fontsize=14)
        # ax.set_title(title, fontsize=18)
        plt.tight_layout()
        plt.show()

def heatmaps_preparation(participant: str):
    """
    Loads test predictions and merges them with metadata for either 'same' or 'different' participants.

    Args:
        participant (str): 'same' to return merged DataFrame where training and testing participants are the same or 'different' to return merged DataFrame where training and testing participants are different

    Returns:
        pd.DataFrame: Merged DataFrame with predictions and metadata used to plot heatmaps.
    """

    # Define file path
    folder_path = "./results/test"

    # Collect files from the folder
    files = [
        os.path.join(root, file)
        for root, _, file_list in os.walk(folder_path)
        for file in file_list
        if '_1_participant_predictions' in file
    ]

    # Initialize lists to store results
    test_names, all_IDs, all_Diagnoses = [], [], []

    # Process files to extract predictions
    for file_path in files:
        filename = os.path.basename(file_path)
        test_name = filename.split('_participant_predictions')[0]  # Extract the test name
        
        preds = pd.read_csv(file_path)  # Load predictions
        test_names.extend([test_name] * len(preds))  # Repeat test_name for each row
        all_IDs.extend(preds['ID'].tolist())  # Extract IDs
        all_Diagnoses.extend(preds['ensemble_predictions'].tolist())  # Extract predictions

    # Combine results into a DataFrame
    icr = pd.DataFrame({'test_name': test_names, 'ID': all_IDs, 'Diagnosis': all_Diagnoses})
    icr_wide = icr.pivot(index="ID", columns="test_name", values="Diagnosis")  # Pivot to wide format

    # Select the appropriate test results based on 'participant' input
    if participant == 'same':
        test_filtered = icr_wide.filter(like='80', axis=1).dropna()  # 'same' -> filter for '80'
    else:  # 'different'
        test_filtered = icr_wide.filter(like='20', axis=1).dropna()  # 'different' -> filter for '20'

    # Load and clean metadata
    meta = pd.read_csv('../data/metadata.csv')
    meta_subset = meta[['intake_ldc_pin', 'intake_final_group', 'intake_sex', 'tsi_participantinfo_age']].copy()

    # Clean metadata columns
    meta_subset.rename(columns={
        'intake_ldc_pin': 'ID',
        'intake_final_group': 'diagnosis',
        'intake_sex': 'sex',
        'tsi_participantinfo_age': 'age'
    }, inplace=True)
    meta_subset['diagnosis'] = meta_subset['diagnosis'].replace({'TDC': 'NT', 'ASD': 'A'})

    # Merge predictions with metadata
    merged = pd.merge(test_filtered, meta_subset, on='ID', how='inner')

    # Add 'true_diagnosis' column
    merged['true_diagnosis'] = np.where(merged['diagnosis'] == 'NT', 0, 1)

    return merged

def heatmaps_plot(grouping: str, data: pd.DataFrame):
    """
    Function that creates agreement heatmaps between models grouped by a demographic.
    
    Args:
        grouping (str): 'sex' or 'age' and 'diagnosis'
        data (pd.DataFrame): The DataFrame for plotting, correctly formatted (output of heatmaps_preparation)

    Returns:
        None: Displays heatmaps with subplots showing model agreement.
    """
    # Columns to exclude from the matrix
    columns_to_exclude = ['sex', 'diagnosis', 'age', 'true_diagnosis', 'ID']
    relevant_columns = [col for col in data.columns if col not in columns_to_exclude]

    # Group by the chosen demographic (only done once)
    if grouping == 'age':
        data['age_group'] = data['age'].apply(lambda x: 'young' if x <= 10 else 'old')
        groups = data.groupby(['age_group', 'diagnosis'])
    else: 
        groups = data.groupby([grouping, 'diagnosis'])

    # Define the custom colormap and boundaries
    cmap = mcolors.ListedColormap(['beige', 'red', 'green'])  # Colors for 0, 50, 100
    bounds = [0, 25, 75, 100]  # Boundaries for beige, red, and green
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # Loop over the grouped data
    for group_value, group in groups:
        num_participants = group.shape[0]
        cols = 4
        rows = (num_participants + cols - 1) // cols  # Calculate rows for subplots

        # Create the subplots once per group
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
        axes = axes.flatten()  # Flatten the axes array for easier access

        for i, (_, participant_data) in enumerate(group.iterrows()):
            # Create participant matrix with relevant columns
            participant_matrix = participant_data[relevant_columns].to_frame().T
            true_diagnosis = participant_data['true_diagnosis']
            models = participant_matrix.columns

            agreement_matrix = pd.DataFrame(index=models, columns=models, dtype=float)

            # Populate the agreement matrix
            for model1 in models:
                for model2 in models:
                    if participant_matrix[model1].iloc[0] == participant_matrix[model2].iloc[0]:
                        agreement_matrix.loc[model1, model2] = (
                            100 if participant_matrix[model1].iloc[0] == true_diagnosis else 50
                        )
                    else:
                        agreement_matrix.loc[model1, model2] = 0

            # Plot the heatmap
            sns.heatmap(
                agreement_matrix, annot=False, cmap=cmap, norm=norm, 
                cbar=False, ax=axes[i], linewidths=0.5, fmt="", 
                xticklabels=models, yticklabels=models
            )
            axes[i].set_title(f"Participant {participant_data['ID']}")
            axes[i].set_xlabel("")
            axes[i].set_ylabel("")

        # Remove unused subplots (if any)
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        fig.suptitle(f"Heatmaps for Participants Grouped by {grouping}: {group_value}", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to avoid overlap
        plt.show()
