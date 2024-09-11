import pandas as pd
import numpy as np
import os, random, joblib, statistics
import sklearn as sk
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import GroupKFold
import functions
random.seed(1213870)

path1 = os.path.join('data', 'speech_data')
# Load training + validation sets
data_dk_stories = pd.read_csv(os.path.join(path1,'data_egemaps_dk_stories.csv'), index_col = 0)
data_dk_triangles = pd.read_csv(os.path.join(path1, 'data_egemaps_dk_triangles.csv'), index_col = 0)
data_us_stories = pd.read_csv(os.path.join(path1, 'data_egemaps_us_stories.csv'), index_col = 0)

# Load test sets
model_dk_stories_test_on_dk_stories = pd.read_csv(os.path.join(path1, "egemaps_model_dk_stories_test_on_dk_stories.csv"), index_col = 0)
model_dk_stories_test_on_not_dk = pd.read_csv(os.path.join(path1, "egemaps_model_dk_stories_test_on_not_dk.csv"), index_col = 0)
model_dk_stories_test_on_not_stories = pd.read_csv(os.path.join(path1, "egemaps_model_dk_stories_test_on_not_stories.csv"), index_col = 0)
model_dk_triangles_test_on_dk_triangles = pd.read_csv(os.path.join(path1,"egemaps_model_dk_triangles_test_on_dk_triangles.csv"), index_col = 0)
model_dk_triangles_test_on_not_dk = pd.read_csv(os.path.join(path1,"egemaps_model_dk_triangles_test_on_not_dk.csv"), index_col = 0)
model_dk_triangles_test_on_not_triangles = pd.read_csv(os.path.join(path1, "egemaps_model_dk_triangles_test_on_not_triangles.csv"), index_col = 0)
model_us_stories_test_on_not_us = pd.read_csv(os.path.join(path1, "egemaps_model_us_stories_test_on_not_us.csv"), index_col = 0)
model_us_stories_test_on_us_stories = pd.read_csv(os.path.join(path1, "egemaps_model_us_stories_test_on_us_stories.csv"), index_col = 0)

#Path for getting feature lists
path2 = os.path.join('data', 'feature_lists')

features_dk_stories = functions.get_feature_list(path2, 'features_egemaps_dk_stories.csv', 5)
features_dk_triangles = functions.get_feature_list(path2, 'features_egemaps_dk_triangles.csv', 5)
features_us_stories = functions.get_feature_list(path2, 'features_egemaps_us_stories.csv', 5)

if __name__=='__main__':
    #Train and tweak hyperparameters
    save = True
    output = functions.train_tweak_hyper(train=data_dk_stories, train_name="model_dk_stories", feature_lists=features_dk_stories, kernel = "rbf", save=save)
    output = functions.train_tweak_hyper(train=data_dk_triangles, train_name="model_dk_triangles", feature_lists=features_dk_triangles, kernel = "rbf", save=save)
    output = functions.train_tweak_hyper(train=data_us_stories, train_name="model_us_stories", feature_lists=features_us_stories, kernel = "rbf", save=save)

    # Testing using dk_stories model
    output = functions.test_model(train_name="model_dk_stories", test_name="dk_stories", kernel="rbf", save=save, test=model_dk_stories_test_on_dk_stories, feature_lists=features_dk_stories)
    output = functions.test_model(train_name="model_dk_stories", test_name="us_stories", kernel="rbf", save=save, test=model_dk_stories_test_on_not_dk, feature_lists=features_dk_stories)
    output = functions.test_model(train_name="model_dk_stories", test_name="dk_triangles", kernel="rbf", save=save, test=model_dk_stories_test_on_not_stories, feature_lists=features_dk_stories)
    
    # Testing using dk_triangles model
    output = functions.test_model(train_name="model_dk_triangles", test_name="dk_stories", kernel="rbf", save=save, test=model_dk_triangles_test_on_not_triangles, feature_lists=features_dk_triangles)
    output = functions.test_model(train_name="model_dk_triangles", test_name="us_stories", kernel="rbf", save=save, test=model_dk_triangles_test_on_not_dk, feature_lists=features_dk_triangles)
    output = functions.test_model(train_name="model_dk_triangles", test_name="dk_triangles", kernel="rbf", save=save, test=model_dk_triangles_test_on_dk_triangles, feature_lists=features_dk_triangles)

    # Testing using us_stories model
    output = functions.test_model(train_name="model_us_stories", test_name="dk_stories", kernel="rbf", save=save, test=model_us_stories_test_on_not_us, feature_lists=features_us_stories)
    output = functions.test_model(train_name="model_us_stories", test_name="us_stories", kernel="rbf", save=save, test=model_us_stories_test_on_us_stories, feature_lists=features_us_stories)

    # Disentangle performance for same participants new task and new participants new task 
    stories_same, stories_new = functions.test_same_new_split(model_dk_stories_test_on_not_stories, data_dk_stories)
    triangles_same, triangles_new = functions.test_same_new_split(model_dk_triangles_test_on_not_triangles, data_dk_triangles)

    # Test on the new test sets
    output_tri_same = functions.test_model(train_name="model_dk_triangles", test_name="dk_tri_same", kernel="rbf", save=save, test=triangles_same, feature_lists=features_dk_triangles)
    output_tri_new = functions.test_model(train_name="model_dk_triangles", test_name="dk_tri_new", kernel="rbf", save=save, test=triangles_new, feature_lists=features_dk_triangles)
    output_sto_same = functions.test_model(train_name="model_dk_stories", test_name="dk_sto_same", kernel="rbf", save=save, test=stories_same, feature_lists=features_dk_stories)
    output_sto_new = functions.test_model(train_name="model_dk_stories", test_name="dk_sto_new", kernel="rbf", save=save, test=stories_new, feature_lists=features_dk_stories)
