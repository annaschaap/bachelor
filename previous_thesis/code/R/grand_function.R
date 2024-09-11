library(pacman)
pacman::p_load(tidyverse, readr, glmnet, data.table, broom, forcats, e1071, cvms)
source("functions/new_partitioning.R")
source("functions/Normalize_function.R")
source("functions/feature_selection.R")
source("functions/combine_dfs.R")
source("functions/id_wrangl.R")


grand_function <- function(features, 
                           other_dataframe,
                           demo, 
                           lang, 
                           task,
                           featureset, 
                           n_lasso_folds = 5){
  set.seed(1234)
  
  ###Cleaning the data###
  demo <- demo %>% filter(Gender == "Male")
  
  id_clean <- id_wrangling(features, demo, other_dataframe, language = lang)
  features <- id_clean[[1]]
  other_dataframe <- id_clean[[2]]
  demo <- id_clean[[3]]
  
  #Combining with demo 
  own_demo = filter(demo, language == lang)
  other_demo = filter(demo, language != lang)
  

  features <- combined_data(data = features, demo = own_demo)
  other_dataframe <- combined_data(data = other_dataframe, demo = other_demo)
  
  ###relevant task
  features_task <- features %>% filter(condition == task)

  features_other_task <- features %>% filter(condition != task)

  
  ###partitioning###
  if (lang == "dk"){
    partitions <- partition_dk(features = features_task, 
                               demo = own_demo)
  }
  if (lang == "us"){
    partitions <- partition_us(features = features_task,
                               demo = own_demo)
  }
  
  train <- partitions[[2]]
  hold_out <- partitions[[1]]
  
  #normalizing - all datasets are normalized according to the train set
  train_scaled <- as.data.frame(
    scale_function(train, datatype = "train"))
  
  hold_out_scaled <- as.data.frame(
    scale_function(train, 
                   hold_out, 
                   datatype = "test"))
  
  hold_out_other_scaled <- as.data.frame(
    scale_function(train, 
                   other_dataframe, 
                   datatype = "test"))
  
  features_other_task_scaled <- as.data.frame(
    scale_function(train, 
                   features_other_task, 
                   datatype = "test"))
  
  # ###Saving dataframe with other language after scaling 
   if (lang == "dk"){
     write.csv(hold_out_other_scaled, 
               paste(paste("../data/speech_data/",featureset, "model", lang, task, "test_on", "not", lang, sep = "_"), 
                     "csv", sep = "."))
   }
   if (lang == "us"){
     hold_out_other_scaled <- hold_out_other_scaled %>% filter(condition == task)
     write.csv(hold_out_other_scaled, 
               paste(paste("../data/speech_data/", featureset, "model", lang, task, "test_on", "not", lang, sep = "_"), 
                     "csv", sep = "."))
   }
   
  # #Saving dataframe with other task after scaling
   if (lang == "dk"){
    write.csv(features_other_task_scaled,
             paste(paste("../data/speech_data/",featureset, "model", lang, task, "test_on" , "not", task, sep = "_"),
                    "csv", sep = "."))
   }
   
  # #Saving hold out set from same task
   write.csv(hold_out_scaled,
            paste(paste("../data/speech_data/",featureset, "model", lang, task, "test_on", lang, task, sep = "_"),
                  "csv", sep = "."))

  train_scaled$ID <- as.factor(train_scaled$ID)
  # # ##Elastic net ###
   elastic(train_data = train_scaled,
           folds = n_lasso_folds,
           id_col = "ID",
          featureset = featureset,
           language = lang,
           hold_set = hold_out_scaled,
           task = task,
           demo_set = own_demo)
  #return(features)
}
