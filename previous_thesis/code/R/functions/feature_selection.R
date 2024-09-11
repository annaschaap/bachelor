#Lasso loop
elastic <- function(train_data, 
                           folds, 
                           id_col, 
                           featureset = featureset, 
                           language,
                           hold_set,
                           task,
                           demo_set){
  #parition using groupdata 2, keeping IDs in same fold
  fold_train <- train_data %>% 
    groupdata2::fold(.,
                     k = folds,
                     id_col = id_col)  #new col called .fold

  feature_list = NULL #Make empty list for storing selected features
  for(i in (1:length(unique(fold_train$.folds)))){
    print(paste("now looping through fold", i, sep = " "))
    lasso_train <- fold_train  %>% 
      filter(.folds != i) %>% 
      select(-c(id_col))
    
    lasso_test <- fold_train  %>% 
      filter(.folds == i)
    
    ###DEFINING VARIABLES
    lasso_data <- lasso_train %>% select(-c(story_type, condition, Gender, 
                                            .folds, country, feature_set))
    x <- model.matrix(Diagnosis ~ ., data = lasso_data) #making a matrix from formula
    y <- lasso_train$Diagnosis #choosing the dependent variable
    
    ###FEATURE SELECTION
    set.seed(1234) #set seed
    cv_lasso <- cv.glmnet(x, 
                          y, 
                          alpha = 0.5, # Setting alpha between 0 and 1 implements elastic
                          standardize = F,
                          family = "binomial",
                          type.measure = "auc")
    
    ###EXTRACTING COEFFICIENTS
    lasso_coef <- tidy(cv_lasso$glmnet.fit) %>%  
      filter(lambda == cv_lasso$lambda.1se,
             term != "(Intercept)") %>% #we do not want the intercept
      select(term, estimate) %>% 
      mutate(abs = abs(estimate),
             term = str_remove_all(term, "`"), #clean the term string
             lambda_1se = paste(cv_lasso$lambda.1se),
             test_fold = paste(i)) %>% 
      filter(abs > 0)
    
    # return(cv_lasso) # Do this if you want to get the lambda plot
    
    # #selecting columns to keep in csv file
    lists <- data.frame(features = ifelse(str_detect(lasso_coef$term, ".folds") == F,
                                          lasso_coef$term,
                                          NA))
    lists <- subset(lists, !is.na(lists$features))
    lists$fold <- paste(i)
    feature_list <- rbind(feature_list, lists)
  }
  #writing the csvs
  write.csv(fold_train, paste(paste("../data/speech_data/data", featureset, language, task, sep = "_"), "csv", sep = "."))
  write.csv(feature_list, paste(paste("../data/feature_lists/features", featureset, language, task, sep = "_"),"csv", sep = "."))
}