#Lasso loop
elastic <- function(train_data, 
                    folds, 
                    task){
  
  #Divide train_data into 5 folds by adding new column ranging from [1:5]
  fold_train <- train_data %>% 
    groupdata2::fold(.,
                     k = folds,
                     cat_col = "Diagnosis",
                     id_col = "ID" )
  
  feature_list = NULL #Make empty list for storing selected features
  for(i in (1:length(unique(fold_train$.folds)))){
    print(paste("now looping through fold", i, sep = " "))
    lasso_train <- fold_train  %>% 
      filter(.folds != i) %>% 
      select(-c(ID))
    
    lasso_test <- fold_train  %>% 
      filter(.folds == i)
    
    ###DEFINING VARIABLES
    lasso_data <- lasso_train %>% select(-c( .folds))
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
  
  
  #writing the csv's
  path_fold_train <- file.path("/work/bachelor/data/Preprop_data_w_folds/", paste0(task, "_w_3folds.csv"))
  write_delim(fold_train, path_fold_train, delim = ";")
  
  path_feature_list <- file.path("/work/bachelor/data/feature_lists/", paste0(task, "_3fold_featurelist.csv"))
  write_delim(feature_list, path_feature_list, delim = ";")
}