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
    
    # DEFINING VARIABLES
    lasso_data <- lasso_train %>% select(-c( .folds))
    x <- model.matrix(Diagnosis ~ ., data = lasso_data) 
    y <- lasso_train$Diagnosis #choosing the dependent variable
    
    # FEATURE SELECTION
    set.seed(1234)
    cv_lasso <- cv.glmnet(x, 
                          y, 
                          alpha = 0.5, 
                          standardize = F,
                          family = "binomial",
                          type.measure = "auc")
    
    # EXTRACTING COEFFICIENTS
    lasso_coef <- tidy(cv_lasso$glmnet.fit) %>%  
      filter(lambda == cv_lasso$lambda.1se,
             term != "(Intercept)") %>% #we do not want the intercept
      select(term, estimate) %>% 
      mutate(abs = abs(estimate),
             term = str_remove_all(term, "`"), 
             lambda_1se = paste(cv_lasso$lambda.1se),
             test_fold = paste(i)) %>% 
      filter(abs > 0)
    
    # selecting columns to keep in csv file
    lists <- data.frame(features = ifelse(str_detect(lasso_coef$term, ".folds") == F,
                                          lasso_coef$term,
                                          NA))
    lists <- subset(lists, !is.na(lists$features))
    lists$fold <- paste(i)
    feature_list <- rbind(feature_list, lists)
  }
  
  
  #writing the csv's
  if (!dir.exists("/work/bachelor/data/Preprop_data_w_folds")) {
    dir.create("/work/bachelor/data/Preprop_data_w_folds", recursive = TRUE)
  }
  
  path_fold_train <- file.path("/work/bachelor/data/Preprop_data_w_folds", paste0(task, "_w_5folds.csv"))
  write_delim(fold_train, path_fold_train, delim = ";")
  
  
  if (!dir.exists("/work/bachelor/data/feature_lists")) {
    dir.create("/work/bachelor/data/feature_lists", recursive = TRUE)
  }
  
  path_feature_list <- file.path("/work/bachelor/data/feature_lists", paste0(task, "_5fold_featurelist.csv"))
  write_delim(feature_list, path_feature_list, delim = ";")
}