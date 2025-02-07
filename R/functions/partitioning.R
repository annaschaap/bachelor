partitioning <- function(train_ids, data, task){
  
  # filtering the training ids of the input data. Same set is used as the 80 test set
  train_set <- data[data$ID %in% train_ids$ID,]

  # rest of the IDs are used for test set for different participants (20 test set)
  test_different <- data[!(data$ID %in% train_ids$ID),]
  
  # saving the sets
  folder_path <- paste0("/work/bachelor/preprocessing/partitioned")
  if (!dir.exists(folder_path)) {
    dir.create(folder_path, recursive = TRUE)
  }
  
  if (task == "MatchingGame"){
    # saving the test sets with different participants
    set1 <- file.path(folder_path, paste0("test_", task, "_20.csv"))
    write_delim(test_different, set1, delim = ";")
    
    # saving the training set and same participant test set
    train <- file.path(folder_path, paste0("train_", task, ".csv"))
    set2 <- file.path(folder_path, paste0("test_", task, "_80.csv"))
    write_delim(train_set, train, delim = ";")
    write_delim(train_set, set2, delim = ";") 
  }
  
  if (task == "Questions_fam"){
    # saving the test sets with different participants
    set1 <- file.path(folder_path, paste0("test_", task, "_20.csv"))
    write_delim(test_different, set1, delim = ";")
    
    # saving the training set and test set with same participants
    train <- file.path(folder_path, paste0("train_", task, ".csv"))
    set2 <- file.path(folder_path, paste0("test_", task, "_80.csv"))
    write_delim(train_set, train, delim = ";")
    write_delim(train_set, set2, delim = ";")
  }
  
  if (task == "Questions_unfam"){
    # saving the test sets with different participants
    set1 <- file.path(folder_path, paste0("test_", task, "_20.csv"))
    write_delim(test_different, set1, delim = ";")
    
    # saving the training set and same test set
    train <- file.path(folder_path, paste0("train_", task, ".csv"))
    set2 <- file.path(folder_path, paste0("test_", task, "_80.csv"))
    write_delim(train_set, train, delim = ";")
    write_delim(train_set, set2, delim = ";") 
  }
}








