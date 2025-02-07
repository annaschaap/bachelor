normalizing <- function(train_set, test_set_list, datatype){
  
  normalize_column <- function(data, reference_data) {
    
    # making sure that ID and Diagnosis are factors
    data$ID <- as.factor(data$ID)
    data$Diagnosis <- as.factor(data$Diagnosis)
    
    for (i in seq_along(data)) { # Loop over each column index
      if (is.numeric(data[[i]])) { # only normalize numeric columns
        xmin <- min(reference_data[[i]], na.rm = TRUE)
        xmax <- max(reference_data[[i]], na.rm = TRUE)
        data[[i]] <- (data[[i]] - xmin) / (xmax - xmin)
      }
    }
    return(data)
  }
  
  # Normalize based on whether it is train or test: if train set is normalized, this normalized train set is returned as it is used for the elastic net regression. 
  #Whereas if test_set is normalized, it is just saved in a folder
  if (datatype == "train") {
    normalized <- normalize_column(train_set, train_set)
    
    return(normalized)
    
  } else if (datatype == "test") {
    for (i in test_set_list){
      test_set <- read_delim(i, delim = ";", show_col_types = FALSE)
      name <- basename(i)
      normalized <- normalize_column(test_set, train_set)
      
      # saving the test_sets
      folder_path <- paste0("/work/bachelor/data/test_sets/", deparse(substitute(train_set))) # this extracts the name of the train_set
      if (!dir.exists(folder_path)) {
        dir.create(folder_path, recursive = TRUE)
      }
      path_acoustic <- file.path(folder_path, name)
      write_delim(normalized, path_acoustic, delim = ";")
    }
  }
}