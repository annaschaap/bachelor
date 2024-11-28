
acoustics_preprop <- function(transcripts, task, files) {
  
  # listing preprocessed transcript files:
  transcripts <- list.files(transcripts, full.names = TRUE)
  
  # listing the raw acoustics:
  acoustics <- files$acoustics
  # counter
  counter <- 0
  # listing through the preprocessed transcript files
  for (i in transcripts){
    counter <- counter + 1
    print(counter)
    
    basename <- basename(i)
    identifier <- sub("_preprop.*", "", basename)
    
    # reading in the transcript file
    transcript <- read_tsv(i, show_col_types = FALSE)
    
    # finding matching acoustic files
    matching_acoustic_files <- acoustics[grepl(identifier, acoustics)]
    
    if (length(matching_acoustic_files) == 0){
      next
    }
    
    if (length(matching_acoustic_files) == 1){
      acoustic <- read_delim(matching_acoustic_files[1], delim = ";", show_col_types = FALSE)
    }
    if (length(matching_acoustic_files) == 2){
      acoustic <- read_delim(matching_acoustic_files[2], delim = ";", show_col_types = FALSE)
    }
    
 
    # Initialize an empty dataframe to store median and IQR values for each turn
    summary_data <- data.frame()   
    
    # Loop through each row (conversational turn) in the transcript file
    for (j in 1:nrow(transcript)) {
      start_time <- transcript$time_start[j]
      end_time <- transcript$time_end[j]
      utterance_length <- end_time - start_time
      
      # Filter acoustic data within the current conversational turn timeframe
      turn_data <- acoustic %>%
        dplyr::filter(frameTime >= start_time & frameTime <= end_time) %>%
        filter(F0final_sma != 0)  # Exclude rows where F0final_sma is zero
      
      # Skip to next iteration if turn_data is empty
      if (nrow(turn_data) == 0) {
        next
      }
      
      turn_data <- turn_data %>% select(-c("name", "frameTime"))
      turn_data <- turn_data %>% drop_na()
      # Calculate median and IQR for each feature in turn_data
      
      turn_data_aggregated <- data.frame(matrix(NA, nrow = 1, ncol = 0))
      
      
      for (f in 1:ncol(turn_data)) {
        median <- median(turn_data[[f]], na.rm = TRUE)
        iqr <- IQR(turn_data[[f]], na.rm = TRUE)
        
        # Create new column names dynamically
        median_col_name <- paste("median_", colnames(turn_data)[f], sep = "")
        iqr_col_name <- paste("iqr_", colnames(turn_data)[f], sep = "")
        
        # Create a temporary dataframe with these values
        temp_df <- data.frame(median, iqr)
        colnames(temp_df) <- c(median_col_name, iqr_col_name)
        
        # Add the temporary dataframe to the aggregated dataframe
        turn_data_aggregated <- cbind(turn_data_aggregated, temp_df)
      }
      
      # Add utterance length to the aggregated dataframe
      turn_data_aggregated$utterance_length <- utterance_length
      
      summary_data <- rbind(summary_data, turn_data_aggregated)
      
    }
      

    # Save the preprocessed acoustic file
    folder_path <- paste0("/work/bachelor/Anna/preprocessed_", task)
    path_acoustic <- file.path(folder_path, paste0(identifier, "_preprop.csv"))
    write_delim(summary_data, path_acoustic, delim = ";")
  }
}

