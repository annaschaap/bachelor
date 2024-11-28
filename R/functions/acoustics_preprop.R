
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
    
    # preprocessing the acoustic file:
    
    # only filtering child utterances
    acoustic <- acoustic %>%
      dplyr::filter(frameTime >= transcript$time_start & frameTime <= transcript$time_end)
    
    # Identiying pauses in speech:
    # Filter rows where F0final_sma is zero
    zeros <- acoustic %>% filter(F0final_sma == 0) %>% 
      select(frameTime, F0final_sma)
    
    if (nrow(zeros) > 0) {
      # Create a time difference column
      zeros <- zeros %>%
        mutate(diff = frameTime - lag(frameTime, default = first(frameTime)))
      zeros$diff <- round(zeros$diff, digits = 2)
      
      # creating sequences marked by cuts in frameTime
      zeros <- zeros %>%
        mutate(seq_group = cumsum(diff > 0.01))
      
      # Count the length of each sequence and filtering only sequences > 20 ( = 200 ms), as these equals to speech pauses
      sequence_counts <- zeros %>%
        group_by(seq_group) %>%
        summarise(count = n(), 
                  time_begin = first(frameTime), 
                  time_end = last(frameTime)) %>%
        ungroup() %>%
        filter(count > 20)  # Filter for sequences longer than 20
      
      # removing the intervals of sequence_counts in the acoustic file
      acoustic <- acoustic %>%
        dplyr::filter(!sapply(frameTime, function(ft) {
          any(ft >= sequence_counts$time_begin & ft <= sequence_counts$time_end)
        }))
    }

    # Save the preprocessed acoustic file
    folder_path <- paste0("/work/bachelor/Anna/preprocessed_", task)
    path_acoustic <- file.path(folder_path, paste0(identifier, "_preprop.csv"))
    write_delim(acoustic, path_acoustic, delim = ";")
  }
}

