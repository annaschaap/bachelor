transcript_preprop <- function(task, files) {
  transcripts <- files$transcripts
  
  for (i in transcripts){
    transcript <- readr::read_tsv(i, col_names = FALSE, show_col_types = FALSE)
    colnames(transcript) <- c("name", "time_start", "time_end", "transcript", "speaker", "task_type")
    
    basename <- basename(i)
    identifier <- sub("_VBM.*", "", basename)
    
    # filtering out NA speakers
    transcript <- transcript %>%
      filter(!is.na(speaker))
    
    # filtering the task put in the argument:
    transcript <- transcript %>%
      filter(task_type == task)
    
    # filtering only child speech:
    transcript <- transcript %>% 
      filter(speaker == "Child")
  
    # saving the files: saving it in correct folder, here decided by task
    folder_path <- paste0("/work/bachelor/preprocessing/preprocessed_", task)
    if (!dir.exists(folder_path)) {
      dir.create(folder_path, recursive = TRUE)
    }
    
    path_transcript <- file.path(folder_path, paste0(identifier, "_preprop.txt"))
    write_tsv(transcript, path_transcript)  
  }
  return(folder_path)
}