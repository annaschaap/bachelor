load_files <- function(task, preprocessed = FALSE){
  folder_path <- if (preprocessed) {
    paste0("/work/bachelor/preprocessing/preprocessed_", task)
  } 
  else {
    paste0("/work/bachelor/data/", task)
  }
  
  files <- list.files(folder_path, full.names = TRUE)
  
  transcripts <- files[grepl("\\.txt$", files)]
  acoustics <- files[grepl("\\.csv$", files)]
  
  return(list(transcripts = transcripts, acoustics = acoustics))
}