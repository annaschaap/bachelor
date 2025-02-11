
remove_signs <- function(task){
  
  #Load file
  folder_path <- "/work/bachelor/preprocessing/formatted/"
  task_path <- file.path(folder_path, paste0(task, "_acoustics.csv"))
  
  file <- read_delim(task_path, delim = ";", show_col_types = FALSE)
  
  #Remove special characters
  colnames(file) <- str_replace_all(colnames(file), "[\\[\\]><]", "_")
  
  #Overwrite previous file
  write_delim(file, task_path, delim = ";")
  
  return(file)
}