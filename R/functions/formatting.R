source("functions/load_files.R")

formatting <- function(meta, task){
  
  counter <- 0
  
  files <- load_files(preprocessed = TRUE, task)
  acoustics <- files$acoustics
  
  # making one big df with all the data
  all_acoustics <- data.frame()
  
  for (i in acoustics){
    counter <- counter + 1
    print(counter)
    
    # extracting the pin, since pin = ID in meta
    basename <- basename(i)
    pin <- str_extract(basename, "(?<=PIN)[0-9]+")
    
    # finding the matching row in meta and the diagnosis
    matching_row <- meta %>% filter(ID == pin)
    diagnosis <- matching_row$Diagnosis
    
    # troubleshooting
    if (nrow(matching_row) == 0){
      print(paste0("issue with file:", basename))
    } 
    
    else{
      
      file <- read_delim(i, delim = ";", show_col_types = FALSE)
      
      # adding Diagnosis and pin to the acoustic file and deleting name and frameTime
      file$Diagnosis <- diagnosis
      file$Diagnosis <- as.factor(file$Diagnosis)
      file$ID <- pin
      file$ID <- as.factor(file$ID)
      
      # adding to the big df
      if (nrow(all_acoustics) == 0) {
        all_acoustics <- file
      }
      else {
        all_acoustics <- rbind(all_acoustics, file)
      }
    }
  }
  
  # saving the formatted file
  folder_path <- "/work/bachelor/Anna/formatted"
  if (!dir.exists(folder_path)) {
    dir.create(folder_path, recursive = TRUE)
  }
  
  path_acoustic <- file.path(folder_path, paste0(task, "_acoustics.csv"))
  write_delim(all_acoustics, path_acoustic, delim = ";")
  
  return(all_acoustics)
}