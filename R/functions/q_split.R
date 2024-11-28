source("functions/load_files.R")

q_split <- function(task) {
  
  if (task == "MatchingGame"){
    stop()
  }
  
  else {
  # Loading in the acoustic preprocessed question files
  files <- load_files(preprocessed = TRUE, task)
  acoustics <- files$acoustics
  
  # Extracting the PIN and date
  name <- basename(acoustics)
  pins <- str_extract(name, "PIN[0-9]+")
  date <- str_extract(name, "^\\d{8}")
  acoustics_df <- data.frame(filename = name, PIN = pins, date = date)
  
  # Set up folders, which files are sorted in + creating them if they dont already exist
  familiar_folder <- "/work/bachelor/Anna/preprocessed_Questions_fam"
  unfamiliar_folder <- "/work/bachelor/Anna/preprocessed_Questions_unfam"
  source_folder <- "/work/bachelor/Anna/preprocessed_Questions"
  
  if (!dir.exists(unfamiliar_folder)) {
    dir.create(unfamiliar_folder, recursive = TRUE)
  }
  if (!dir.exists(familiar_folder)) {
    dir.create(familiar_folder, recursive = TRUE)
  }
  
  # Process non-problematic files
  split_acoustics <- acoustics_df %>%
    group_by(PIN) %>%
    summarize(
      even = list(filename[c(seq(2, n(), by = 2))]),  # Select even indices
      odd = list(filename[c(seq(1, n(), by = 2))]),    # Select odd indices
      .groups = 'drop'
    )
  
  # Move even files to the destination folder
  even_acoustics <- unlist(split_acoustics$even, use.names = FALSE)
  file.copy(file.path(source_folder, basename(even_acoustics)), 
              file.path(familiar_folder, basename(even_acoustics)))
  
  odd_acoustics <- unlist(split_acoustics$odd, use.names = FALSE)
  file.copy(file.path(source_folder, basename(odd_acoustics)), 
              file.path(unfamiliar_folder, basename(odd_acoustics)))
  
  # renaming old folder
  file.rename(source_folder, "/work/bachelor/Anna/Questions_transcripts")
  }
}

