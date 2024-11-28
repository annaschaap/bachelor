source("functions/load_files.R")


q_duplicate_dates <- function(task) {
  
  if (task == "MatchingGame"){
    stop()
  }
  
  else {
    # Loading in the acoustic preprocessed question files
    files <- load_files(preprocessed = TRUE, task)
    acoustics <- files$acoustics
    
    # Extracting the PIN and date
    name <- acoustics
    basename <- basename(acoustics)
    pins <- str_extract(basename, "PIN[0-9]+")
    date <- str_extract(basename, "^\\d{8}")
    acoustics_df <- data.frame(filename = name, basename = basename, PIN = pins, date = date)
    
    # Group by PIN and check if n(unique_dates) = n(files) as all files per participant needs to have
    # a unique date in order to sort them correctly
    date_check <- acoustics_df %>%
      group_by(PIN) %>%
      summarize(
        unique_dates = n_distinct(date),  # Count unique dates for each PIN
        total_files = n()                  # Count total files for each PIN
      ) %>%
      mutate(
        date_check = ifelse(unique_dates == total_files, 1, 0)  # Check for date uniqueness
      )
    
    # Check for any PIN with a date check of 0 and fix those
    problematic_pins <- date_check %>% filter(date_check == 0)
  
    # fixing the problematic PINS by rbinding those acoustic files
    if (nrow(problematic_pins) > 0){
      problematic_files <- acoustics_df %>% filter(PIN %in% problematic_pins$PIN)
      
      for (pin in unique(problematic_files$PIN)) {
        pin_data <- problematic_files[problematic_files$PIN == pin, ]
        
        for (dates in pin_data){
          
          # Identify the duplicate dates
          duplicate_date <- pin_data$date[duplicated(pin_data$date)]
          duplicate_file <- pin_data %>% 
            filter(PIN == pin & date %in% duplicate_date) %>% 
            pull(filename)
          
          if (all(file.exists(duplicate_file))) { 
            # Loading in the files
            file1 <- read_delim(duplicate_file[1], delim = ";", show_col_types = FALSE)
            file2 <- read_delim(duplicate_file[2], delim = ";", show_col_types = FALSE)
            file <- rbind(file1, file2)
            
            # saving the new combined file
            name <- paste0(duplicate_date, "_", pin, "preprop.csv")
            path <- paste0("/work/bachelor/Anna/preprocessed_Questions/", name)
            write_delim(file, path, delim = ";")
            
            # deleting old files
            file.remove(duplicate_file[1])
            file.remove(duplicate_file[2])
          }
        }
      }
    }
  }
}