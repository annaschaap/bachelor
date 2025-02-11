meta_preprop <- function(){
  # loading in the meta data
  meta <- read_csv("/work/bachelor/data/metadata.csv")
  
  # renaming columns into useful names
  meta <- meta %>% 
    rename(
      ID = intake_ldc_pin,
      Gender = intake_sex,
      Diagnosis = intake_final_group,
      IQ = iq
    )
  
  meta$Diagnosis <- ifelse(meta$Diagnosis == "ASD", 1, 0)
  
  return(meta)
}