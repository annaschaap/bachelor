combined_data <- function(data, demo){
  #combining data with relevant columns from demodata
  demo = demo %>% select(c(ID, Diagnosis, Gender))
  data = data %>% right_join(demo, by = "ID") %>%
    mutate(
      Diagnosis = as.factor(Diagnosis),
      Gender = as.factor(Gender)
    ) 
  return(data)
}

