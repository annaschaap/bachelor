train_ids <- function(meta){
  #Using meta to partition participants into train:
  #splitting into groups: gender, diagnosis, IQ-score
  # males
  male_asd <- meta %>% 
    filter(Gender == "Male") %>% filter(Diagnosis == 1)
  # splitting male_asd into two groups based on IQ score
  male_asd1 <- male_asd %>% 
    filter(IQ <= median(IQ, na.rm = TRUE)) %>% sample_frac(0.8) # rounds to nearest integer. 
  #na.rm since there is one instance of NA IQ. Because it is the only one, we figured, this was an ok method
  male_asd2 <- male_asd %>% 
    filter(IQ > median(IQ, na.rm = TRUE)) %>% sample_frac(0.8) 
  
  male_td <- meta %>% 
    filter(Gender == "Male") %>% filter(Diagnosis == 0) %>% sample_frac(0.8)
  # splitting male_td into two groups based on IQ score
  male_td1 <- male_td %>% 
    filter(IQ <= median(IQ, na.rm = TRUE)) %>% sample_frac(0.8)
  male_td2 <- male_td %>% 
    filter(IQ > median(IQ, na.rm = TRUE)) %>% sample_frac(0.8) 
  
  
  # females
  female_asd <- meta %>% 
    filter(Gender == "Female") %>% filter(Diagnosis == 1)
  # splitting female_asd into two groups based on autism severity
  female_asd1 <- female_asd %>% 
    filter(IQ <= median(IQ)) %>% sample_frac(0.8)
  female_asd2 <- female_asd %>% 
    filter(IQ > median(IQ)) %>% sample_frac(0.8)
  
  female_td <- meta %>% 
    filter(Gender == "Female") %>% filter(Diagnosis == 0) %>% sample_frac(0.8)
  female_td1 <- female_td %>% 
    filter(IQ <= median(IQ)) %>% sample_frac(0.8)
  female_td2 <- female_td %>% 
    filter(IQ > median(IQ)) %>% sample_frac(0.8)
  
  # this is the 80% of participants that are always used for training:
  ids <- rbind(male_asd1, male_asd2, male_td1, male_td2, female_asd1, female_asd2, female_td1, female_td2)
  
  return(ids)
}