###Only males partitioning
##Danish partitioning
partition_dk <- function(demo, features, hold_size = 0.2){
  set.seed(1234)
  #Using demoset to partition participants, then splitting data from the feature dataframe
  #How many participants should go into hold-out set from each subgroup (gender/diagnosis)
  n = round((nrow(demo) * hold_size)/2,0) 

  #Males
  male_asd <- demo %>% 
     filter(Diagnosis == "ASD") %>% sample_n(n)
  male_td <- demo %>% 
     filter(Diagnosis == "TD") %>% sample_n(n)
  
  hold_out_1 <- rbind(male_asd,male_td)
  hold_out_2 <- features[features$ID %in% hold_out_1$ID,]
  train <- features[!(features$ID %in% hold_out_1$ID),]
  return(list(hold_out_2,train)) #hold out first, train second
}

##Danish partitioning
partition_us <- function(demo, features, hold_size = 0.2){
  set.seed(1234)
  #Using demoset to partition participants, then splitting data from the feature dataframe
  #How many participants should go into hold-out set from each subgroup (gender/diagnosis)
  n = round((nrow(demo) * hold_size)/2,0) 
  
  #Males
  male_asd <- demo %>% 
    filter(Diagnosis == "ASD") %>% sample_n(n)
  male_td <- demo %>% 
    filter(Diagnosis == "TD") %>% sample_n(n)
  
  hold_out_1 <- rbind(male_asd,male_td)
  hold_out_2 <- features[features$ID %in% hold_out_1$ID,]
  train <- features[!(features$ID %in% hold_out_1$ID),]
  return(list(hold_out_2,train)) #hold out first, train second
}