clean_cols <- function(df){
  badcolumns <- NULL #making an empty list
  for (columns in 1:length(df)){ #every column 
    if (is.factor(df[,columns])){ #is the column a factor?
      print(columns)
      if(uniqueN(df[,columns])<2){ #does the column have below 2 levels?
        bad_column_name <- colnames(df)[columns] #add the column name to a list of bad columns
        badcolumns <- c(badcolumns, bad_column_name) #combine it with the existing list
      }
    }
    if (is.numeric(df[,columns])){ #is the column numeric?
      print(columns)
      if(var(df[,columns], na.rm = T)==0|is.na(var(df[,columns]))){ #is variance 0?
        #maybe try to remove the is.na condition
        bad_column_name <- colnames(df)[columns]  #add the column name to a list of bad columns
        badcolumns <- c(badcolumns, bad_column_name)#combine it with the existing list
      }  
    }
  }
  return(badcolumns)
}