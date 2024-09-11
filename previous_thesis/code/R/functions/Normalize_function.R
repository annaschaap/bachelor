scale_function <- function(df1, df2, datatype){ #df1 is the train set and df2 is the test set to be normalized with same values
  #If train test, just normalize with own values
  if (datatype=="train") {
    for (i in 1:length((df1))){
      if (sapply(df1[,i], is.numeric) == T){ #only normalizing numeric columns
        xmin = min(df1[i])
        xmax = max(df1[i])
        df1[i] <- (df1[i]-xmin)/(xmax - xmin)
      }
    }
  }
  #If test set, normalize with values from test set
  if (datatype == "test"){
    for (i in names(df2)){
      if(sapply(df1[,i], is.numeric) == T){ #only normalizing numeric columns
        xmin = min(df1[i])
        xmax = max(df1[i])
        df2[i] <- (df2[i]-xmin)/(xmax - xmin)
      }
    }
    df1 <- df2
  }
  return(df1)
}