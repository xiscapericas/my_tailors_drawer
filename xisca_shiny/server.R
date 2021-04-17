#------------------------------------------------
# Shiny Xis
# server.R
# Author: Xisca Pe 
# Date: 2020-12-28
#------------------------------------------------


shinyServer(function(input, output, session) {
  

  # home --------------------------------------------------------------------
  source("tabs/server/home.R", local = TRUE)
  
  # about --------------------------------------------------------------------
  source("tabs/server/about.R", local = TRUE)
  
  
})