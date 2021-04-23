#---------------------------
# about.R 
# SERVER
# Xisca Pe
#---------------------------


output$downloadCV <- downloadHandler(
  filename <- function() {
    paste("xiscapericas_CV_2020", "pdf", sep=".")
  },
  
  content <- function(file) {
    file.copy("www/xiscapericas_cv_2020.pdf", file)
  }
)
