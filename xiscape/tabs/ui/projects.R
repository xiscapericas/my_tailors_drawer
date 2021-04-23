projects <- tabPanel(
  title = 'Projects', 
  value = 'projects', 
  
  fluidRow(
    column(width = 2), 
    column(width = 8, align = 'center', 
           
           tags$div(class = 'projects_div', 
                    
                    fluidRow(
                      column(width = 2), 
                      column(width = 8, align = 'center', 
                             
                        # Projects title ----------------------------------------------------------

                             
                       fluidRow(
                         column(width = 12, align = 'center', 
                                h1('Projects')
                        )),
                       
                       fluidRow(
                         column(
                            width = 12, align = 'left', 
                            h4('Here is a sample of different projects I have worked on. Just ping me if this info does not fullfill all your curiosity!')
                         )
                       ),

                        # Projects ----------------------------------------------------------------
                        fluidRow(
                          
                          wellPanel(
                            fluidRow(
                              column(width = 12, align = 'left',
                                h2("My website with R-shiny"),
                                br(), 
                                p("This project is focused on the development of my own website using tools that I use in my day-to day work. I am not a web-developer, however the Shiny Package make it easier for people like me to develop apps in a very easy way. Therefore I asked my self: why not using it to develop a website? And... why not my own website? And here is the first iteration of the result :)!"), 
                                br(), 
                                p('#webdevelopment', class = 'hashtag'), 
                                p('#R', class = 'hashtag'), 
                                p('#Shiny', class = 'hashtag')
                              )
                              
                            ), 
                            fluidRow(
                              column(width = 12, align = 'right', 
                                     actionButton("button_project_xiscashiny", 
                                                  "View Project", class = 'btn-secundary',
                                                  onclick ="window.open('https://github.com/xiscapericas/my_tailors_drawer/tree/main/xiscape', '_blank')"
                                                  
                                     )
                                     )
                            )
                          ),
                          
                          wellPanel(
                            fluidRow(
                              column(width = 12, align = 'left', 
                                     h2("Nightingale 30 Days Challenge"),
                                     br(), 
                                     p("#30diasdegrafico was a challenge to commemorate the 200 year birth anniversary of Florence Nightingale (12th May 1820). To complete the challenge, all participant had to publish one graphic per day for 30 days. The type of graph was defined by the crators of the challenge but what data to use or style was decided by each participant. The unique condition was to do everything using only R. Click the button if you want to check all my graphs!"), 
                                     p('#graph', class = 'hashtag'), 
                                     p('#R', class = 'hashtag'), 
                                     p('#ggplot2', class = 'hashtag')
                              )
                            ),
                            fluidRow(
                              column(width = 12, align = 'right', 
                                     ## Button check it out 
                                     actionButton("button_project_nightingale", 
                                                  "View Project", class = 'btn-secundary', 
                                                  onclick ="window.open('https://github.com/xiscapericas/my_tailors_drawer/tree/main/nightingale_challenge', '_blank')"
                                     )
                              )
                            )
                          ), # End WellPanel 
                          
                          
                          wellPanel(
                            fluidRow(
                              column(width = 12, align = 'left', 
                                     h2("Cajamar Predictive"),
                                     br(), 
                                     p("Here you have the project me and my my teammate Alex✿ did for the 2017 Cajamar Hackathon, for which we received the First Winning Prize! The aim of the challenge was to develope the best Association Rules predictive model that could work as a recommendation engine for bank products."), 
                                     p('#hackathon', class = 'hashtag'), 
                                     p('#R', class = 'hashtag'), 
                                     p('#ML', class = 'hashtag')
                              )
                            ), 
                            fluidRow(
                              column(width = 12, align = 'right', 
                                     ## Button check it out 
                                     actionButton("button_project_cajamar_video", 
                                                  "View presentation", 
                                                  icon(name = 'youtube'), 
                                                  class = 'btn-secundary', 
                                                  onclick ="window.open('https://youtu.be/F-R13FPPysk?t=2668', '_blank')"), 
                                     actionButton("button_project_cajamar", 
                                                  "View project",  class = 'btn-secundary', 
                                                  onclick ="window.open('https://random-6.github.io/cajamar_predictive/', '_blank')")
                              )
                            )
                          ), # End WellPanel 
                          
                          
                          wellPanel(
                            fluidRow(
                              column(width = 12, align = 'left', 
                                     h2("Open DOGC"),
                                     br(), 
                                     p("This project was inspired by the opengov.cat platform. The aim was to set-up an automated system to perform web-scrapping the official webpage, download and process text from released documents and extract political information"), 
                                     p('#python', class = 'hashtag'), 
                                     p('#NLP', class = 'hashtag'), 
                                     p('#OpenDocuments', class = 'hashtag')
                              )
                            ), 
                            fluidRow(
                              column(width = 12, align = 'right', 
                                     ## Button check it out 
                                     actionButton("button_project_opendogc", 
                                                  "View project",  class = 'btn-secundary', 
                                                  onclick ="window.open('https://github.com/Random-6/open-dogc', '_blank')")
                              )
                            )
                          ),  # End WellPanel 
                          
                          
                          wellPanel(
                            fluidRow(
                              column(width = 12, align = 'left', 
                                     h2("RSA Publication"),
                                     br(), 
                                     p("Research assessing respiratory sinus arrhythmia (RSA) index through signal processing methods to improve the differentiation of ischemical and dilated cardiomyopathies"), 
                                     p('#biomedicalengineering', class = 'hashtag'), 
                                     p('#signalprocessing', class = 'hashtag'), 
                                     p('#Matlab', class = 'hashtag')
                              ) 
                            ),
                            fluidRow(
                              column(width = 12, align = 'right', 
                                     ## Button check it out 
                                     actionButton("button_project_rsa", 
                                                  "Read Paper",  
                                                  icon(name = 'book'), 
                                                  class = 'btn-secundary', 
                                                  onclick ="window.open('https://pubmed.ncbi.nlm.nih.gov/30441432', '_blank')")
                              )
                            )
                          ) # End WellPanel 
                        ) # End FludRow Panels
          ),  #end column 8 
          column(width = 2)
        )
                    
                    
      ) # end div class project
           
           
    ), # end column width 8
    column(width = 2)
  ) # end fluidRow main 

         
) #end tabPanel
  
 