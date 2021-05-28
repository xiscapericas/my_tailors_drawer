outliers <- tabPanel(
  title = 'Outliers', 
  value = 'outliers', 
  
  fluidRow(
    column(width = 2), 
    column(width = 8, align = 'center', 
           
           tags$div(class = 'projects_div', 
                    
                    fluidRow(
                      column(width = 2), 
                      column(width = 8, align = 'center', 
         
                       # Projects Title  ---------------------------------------------------------
                       
                       
                       fluidRow(
                         column(width = 12, align = 'center', 
                                h1('Outlier projects')
                         )),
                       
                       fluidRow(
                         column(
                           width = 12, align = 'left', 
                           h4('Not all my work has been around Data Science. Here are some links to my other reading/musical/knitting projects!'), 
                         )
                       ),

                      # Projects ----------------------------------------------------------------
                      fluidRow(
                        
                        wellPanel(
                          fluidRow(
                            column(width = 12, align = 'left',
                                   h2("Mientras leíamos"),
                                   br(), 
                                   p("This is my book-review blog. Here is where I talk about books that I have read. You will find fantasy, drama and japanese literature!"), 
                                   br(), 
                                   p('#books', class = 'hashtag'), 
                                   p('#fantasy', class = 'hashtag'), 
                                   p('#review', class = 'hashtag')
                            )
                            
                          ), 
                          fluidRow(
                            column(width = 12, align = 'right', 
                                   actionButton("button_project_mientrasleiamos", 
                                                "Check Blog", 
                                                icon('book'), 
                                                class = 'btn-secundary', 
                                                onclick ="window.open('https://mfpericas.wixsite.com/mientrasleiamos', '_blank')"
                                                
                                   )
                            )
                          )
                        ), # end well panel 
                        
                        
                        wellPanel(
                          fluidRow(
                            column(width = 12, align = 'left',
                                   h2("Hi I'm Xis"),
                                   br(), 
                                   p("This is my music channel. You can find my on Youtube or checkout some of my scores on MuseScore"), 
                                   p('#music', class = 'hashtag'), 
                                   p('#musescore', class = 'hashtag'), 
                                   p('#youtube', class = 'hashtag')
                            )
                            
                          ), 
                          fluidRow(
                            column(width = 12, align = 'right', 
                                   actionButton("button_project_hiimxis_youtube", 
                                                "View youtube", 
                                                icon(name = 'youtube'), 
                                                class = 'btn-secundary', 
                                                onclick ="window.open('https://www.youtube.com/channel/UCgLuX7RyR3ZqhkvgLB9IJew', '_blank')"), 
                                   actionButton("button_project_hiimxis_musescore", 
                                                "Check scores",  icon('music'), class = 'btn-secundary', 
                                                onclick ="window.open('https://musescore.com/user/37860943?share=copy_link', '_blank')")
                                  )
                            )
                        ), # end well panel 
                        
                        
                        wellPanel(
                          fluidRow(
                            column(width = 12, align = 'left',
                                   h2("Knitting"),
                                   br(), 
                                   p("Knitting is also one of my hobbies. It helps me to be creative and also to relax. Best thing to do while watching a movie"), 
                                   p('#knitting', class = 'hashtag'), 
                                   p('#crochet', class = 'hashtag'), 
                                   p('#amigurumi', class = 'hashtag')
                            )
                            
                          ), 
                          fluidRow(
                            column(width = 12, align = 'right', 
                                   actionButton("button_project_cajamar", 
                                                "View art",  class = 'btn-secundary', 
                                                onclick ="window.open('https://www.deviantart.com/xiscape', '_blank')")
                            )
                          )
                        ) # end well panel 
                        
                        
                      ) # End FludRow Panels
                      ),  #end column 8 
                      column(width = 2)
                    )
                    
                    
           ) # end div class project
           
           
    ), # end column width 8
    column(width = 2)
  ) # end fluidRow main 
  
  
) #end tabPanel
