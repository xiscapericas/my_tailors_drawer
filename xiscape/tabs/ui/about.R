about <- tabPanel(
  title = 'About', 
  value = 'about', 
  
  fluidRow(
    column(width = 2), 
    column(width = 8, align = 'center', 
           
           # Picture and Hi -----------------------------------------------------------
           fluidRow(
             
             column(width = 2), 
             column(width = 8, 
                    column(width = 12, 
                           fluidRow(
                             wellPanel(
                               width = 12, 
                               class = 'well-panel-picture', 
                               img(src = "img/xisca-pika-circle.png", height = 300, width = 300)
                             ))
                    ), # End center align column  
                    
                    column(width = 12, align = 'left', 
                           #h1("Hi I'm Xis"),
                           #h1('Data Scientist & Biomedical Engineer'), 
                           tags$div(class = 'description', 
                                    
                                    tags$p("Hi everyone! my name is Xisca and I'm a Senior Data Scientist from Mallorca (Spain). Currently I work as a Data Director for 
                                           a payment processor company, but I have also experience as product researcher and in consultancy for the data and advanced analytics world"),
                                    tags$p("I love data in all forms! Along the past years I have joined several data/hackathons 
                                           focused in different topics as shopping carts, accidents and weather predictions and so on. 
                                           You can check all of them on my projects section. But beyond this data-projects, I also love books and music. 
                                           I have a book blog review and a music channel. Feel free to check both on my outside projects section."), 
                                    tags$p("Finally, you can find my CV in the below button and my email and social network buttons as the bottom of the page. Ping me if you have any doubt or any interesting proposal!")
                                    ) 
                                    ), # End left align column
                    
                    # Download button ---------------------------------------------------------
                    fluidRow(
                      column(width = 12, align = 'center', 
                             downloadButton('downloadCV', 'Download resume', class = 'btn-primary')
                      )
                    )
                    
                    ), # end column width = 10
             column(width = 2)
             
                    ),  # End picture 
           
           
           
           # Timeline ----------------------------------------------------------------
           fluidRow(
             column(width = 1, align = 'left', 
                    includeHTML('html/timeline.html')
             ), 
             column(width = 11, br())
           )
           
           
           ),  # end column width 2 
    column(width = 2)
  )

         

  
) # end about

