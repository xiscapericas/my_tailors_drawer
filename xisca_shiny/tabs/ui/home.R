home <- tabPanel(
  title = 'Home', 
  value = 'home', 
  
  fluidRow(
    column(width = 2), 
    column(width = 8, align = 'center', 
           
           
           # Picture and Hi -----------------------------------------------------------
           fluidRow(
             column(width = 2),
             column(width = 8, 
                    fluidRow(
                      wellPanel(
                        width = 12, 
                        class = 'well-panel-picture', 
                        #style = 'background:transparent;border-color:transparent;margin-top:0px;margin-bottom:0px;box-shadow: 0px 0px 0px;padding-top:0px', 
                        img(src = "img/xisca-profile-250-circle.png", height = 210, width = 210),
                        br(), br(), 
                        h1("Hi I'm Xis"),
                        h1('Data Scientist & Biomedical Engineer')
                      )
                    ),
                    
                    ## Adding my latest update text 
                    fluidRow(), 
                    fluidRow(), 
                    fluidRow(
                      h3('Check my latest updates')
                    ),
                    ## Space
                    #fluidRow(br()), 
                    # Latest Updates -----------------------------------------------------------
                    fluidRow(
                      wellPanel(
                        width = 12, align = 'left', 
                        column(width = 11, 
                               h2("Hi people, "),
                               h2("I have recently added a new project called Xisca-Shiny"), 
                               br(), 
                               p("My new project is focused on the design and development of the webpage you are now checking. This page has been built completly using R, Shiny and a bit of HTML/Javascript/CSS. If you want to learn how I built it and find some resources to build your own, just click on the button below.")
                        ), 
                        column(width = 1), 
                        fluidRow(
                          #column(width = 10), 
                          column(width = 12, align = 'right', 
                                 ## Button check it out 
                                 actionButton("button_check_it_out_xisca_shiny", 
                                              "Check it out", class = 'btn-secundary'
                                              #onclick ="window.open('https://github.com/xiscapericas/my_tailors_drawer/tree/main/nightingale_challenge', '_blank')"),
                                 )
                          )
                        )
                      ), # End WellPanel 
                      
                      wellPanel(
                        width = 12, align = 'left', 
                        column(width = 11, 
                               h2("Hola a todos, "),
                               h2("Aquí os comparto mi último Book-Review: Momo"), 
                               br(), 
                               p("Existe una cosa muy misteriosa, pero muy cotidiana. [...] Casi todos se limitan a tomarla como viene, sin hacer preguntas. Esta cosa es el tiempo", style = 'font-style:italic')
                        ), 
                        column(width = 1), 
                        fluidRow(
                          #column(width = 10), 
                          column(width = 12, align = 'right', 
                                 ## Button check it out 
                                 actionButton("button_check_it_out_xisca_shiny_2", 
                                              "Leelo aquí", class = 'btn-secundary',
                                              onclick ="window.open('https://mfpericas.wixsite.com/mientrasleiamos/post/momo-y-la-cuesti%C3%B3n-del-tiempo', '_blank')"
                                 )
                          )
                        )
                      ) # End WellPanel 
                      
                    ) # End fluidrow latest updates 
                    
                    
                    
             )
             
           )  # end figure 
           
           # Latest Tweets -----------------------------------------------------------
           
           
           ), # end column width 8
    column(width = 2)
  )


) # end home
  
  