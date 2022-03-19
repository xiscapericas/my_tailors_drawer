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
                               p("My new project is focused on the design and development of the webpage you are now checking. This page has been built completly using R, Shiny and a bit of HTML/Javascript/CSS"),  
                               tags$div(class = 'p_with_href', 
                                 tags$p("Thanks a lot to ", style = "display:inline"),   
                                 tags$a(href = 'https://www.miquelmirmiquel.com/', "Miquel Mir", style = "margin-top: 0; margin-bottom: 1.5rem; font-size: 16px; color: gray; line-height:2.7rem; display: inline; color: var(--pink-color-bold9)"), tags$p(" (UI/UX designer) for all his help in building this UI!", style = "display:inline")
                               )
                               
                        ), 
                        column(width = 1), 
                        fluidRow(
                          #column(width = 10), 
                          column(width = 12, align = 'right', 
                                 ## Button check it out 
                                 actionButton("button_check_it_out_xisca_shiny", 
                                              "Check it out", class = 'btn-secundary',
                                              onclick ="window.open('https://github.com/xiscapericas/my_tailors_drawer/tree/main/xiscape', '_blank')"
                                 )
                          )
                        )
                      ), # End WellPanel 
                      
                      wellPanel(
                        width = 12, align = 'left', 
                        column(width = 11, 
                               h2("Hola a todos, "),
                               h2("Aquí os comparto mi último Book-Review: El nombre del mundo es bosque"), 
                               br(), 
                               p("-¿Sois del tiempo del sueño o del tiempo del mundo?- le preguntó el viejo al final. -Del tiempo del mundo.", style = 'font-style:italic')
                        ), 
                        column(width = 1), 
                        fluidRow(
                          #column(width = 10), 
                          column(width = 12, align = 'right', 
                                 ## Button check it out 
                                 actionButton("button_check_it_out_xisca_shiny_2", 
                                              "Leelo aquí", class = 'btn-secundary',
                                              onclick ="window.open('https://mfpericas.wixsite.com/mientrasleiamos/post/de-lo-que-debiera-ser-el-nombre-del-mundo', '_blank')"
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
  
  