# Contact buttons -----------------------------------------------------------
contact_button <- tags$div(class = 'contact_buttons_div', 
         
         
         fluidRow(
           br(),br(), 
           column(width = 6, align = 'left', 
                  fluidRow(br()),
                  fluidRow(
                    h2('Say hi!', style='display:inline;'), 
                    icon(name = 'envelope'), 
                    tags$a(href = "mailto: abc@example.com", h4('mfpericas@gmail.com', style = 'display:inline;margin-left:2rem;'))
                  )
           ), 
           column(width = 6, align = 'right', 
                  ## Adding button 
                  actionButton("linkeding_button",
                               "linkedin", 
                               icon(name = 'linkedin-in'), 
                               class = "btn-primary", 
                               onclick ="window.open('https://www.linkedin.com/in/maria-francisca-peric%C3%A0s/', '_blank')"
                  ),
                  actionButton("github_button",
                               "github", 
                               icon(name = 'github'), 
                               class = "btn-primary", 
                               onclick ="window.open('https://github.com/xiscapericas', '_blank')"
                  ),
                  actionButton("twitter_button",
                               "twitter", 
                               icon(name = 'twitter'), 
                               class = "btn-primary", 
                               onclick ="window.open('https://twitter.com/XiscaPericas', '_blank')"
                  )
                  
           )
           
         )
         
         
) # end contact button div 