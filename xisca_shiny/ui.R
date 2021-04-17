#------------------------------------------------
# Shiny Xis
# ui.R
# Author: Xisca Pe 
# Date: 2020-12-28
#------------------------------------------------


source('tabs/ui/home.R')
source('tabs/ui/about.R')
source('tabs/ui/projects.R')
source('tabs/ui/outliers.R')
source('tabs/ui/contact_button.R')


# UI ----------------------------------------------------------------------
shinyUI(
  fluidPage(
    
    
    
  # Favicon -----------------------------------------------------------------
  ##-- Favicon ----
  tags$head(
    tags$link(rel = "shortcut icon", href = "img/favicon.ico"),
    #-- biblio js ----
    tags$link(rel="stylesheet", type = "text/css", href = "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css"),
    tags$link(rel="stylesheet", type = "text/css", href="//netdna.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap.min.css"), 
    tags$link(rel="stylesheet", type = 'text/css', href = "timeline.css"),
    tags$link(rel="stylesheet", type = 'text/css', href = "xisca_shiny.css")
  ),

    

  # Logo --------------------------------------------------------------------
  #div(style="padding: 1px 0px; width: '100%'",titlePanel(title="", windowTitle = "Shiny Xisca")),
  ## Image 
  #div(class = 'cat-icon-div', 
    #conditionalPanel(
      #  condition = "input.navbar == 'home'",
    #img(src="img/cat--v2.png", height = "50px")
    #  ), 
    #conditionalPanel(
      #condition = "input.navbar != 'home'",
      #img(src="img/cat--v2--b.png", height = "50px"),
      #  tags$a(href = route_link('home'), target="_target",  'Back to home', style = 'display:inline;color:black;vertical-align:-10px;font-size:22px')
    #),
    #style = "padding-right:100px;"
    #),

    


  # Header ------------------------------------------------------------------
  ## Move navbar to right alignment 
  tags$head(
    tags$style(HTML("
                    .navbar .navbar-nav {float: right}
                    .navbar .navbar-header {float: right}
                    "))
    ),
 
  tags$head(tags$script(type="text/javascript", src = "img_code.js")),

  # Tabs --------------------------------------------------------------------
  fluidRow(
           
           navbarPage(
             title = '', 
             id = "navbar",
             selected = "home",
             fluid = TRUE,
             ##-- Abas ----
             home, 
             about, 
             projects,
             outliers
           ), ## End Header 
           
           ## Button 
           fluidRow(
             column(width = 2), 
             column(width = 8, 
                    # Contact Button ----------------------------------------------------------
                    contact_button,
                    # Footer ------------------------------------------------------------------
                    div(class = "footer",
                        includeHTML("html/footer.html")
                    ))

                    
                    
                    ), 
             column(width = 2)
           )
  )

  
  



                  
) # End shinyUI
