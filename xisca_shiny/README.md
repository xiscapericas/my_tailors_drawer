# Xisca Shiny


## 1. Intro

I have wanted to build my own website for quite some time now. As I am not a webdeveloper (and at the beginnig of this I did know very little of HTML/CSS/Javascript) I did check different options as getting a 3rd party architecture (as wixsite) or use one of the already-done github.io... but then I found the [Voronoys Shiny App](https://voronoys.shinyapps.io/voronoys/) which looks like a website, and the light came! I could build my own website using R+Shiny even if I'm not a webdeveloper!

And here I am, and that's the reason why this project is here! 

However, before going further, I want to thank my collegue [Miquel Mir](https://www.miquelmirmiquel.com/) (UX and graphi designer) for all his help on the UX/UI part of the webpage. Thanks a lot 🌸

## 2. Xisca-Shiny Description 

Actually my own website is not really a website, it is a [Shiny App](https://shiny.rstudio.com/) that looks and feels like a webpage. 

As any other Shiny App page, xisca shiny main architecture is composed by three files: 

- global.R: which loads all needed R-libraries and runs the app. 
- ui.R: which builds the main User Interface. This main ui.R contains the source of the tabs, but also the main fluidPage (with the head an body elements). 
- server.R: this is the part of the code that contains the server funtionality. 

Then, we have  secundary files: 
- tabs folder: this folder contains the ui.R and server.R for each tab of the app (i.e: about, projects, etc). Those tabs have its own ui/server duo as the components of the app have been splitted into modules (so it is easy to develop and debug). 
- www folder: this contains CSS, Javascript and images files that are loaded by the app. 
- html: folder that contains .html files with parts of the app (developed purely using HTML). 

To deploy the app I have used [shinyapps.io](https://www.shinyapps.io/). ShinyApps Io is a service that allows you to deploy your shiny apps in a very easy (really really easy) way. You only need to install the  `rsconnect` package, which will take care of uploading all necessary files and do the app deploy. 


## 3. Resources	📚

If you like the app or if you are interested in reading more about it, here are some resources that I recommend you to have a look: 

- [Mastering Shiny](https://mastering-shiny.org/index.html): o'Reilly book to know about Shiny Apps
- [Shiny RInterface](https://unleash-shiny.rinterface.com/web-intro.html): a second book to understand how the webdevelopment works for a Shiny App. 
- [Dean Attali Webpage](https://deanattali.com/: this is a blog from a Shiny Expert. He has developed several shiny packages and is always blogging about nice tricks to apply on your shiny apps. 


## 4. Contact details	✉️

**Author**: Xisca Pericas - mfpericas@gmail.com

**Last update date** : 2021-04-17
