#
# This is a Shiny web application. You can run the application by clicking
# the 'Run App' button above.
#
# Find out more about building applications with Shiny here:
#
#    http://shiny.rstudio.com/
#

# Required libraries
library(shiny)
library(ggplot2)
library(ggthemes)
library(plotly)

# Required sourced functions
plot_rel_error <- source("shiny_app/plot_rel_error.R")
calculate_mu <- source("shiny_app/calculate_mu.R")
calculate_sigma_mu <- source("shiny_app/calculate_sigma_mu.R")
create_test_dataset <- source("shiny_app/create_test_dataset.R")



# Define UI for application
ui <- fluidPage(

    # Application title
    titlePanel("Raman SIP Incubation App"),
    
    # Input for test dataset options
    column(4,
      textInput("fls", label = "Label strengths (%), comma separated", value = "5, 10, 15, 20, 25, 30, 35, 40, 45, 50"),
      textInput("assim_a", label = "Hydrogen Assimilation Efficiency (a) between 0 and 1", value = 1),
      textInput("sassim_a", label = "Uncertainty in hydrogen assimilation efficiency", value = 0), 
      textInput("inc_times", label = "Incubation Times (days), comma separated", value = "1, 5, 7, 30, 60, 100"),
      textInput("sfl", label = "SE of Label Strength (%)", value = 0),
      textInput("sf2", label = "SE of Raman 2H measurements (%)", value = 2.5)
    ),

    # Input for plot options
    column(4,
      numericInput("dt", label = "Highlighted Incubation Time", value = 30),
      numericInput("f_label", label = "Highlighted Tracer Strength (%)", value = 20),
      sliderInput("xlimits", label = "Plot x limits", value = c(0, 300), min = 0, max = 1000)
    ),
    
    # Outputs
    textOutput("greeting"),
    plotOutput("p_rel_error")
    
)

# Define server logic required to plot
server <- function(input, output) {
  # Define reactive things
  concat_str <- reactive({
    paste("A", input$dt, "day incubation time with", input$f_label, "% D2O:")
  })
  
  fls_cat <- reactive({
    as.numeric(str_split_1(input$fls, pattern = ","))
  })
  
  incs_cat <- reactive({
    as.numeric(str_split_1(input$inc_times, pattern = ","))
  })
  
  simulated_data <- reactive({
    create_test_dataset(
      a = input$assim_a,
      sigma_a = input$sassim_a,
      FL = fls_cat(), # a reactive that has parsed the user input
      dt = incs_cat(),
      sigma_FL = input$sfl,
      sigma_F2 = input$sf2
    )
  })
  
  
  # Define outputs
  output$greeting <- renderText(concat_str())
  output$p_rel_error <- renderPlot({
    plot_rel_error(
      simulated_data(),
      d_t = input$dt,
      f_label = input$f_label,
      xlimits = input$xlimits,
      include_legend = TRUE
      )
  })

   
}

# Run the application 
shinyApp(ui = ui, server = server)
