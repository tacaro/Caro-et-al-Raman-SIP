#' @param d_t dbl incubation time in days
#' @param f_label strength of the SIP label to highlight in at%
#' @param xlimits character vector: of x axis limits for plot
#' @param include_legend logical: does the plot include the legend?

require(tidyverse)

plot_rel_error <- function(d_t, f_label, xlimits, include_legend = TRUE) {
  upper_lim <- test_dataset |> 
    filter(dt == d_t) |> 
    filter(rel_error <= 0.5) |>
    filter(FL == f_label) |> 
    pull(TD.days) |> 
    min()
  
  lower_lim <- test_dataset |> 
    filter(dt == d_t) |> 
    filter(rel_error <= 0.5) |>
    filter(FL == f_label) |> 
    pull(TD.days) |> 
    max()
  
  test_dataset %>%
    # Argument input here to filter dataset to correct incubation time:
    filter(dt == d_t) |> 
    filter(FL > 15) |> 
    ggplot() +
    aes(
      x = TD.days,
      y = rel_error,
      color = as.factor(FL)
    ) +
    annotate(geom = "segment", x = lower_lim, xend = lower_lim, y = 0, yend = 0.5, color = "gray") +
    annotate(geom = "segment", x = upper_lim, xend = upper_lim, y = 0, yend = 0.5, color = "gray") +
    #annotate(geom = "label", x = upper_lim, y = 0.1, color = "black", label = paste0(round(upper_lim, 1), " d")) +
    #annotate(geom = "label", x = lower_lim, y = 0.1, color = "black", label = paste0(round(lower_lim, 1), " d")) +
    geom_line(linewidth = 1) +
    scale_color_viridis_d(option = "plasma", begin = 0.1, end = 0.9) +
    # Secondary axis with ticks marked
    scale_x_continuous(
      sec.axis = dup_axis(
        breaks = c(round(upper_lim, 1), round(lower_lim, 1))
      )
    ) +
    # Secondary axis with growth rate:
    # scale_x_continuous(
    #   sec.axis = sec_axis(~ log(2)/., breaks = c(0.1, 0.01, 0.002)),
    #   
    #   ) +
    scale_y_continuous(labels = scales::percent) + 
    coord_cartesian(
      ylim = c(0, 0.5),
      xlim = xlimits,
      expand = FALSE
    ) +
    labs(
      x = "Generation Time (Days)",
      y = latex2exp::TeX("Relative Error (%) =     $\\frac{\\sigma_{\\mu}}{\\mu}$"),
      color = latex2exp::TeX("Tracer Strength $^{2}H \\%$"),
      title = paste(d_t, "day")
    ) +
    theme_classic() +
    theme(
      legend.position = if_else(include_legend, "bottom", "NA"),
      axis.text.x.top = element_text(angle = 45, hjust = 0),
      axis.title.x.top = element_blank(),
      #panel.border = element_rect(color = "black", size = 2, fill = NA),
      #axis.ticks = element_line(linewidth = 1)
    )
}