#' Plot a Raman spectrum
#' @param df tibble: raman spectrum to plot
#' @return ggplot
#' 


plot_raman <- function(df) {
  df %>% ggplot(
    aes(
      x = wavenumber_cm,
      y = intensity,
    )
  ) +
    # Shade the CD stretch
    geom_area(data = df %>% filter(band == "CD"),
              fill = "red",
              alpha = 0.5
    ) +
    # Shade the CH stretch
    geom_area(data = df %>% filter(band == "CH"),
              fill = "blue",
              alpha = 0.5
    ) +
    # Add the zero line
    geom_hline(
      yintercept = 0,
      alpha = 0.5,
      color = "gray") +
    geom_line(color = "black") +
    theme_classic() +
    labs(
      x = TeX("Wavenumber $(cm^{-1})$"),
      y = "Intensity",
    )
}
