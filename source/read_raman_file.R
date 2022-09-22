#' Read in Raman spectrum
#' @param file character vector: directory of Raman file
#' @return tibble containing Raman intensity
#' 

read_raman_file <- function(file) {
  df <- readr::read_delim(
    file,
    col_names = FALSE,
    col_types = cols()
    ) %>% 
    rename(
      `wavenumber_cm` = X1,
      `intensity` = X2
    ) %>% 
    # Add file info
    mutate(
      filename = file
    ) %>% 
    # Add band limits
    mutate(
      band = case_when(
        between(wavenumber_cm, 2040, 2300) ~ "CD",
        between(wavenumber_cm, 2800, 3100) ~ "CH",
        between(wavenumber_cm, 2500, 2700) ~ "SR"
      )
    )
}