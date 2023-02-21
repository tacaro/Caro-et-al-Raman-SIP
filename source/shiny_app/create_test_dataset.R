#' Create a test dataset
#' @param a dbl: assimilation efficiency as value 0 - 1\
#' @param sigma_a dbl: uncertainty in a
#' @param sigma_FL dbl: uncertainty in FL
#' @param sigma_F1 dbl: uncertainty in F1
#' @param sigma_F2 dbl: uncertainty in F2 determined by our calibration
#' @param dt character vector: choice of incubation times
#' @param F1 dbl: F of biomass at time 1
#' @param FL character vector: choice of label strengths 
#' @return test_dataset: a tibble

require(dplyr)

create_test_dataset <- function(a = 1, F1 = 0, sigma_a = 0,
                                sigma_FL = 0, sigma_F1 = 0,
                                sigma_F2 = 2.5, 
                                FL = c(5, 10, 15, 20, 25, 30, 35, 40, 45, 50),
                                dt = c(1, 5, 7, 30, 60, 100)) {
  
  test_dataset <- tibble(
    a = a, # assimilation factor is 1: assume autotrophic growth
    F1 = F1, # ppm
    # errors
    sigma_a = sigma_a, # no uncertainty in alpha because defined cultures
    sigma_FL = sigma_FL, # uncertainty in FL is unknown: determine with pipetting?
    sigma_F1 = sigma_F1, # at%: approx as zero because 2H natabund is negligible
    sigma_F2 = sigma_F2 # at%: RMSE of proxy
  ) %>% 
    crossing(
      FL = FL, # label strengths
      dt = dt, # incubation times
      frac_F2 = seq(-5, 0, by = 0.01) %>% exp() # microbial fractional progression to FL
    ) %>% 
    mutate(
      F2 = F1 + FL * frac_F2, # generate F2 values in at% 
      mu.d = calculate_mu(a = a, F2 = F2, F1 = F1, FL = FL, t2 = dt),
      TD.days = log(2) / mu.d,
      sigma_mu.d = calculate_sigma_mu(
        a = a, F2 = F2, F1 = F1, FL = FL, t2 = dt,
        sigma_a = sigma_a, 
        sigma_F2 = sigma_F2, 
        sigma_F1 = sigma_F1, 
        sigma_FL = sigma_FL
      ),
      rel_error = sigma_mu.d / mu.d
    ) %>%
    filter(!is.na(mu.d), !is.na(rel_error))
  
  return(test_dataset)
}