#' @param a assimilation efficiency as value 0 - 1
#' @param F2 F of biomass at time 2 at%
#' @param F1 F of biomass at time 1 at%
#' @param FL F of label at%
#' @param t2 time 2 days
#' @param t1 time 1 days
#' @param sigma_a uncertainty in assimilation efficiency
#' @param sigma_F2 uncertainty in F2
#' @param sigma_F1 uncertainty in F1
#' @param sigma_FL uncertainty in FL
#' @return growth rate µ in units of t^-1


calculate_sigma_mu <- function(a, F2, F1, FL, t2, t1 = 0, 
                               sigma_a = 0, sigma_F2 = 0, 
                               sigma_F1 = 0, sigma_FL = 0) {
  return(
    sqrt(
      ((a*FL - F2) * sigma_F1)^2 +
        
        ((a*FL - F1) * sigma_F2)^2 +
        
        (a * (F1 - F2) * sigma_FL)^2 +
        
        (FL * (F1 - F2) * sigma_a)^2
    ) / ( 
      (t2-t1) * (a * FL - F1) * (a * FL - F2) 
    )
  )
}