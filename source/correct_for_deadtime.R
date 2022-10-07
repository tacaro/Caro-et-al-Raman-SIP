#' @param N ion counts of a given type
#' @param dwell_time_us counts/px -- reported as dwell time per px in µs
#' @param npx number of pixels, or pixel area
#' @param deadtime_ns deadtime of the instrument in nanoseconds. Most commonly 44ns.
#' @return a dataframe with cd_frac and cd_pc appended as columns
#' 
#' 

correct_for_deadtime <- function(N, dwell_time_us, npx, deadtime_ns) {
  dwell_time_s = dwell_time_us / 1e6 # convert from µs to s
  deadtime_s = deadtime_ns / 1e9 # convert from ns to s
  
  cps_ion = N / (dwell_time_s * npx) # counts per second of the ion equals counts divided by seconds 
  # (seconds = pixel dwell time times number of pixels)
  N_dtc = N / (1 - cps_ion * deadtime_ns * 1e-9) # DTC ion count equals 
  # number ions divided by (1 - counts per second * deadtime_ns * 1e-9)
  return(N_dtc)
  
}