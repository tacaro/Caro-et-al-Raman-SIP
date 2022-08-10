#' @param df a dataframe containing three columns: wavenumber_cm, intensity, and band
#' @return a dataframe with cd_frac and cd_pc appended as columns
#' 
#' 

calculate_CD <- function(df) {
  df_summarized <- 
    df %>% group_by(band) %>% 
    summarize(
      area = sum(intensity)
    ) %>% 
    ungroup() %>% 
    filter(!is.na(band))
  df_cd_area <- df_summarized %>% filter(band == "CD") %>% pull(area)
  df_ch_area <- df_summarized %>% filter(band == "CH") %>% pull(area)
  df_cd_frac <- df_cd_area / df_ch_area
  df_cd_pc <- df_cd_frac * 100
  
  return(
    df %>% mutate(
      cd_frac = df_cd_frac,
      cd_pc = df_cd_pc
    )
  )
}