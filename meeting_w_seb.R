# meeting w seb

tibble(m = 5, sd = 3) |> 
  crossing(n = c(3, 30, 300, 3000, 300000)) |> 
  mutate(
    data = pmap(list(m = m, sd = sd, n = n), 
                function(m, sd, n) rnorm(n = n, mean = m, sd = sd)),
    measured_mean = map_dbl(data, mean), measured_sd = map_dbl(data, ~sd(.x)), 
    se_of_the_mean = measured_sd / sqrt(n)
    )


