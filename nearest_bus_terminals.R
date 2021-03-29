library(dplyr)
library(geosphere)

bus <- read.csv("data_geocoded/bus.csv", stringsAsFactors = FALSE)
hdb <- read.csv("data_geocoded/data_with_coor.csv", stringsAsFactors = FALSE)
findNearestDistance <- function(hdb, facility) {
  hdb_coord <- hdb %>% select(lon,lat)
  facility_coord <- facility %>% select(lon,lat)
  distances <- distm(hdb_coord, facility_coord, fun=distGeo)[1,] %>% round()
  min_dist <- min(distances)
  return(min_dist)
}

findCountWithinDistance <- function(hdb, facility, range) {
  hdb_coord <- hdb %>% select(lon,lat)
  facility_coord <- facility %>% select(lon,lat)
  distances <- distm(hdb_coord, facility_coord, fun=distGeo)[1,] %>% round()
  return(sum(distances <= range))
}

output <- data.frame()

for(i in 1:nrow(hdb)){
  hdb_entry <- hdb[i,]
  dist_nearest_bus_terminal <- findNearestDistance(hdb_entry, bus)
  bus_terminals_within_500m <- findCountWithinDistance(hdb_entry, bus, 500)
  bus_terminals_within_800m <- findCountWithinDistance(hdb_entry, bus, 800)
  bus_terminals_within_1000m <- findCountWithinDistance(hdb_entry, bus, 1000)
  row <- as.data.frame(cbind(dist_nearest_bus_terminal, bus_terminals_within_500m, bus_terminals_within_800m, bus_terminals_within_1000m))
  output <- rbind(output, row)
}
write.csv(output, 'feature_bus_terminals.csv', row.names = FALSE)
