library(dplyr)
library(geosphere)

bus <- read.csv("data_geocoded/bus.csv", stringsAsFactors = FALSE)
hdb <- read.csv("data_geocoded/hdb.csv", stringsAsFactors = FALSE)[1:100,]

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
  dist_nearest_bus_stop <- findNearestDistance(hdb_entry, bus)
  bus_stops_within_500m <- findCountWithinDistance(hdb_entry, bus, 500)
  bus_stops_within_2000m <- findCountWithinDistance(hdb_entry, bus, 2000)
  row <- as.data.frame(cbind(dist_nearest_bus_stop, bus_stops_within_500m, bus_stops_within_2000m))
  output <- rbind(output, row)
}
View(output)


