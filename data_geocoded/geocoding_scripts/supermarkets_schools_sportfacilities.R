library(dplyr)
library(geosphere)

supermarkets <- read.csv("supermarkets.csv", stringsAsFactors = FALSE)
schools <- read.csv("schools.csv", stringsAsFactors = FALSE)
sport_facilities <- read.csv("sport_facilities.csv", stringsAsFactors = FALSE)

hdb <- read.csv("data_with_coor.csv", stringsAsFactors = FALSE)

findNearestDistance <- function(hdb, facility) {
  hdb_coord <- hdb %>% select(lon,lat)
  facility_coord <- facility %>% select(lon,lat)
  distances <- distm(hdb_coord, facility_coord, fun=distGeo)[1,] %>% round()
  min_dist <- min(distances)
  return(min_dist)
}

output <- data.frame()

for(i in 1:nrow(hdb)){
  hdb_entry <- hdb[i,]
  dist_nearest_supermarkets <- findNearestDistance(hdb_entry, supermarkets)
  dist_nearest_schools <- findNearestDistance(hdb_entry, schools)
  dist_nearest_sport_facilities <- findNearestDistance(hdb_entry, sport_facilities)
  row <- as.data.frame(cbind(dist_nearest_supermarkets, dist_nearest_schools, dist_nearest_sport_facilities))
  output <- rbind(output, row)
}

write.csv(output, 'feature_PQR.csv', row.names = FALSE)