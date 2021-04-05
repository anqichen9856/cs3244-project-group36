library(dplyr)
library(geosphere)

mrt <- read.csv("data_geocoded/mrt_lrt.csv", stringsAsFactors = FALSE)
hawker_centers <- read.csv("data_geocoded/hawker_centers.csv", stringsAsFactors = FALSE)
hospitals <- read.csv("data_geocoded/hospitals_clinics.csv", stringsAsFactors = FALSE)

hdb <- read.csv("data_geocoded/data_with_coor.csv", stringsAsFactors = FALSE)

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
  dist_nearest_mrt <- findNearestDistance(hdb_entry, mrt)
  dist_nearest_hawker_centers <- findNearestDistance(hdb_entry, hawker_centers)
  dist_nearest_hospitals <- findNearestDistance(hdb_entry, hospitals)
  row <- as.data.frame(cbind(dist_nearest_mrt, dist_nearest_hawker_centers, dist_nearest_hospitals))
  output <- rbind(output, row)
}

write.csv(output, 'feature_JKL.csv', row.names = FALSE)

