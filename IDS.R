
# Load required packages with proper error handling
packages <- c("sf", "raster", "terra", "dplyr", "caret", "randomForest", 
              "ggplot2", "tidyr", "pROC")

for(pkg in packages) {
  if(!require(pkg, character.only = TRUE, quietly = TRUE)) {
    cat(paste0("Installing package: ", pkg, "\n"))
    install.packages(pkg, dependencies = TRUE)
    library(pkg, character.only = TRUE)
  }
}

# Step 1: Load the Khulna City boundary
khulna_boundary <- st_read("D:/MURP/Thesis/New Analysis/KCC.shp")
khulna_boundary <- st_transform(khulna_boundary, crs = 4326)  # Reproject to WGS84

# Step 2: Load raster data with error handling
raster_paths <- list(
  population_density = "D:/MURP/Thesis/New Analysis/Population_Density.tif",
  livelihood_index = "D:/MURP/Thesis/New Analysis/Livlihood_Index.tif",
  poverty_index = "D:/MURP/Thesis/New Analysis/Poverty_Index.tif",
  infrastructure_index = "D:/MURP/Thesis/New Analysis/Infrastructure_Index.tif",
  lst = "D:/MURP/Thesis/New Analysis/LST.tif"
)

raster_data <- list()
for(name in names(raster_paths)) {
  tryCatch({
    raster_data[[name]] <- raster(raster_paths[[name]])
    cat(paste0("Loaded raster: ", name, "\n"))
  }, error = function(e) {
    cat(paste0("Error loading ", name, ": ", e$message, "\n"))
  })
}

# Step 3: Load vector data with error handling
vector_paths <- list(
  informal_settlement = "D:/MURP/Thesis/New Analysis/Slum.shp",
  drainage = "D:/MURP/Thesis/New Analysis/Drain.shp",
  buildings = "D:/MURP/Thesis/New Analysis/Building.shp",
  railine = "D:/MURP/Thesis/New Analysis/Rail.shp",
  road = "D:/MURP/Thesis/New Analysis/Road.shp",
  road_intersection = "D:/MURP/Thesis/New Analysis/Intersection.shp",
  waterbody = "D:/MURP/Thesis/New Analysis/Waterbody.shp",
  sts = "D:/MURP/Thesis/New Analysis/STS.shp"
)

vector_data <- list()
for(name in names(vector_paths)) {
  tryCatch({
    vector_data[[name]] <- st_read(vector_paths[[name]])
    # Repair invalid geometries
    vector_data[[name]] <- st_make_valid(vector_data[[name]])
    cat(paste0("Loaded vector: ", name, "\n"))
  }, error = function(e) {
    cat(paste0("Error loading ", name, ": ", e$message, "\n"))
  })
}

# Step 4: Load illegal dumping data
illegal_dumping <- st_read("D:/MURP/Thesis/New Analysis/IDS2.shp")
illegal_dumping <- st_zm(illegal_dumping, drop = TRUE, what = "ZM")  # Remove Z value

# Step 5: Reproject all data to a common CRS
common_crs <- st_crs(khulna_boundary)

# Reproject rasters
for(name in names(raster_data)) {
  raster_data[[name]] <- projectRaster(raster_data[[name]], crs = common_crs)
}

# Reproject vectors
for(name in names(vector_data)) {
  vector_data[[name]] <- st_transform(vector_data[[name]], crs = common_crs)
}

# Reproject illegal dumping data
illegal_dumping <- st_transform(illegal_dumping, crs = common_crs)
common_crs <- st_crs(khulna_boundary)$proj4string
# Step 6: Extract predictor variables at illegal dumping locations
# Extract raster values
for(name in names(raster_data)) {
  illegal_dumping[[name]] <- raster::extract(raster_data[[name]], illegal_dumping)
}

# Function to calculate minimum distance with progress indicator
calculate_min_distance <- function(points, features) {
  cat(paste0("Calculating distances...\n"))
  distances <- st_distance(points, features)
  min_distances <- apply(distances, 1, min)
  return(min_distances)
}

# Calculate distances for all vector features
for(name in names(vector_data)) {
  distance_col <- paste0("distance_to_", name)
  illegal_dumping[[distance_col]] <- calculate_min_distance(illegal_dumping, vector_data[[name]])
}

# Step 7: Prepare data for modeling
model_data <- st_drop_geometry(illegal_dumping)
# Check for and report NA values before removing
na_count <- sapply(model_data, function(x) sum(is.na(x)))
print("NA counts before removal:")
print(na_count)

# Remove rows with NA values
model_data <- na.omit(model_data)
cat(paste0("Rows after NA removal: ", nrow(model_data), "\n"))

# Examine the structure of the final dataset
str(model_data)

# Ensure the target variable is a factor
model_data$Name <- as.factor(model_data$Name)
cat("Target variable levels:\n")
print(levels(model_data$Name))

# Step 8: Set up reproducible modeling environment
set.seed(123)

# Split the data (80% training, 20% testing)
train_index <- createDataPartition(model_data$Name, p = 0.8, list = FALSE)
train_data <- model_data[train_index, ]
test_data <- model_data[-train_index, ]

# Step 9: Set up cross-validation control
cv_control <- trainControl(
  method = "repeatedcv",  # Repeated cross-validation for more robust estimates
  number = 5,             # 5-fold cross-validation
  repeats = 3,            # Repeat 3 times
  classProbs = TRUE,      # Generate class probabilities
  summaryFunction = twoClassSummary,  # Use ROC summary metrics
  savePredictions = "final"
)

# Step 10: Define preprocessing
preproc <- c("center", "scale", "nzv")  # Add near-zero variance filter

# Step 11: Train multiple models
# KNN model
knn_model <- train(
  Name ~ ., 
  data = train_data, 
  method = "knn",
  trControl = cv_control,
  preProcess = preproc,
  tuneLength = 10,  # Try different k values
  metric = "ROC"    # Optimize for ROC
)

# MLP model
mlp_model <- train(
  Name ~ ., 
  data = train_data, 
  method = "mlp",
  trControl = cv_control,
  preProcess = preproc,
  tuneLength = 5,   # Try different network configurations
  metric = "ROC"
)

# Random Forest model
rf_model <- train(
  Name ~ ., 
  data = train_data, 
  method = "rf",
  trControl = cv_control,
  importance = TRUE,
  metric = "ROC",
  ntree = 500       # Increase number of trees
)

# Step 12: Compare models
model_list <- list(
  RF = rf_model,
  MLP = mlp_model,
  KNN = knn_model
)

model_results <- resamples(model_list)
summary(model_results)

# Step 13: Evaluate best model on test data
best_model <- model_list[[which.max(sapply(model_list, function(x) max(x$results$ROC)))]]
cat(paste0("Best model: ", names(model_list)[which.max(sapply(model_list, function(x) max(x$results$ROC)))], "\n"))

# Predictions on test data
test_preds <- predict(best_model, test_data)
test_probs <- predict(best_model, test_data, type = "prob")

# Confusion matrix
conf_matrix <- confusionMatrix(test_preds, test_data$Name)
print(conf_matrix)

# ROC curve for best model
roc_obj <- roc(test_data$Name, test_probs[, "Positive"])
roc_ci <- ci.auc(roc_obj)

# Step 14: Feature importance for best model (if RF)
if(names(model_list)[which.max(sapply(model_list, function(x) max(x$results$ROC)))] == "RF") {
  importance <- varImp(best_model)
  print(importance)
  
  # Plot variable importance
  plot(importance)
}

# Step 15: Create performance metrics dataframe for all models
metrics <- data.frame(
  Model = character(),
  ROC = numeric(),
  Sensitivity = numeric(),
  Specificity = numeric(),
  ROC_Lower = numeric(),
  ROC_Upper = numeric(),
  stringsAsFactors = FALSE
)

for(name in names(model_list)) {
  model <- model_list[[name]]
  preds <- predict(model, test_data)
  probs <- predict(model, test_data, type = "prob")[, "Positive"]
  roc_result <- roc(test_data$Name, probs)
  roc_ci_result <- ci.auc(roc_result)
  
  metrics <- rbind(metrics, data.frame(
    Model = name,
    ROC = as.numeric(auc(roc_result)),
    Sensitivity = sensitivity(preds, test_data$Name, positive = "Positive"),
    Specificity = specificity(preds, test_data$Name, negative = "Negative"),
    ROC_Lower = roc_ci_result[1],
    ROC_Upper = roc_ci_result[3],
    stringsAsFactors = FALSE
  ))
}

# Step 16: Visualize model performance
# Reshape for plotting
metrics_long <- pivot_longer(
  metrics, 
  cols = c(ROC, Sensitivity, Specificity), 
  names_to = "Metric", 
  values_to = "Value"
)

# Create performance comparison plot
perf_plot <- ggplot(metrics_long, aes(x = Model, y = Value, fill = Metric)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(
    title = "Model Performance Comparison",
    x = "Model",
    y = "Value",
    fill = "Metric"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold"),
    axis.text.x = element_text(angle = 0, hjust = 0.5),
    legend.position = "bottom"
  ) +
  ylim(0, 1) +
  scale_fill_brewer(palette = "Set1")

# Save plot
ggsave("model_performance_comparison.png", perf_plot, width = 10, height = 6, dpi = 300)

# Step 17: ROC curves comparison
roc_plot <- ggroc(list(
  RF = roc(test_data$Name, predict(rf_model, test_data, type = "prob")[, "Positive"]),
  MLP = roc(test_data$Name, predict(mlp_model, test_data, type = "prob")[, "Positive"]),
  KNN = roc(test_data$Name, predict(knn_model, test_data, type = "prob")[, "Positive"])
)) +
  labs(
    title = "ROC Curves Comparison",
    x = "False Positive Rate (1 - Specificity)",
    y = "True Positive Rate (Sensitivity)"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold"),
    legend.position = "bottom",
    legend.title = element_blank()
  ) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray") +
  scale_color_brewer(palette = "Set1")

# Save ROC plot
ggsave("roc_curves_comparison.png", roc_plot, width = 10, height = 6, dpi = 300)

# Step 18: Save models and results
saveRDS(model_list, "illegal_dumping_models.rds")
saveRDS(metrics, "model_metrics.rds")
write.csv(metrics, "model_metrics.csv", row.names = FALSE)

cat("Analysis complete. Models and results saved.\n")
