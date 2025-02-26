# Load spatial packages
library(raster)
library(tidyverse)
library(sf)
library(rpart)
library(rpart.plot)
library(rasterVis)
library(mapview)
library(mapedit)
library(caret)
library(forcats)
library(terra)
library(rhdf5)
library(ggplot2)
library(sp)
library(dplyr)
library(rgdal)
library(gdalUtils)
library(htmltools)
library(tidyr)
library(plyr)
library(sass)
library(RColorBrewer)
library(grid)
library(prismatic) # Approximation to hexcode colors in console
library(patchwork)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ********************* 1. Loading Datasets ******************************************
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
par(mfrow = c(1, 1))
# Select one of these Untrained Images.
mannum_img = "2024-04-15-00_00_2024-04-15-23_59_Sentinel-2_L1C_True_color"
waikerie_img = "2023-05-28-00_00_2023-05-28-23_59_Sentinel-2_L1C_True_color"
flinders_img = "2024-03-19-00_00_2024-03-19-23_59_Sentinel-2_L2A_True_color"
kingston_img = "2024-04-25-00_00_2024-04-25-23_59_Sentinel-2_L2A_True_color"

# 1.1 Open Image
# --------------------------------------------------------------------------
# Select one of the above images in the code below.
path_name = "flinders"  # Subfolder name where the image is stored.
fname <- paste0("./", path_name, "/", flinders_img, ".tiff")

# Check existence image/filename in current work directory.
if (file.exists(fname)) {
  print("File Found, and Continue.")
} else {
  # Do this for first time only when a train_file.rds does not exist.
  print("Filename Not found!, Check Current Folder Path and Filename!")
}

# Make sure your R code work folder is set to the folder name, e.g., kingston.
image_stack <- stack(fname)
names(image_stack)  # Change the column names if necessary
names(image_stack) <- c("red", "green", "blue")
names(image_stack)

# Visualization of the stacked image as RGB Composite
par(col.axis = "white", col.lab = "white", tck = 0)
plotRGB(image_stack,
        r = 1, g = 2, b = 3,
        stretch = "lin",
        axes = TRUE,
        main = paste0(toupper(path_name), " - Region SA\n RGB (Red, Green, Blue)"))
box(col = "white")

# Save the image.
writeRaster(image_stack, paste0("img_",path_name,".tiff"), format = "GTiff",overwrite = TRUE)

# Open the entire current image file as a dataframe
image_stack_df <- as.data.frame(image_stack, xy = TRUE)
dim(image_stack_df)
head(image_stack_df)

# Check for missing values in band colours
sum(is.na(image_stack_df))

# 1.2 Open Train File
# --------------------------------------------------------------------------
# Load the model from the file and ensure it's in your current folder path.
train_file <- readRDS("train_file.rds")
dim(train_file)
# Quick summary of counts in the RDS file
train_file %>% group_by(image) %>% dplyr::summarise(counts = n())
train_file %>% group_by(image, class) %>% dplyr::summarise(counts = n())

# Convert class and id to factor variables
train_file$id <- as.factor(train_file$id)
train_file$class <- as.factor(train_file$class)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# *********** 2. Extract Threshold from Train File ***********************************
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# 2.1 Summary Stats of Cloud vs Non-Cloud Pixels in Train File
# --------------------------------------------------------------------------
summary(train_file %>% filter(class == "clouds"))
summary(train_file %>% filter(class != "clouds"))

# 2.2 Check Thresholds for Cloud vs Non-Cloud Pixels
# --------------------------------------------------------------------------
# Set threshold value
threshold = 0.60

# Cross-tab of trained images by class
xtabs(~image + class, data = train_file)

# Counts of reflectance values above threshold value
train_file %>% group_by(class) %>%
  dplyr::summarise(totals = n(), red = sum(red >= threshold),
                   green = sum(green >= threshold),
                   blue = sum(blue >= threshold))

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ******************* 3. Preparation of Untrained Dataset ****************************
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# 3.1 Label the Untrained Dataset Image - CLOUD vs NON-CLOUD
# --------------------------------------------------------------------------
image_stack_df2 <- image_stack_df %>%
  mutate(threshold_id = case_when(red >= threshold & green >= threshold & blue >= threshold ~ 1,
                                  TRUE ~ 2))
image_stack_df2$threshold_id <- as.factor(image_stack_df2$threshold_id)

# Check distribution by Class ID
table(image_stack_df2$threshold_id)
head(image_stack_df2)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ******************* 4. Set Train and Test Sets for ML ******************************
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# 4.1 Derive Train Sets
# --------------------------------------------------------------------------
train_file2 <- train_file[, c(1:3, 6:7)]
head(train_file2)

# Extract equal number of samples for each class
set.seed(1234)
train <- train_file2 %>% group_by(image, id) %>% sample_n(size = 4000)
train <- train[, 1:4]
head(train)
dim(train)

# 4.2 Derive Test Sets
# --------------------------------------------------------------------------
head(image_stack_df2)

test <- image_stack_df2[, c(3:6)]
test %>% group_by(test$threshold_id) %>%  dplyr::summarise(counts = n())
dim(test)
head(test)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ******************* 5. Decision Tree Classification for Satellite Image ************
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
library(rpart)
library(rpart.plot)
library(scales)
library(stats)
library(C50)
library(caret)
library(MASS)
library(caTools)
library(e1071)

# 5.1 Cross Validation Train and Test Sets
# --------------------------------------------------------------------------
dt.control <- trainControl(method = "repeatedcv",
                           number = 10,
                           repeats = 3)
set.seed(1234)
dt_cv <- train(id ~ .,
               data = train,
               method = "rpart2",
               trControl = dt.control,
               tuneLength = 15)
dt_cv$results
dt_cv$bestTune

# 5.2 Model Prediction
# --------------------------------------------------------------------------
model.prediction <- predict(dt_cv, test)
# Print confusion matrix and results
confusionMatrix(data = model.prediction, reference = test$threshold_id)

plot(varImp(dt_cv), main = "Decision Tree - Variable Importance")

# Combine the actual and predicted results
results <- test %>% mutate(Predicted_id = model.prediction)


# 5.3 Predict Entire Image Based on the Model
# Plots actual and predicted results.
# --------------------------------------------------------------------------
#Save model-file
place_name="Flinders"
saveRDS(dt_cv, paste0("dt_model_",place_name,".rds"))
# Load the model from the file
dt_model <- readRDS(paste0("dt_model_",place_name,".rds"))


#Predict all pixels classification
result <- predict(image_stack,
                  dt_model,
                  filename = "img_stack",
                  overwrite = TRUE
)

par(mfrow = c(1,2))
#Define colors for class representation - one color per class necessary!
# mycolors <- c("white","#393939")
mycolors <- c("yellow","black")
#
# Plot Classification
plot(result,
     axes = FALSE,
     box = FALSE,
     main=paste0("Predicted Cloud -  ",toupper(place_name),"\n[Yellow - Cloud];\n[Black - Non-Cloud]."),
     col = mycolors
)
writeRaster(result, paste0("predicted_img_",place_name,".tiff"), format = "GTiff",overwrite = TRUE)

# Plotting the main satellite images.
par(col.axis = "white", col.lab = "white", tck = 0)
plotRGB(image_stack,
        r = 1, g = 2, b = 3,
        stretch = "lin",
        axes = TRUE,
        # scale=255,
        main = paste0("Actual Cloud - ",toupper(place_name),"\nRGB (Red, Green, Blue)\n"))
box(col = "white")
par(mfrow = c(1, 1))



# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ******************* 6. Random Forests Classification for Satellite Image ************
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
library(randomForest)

# 6.1 Cross Validation Train and Test Sets
# --------------------------------------------------------------------------
rf.control <- trainControl(method = "repeatedcv",
                           number = 10,
                           repeats = 3)
set.seed(1234)
rf_cv <- train(id ~ .,
               data = train,
               method = "rf",
               trControl = rf.control)
rf_cv$results
rf_cv$bestTune

# 6.2 Model Prediction
# --------------------------------------------------------------------------
model.prediction <- predict(rf_cv, test)
# Print confusion matrix and results
confusionMatrix(data = model.prediction, reference = test$threshold_id)

varImp(rf_cv)  # Most important variables
plot(varImp(rf_cv), main = "Random Forests - Variable Importance")



# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ******************* 7. KNN Classification for Satellite Image ***********************
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# 7.1 Cross Validation Train and Test Sets
# --------------------------------------------------------------------------
knn_control = trainControl(method = "repeatedcv",
                           number = 10,
                           repeats = 3)
set.seed(1234)
knn_cv = train(id ~ .,
               data = train,
               method = "knn",
               trControl = knn_control,
               metric = "Kappa")

# 7.2 Model Prediction
# --------------------------------------------------------------------------
model.prediction <- predict(knn_cv, test)
# Print confusion matrix and results
confusionMatrix(data = model.prediction, reference = test$threshold_id)
plot(varImp(knn_cv), main = "KNN - Variable Importance")

knn_cv$results
knn_cv$bestTune


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ************ 8. SVM Radial Classification for Satellite Image ***********************
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# 8.1 Cross Validation Train and Test Sets
# --------------------------------------------------------------------------
smvRadial_control = trainControl(method = "repeatedcv",
                                 number = 10,
                                 repeats = 3)
set.seed(1234)
svm_rad_cv = train(id ~ .,
                   data = train,
                   method = "svmRadial",
                   trControl = smvRadial_control,
                   tuneLength = 10,
                   metric = "Kappa")
svm_rad_cv$results
svm_rad_cv$bestTune
plot(varImp(svm_rad_cv), main = "SVM Radial - Variable Importance")

# 8.2 Model Prediction
# --------------------------------------------------------------------------
model.prediction <- predict(svm_rad_cv, test)
# Print confusion matrix and results
confusionMatrix(data = model.prediction, reference = test$threshold_id)



# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ******* 9. Gradient Boosting Classification for Satellite Image *********************
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
library(gbm)
library(xgboost)

# 9.1 Cross Validation Train and Test Sets
# --------------------------------------------------------------------------
set.seed(1234)
gbm_cv <- caret::train(id ~ .,
                       data = train,
                       method = "gbm",
                       trControl = trainControl(method = "repeatedcv",
                                                number = 10,
                                                repeats = 3,
                                                verboseIter = FALSE),
                       verbose = 0)

plot(varImp(gbm_cv), main = "GBM - Variable Importance")

# 9.2 Model Prediction
# --------------------------------------------------------------------------
model.prediction <- predict(gbm_cv, test)
# Print confusion matrix and results
confusionMatrix(data = model.prediction, reference = test$threshold_id)
gbm_cv$results
gbm_cv$bestTune



