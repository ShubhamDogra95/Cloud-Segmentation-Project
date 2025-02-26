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
# library(GDAtools)
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
# ********************* 1. Loading Dataset *******************************************
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
par(mfrow = c(1, 1))

# 1.1 Open Image
# --------------------------------------------------------------------------
myponga_img = "2024-04-08-00_00_2024-04-08-23_59_Sentinel-2_L1C_True_color"
pt_augusta_img = "2024-04-03-00_00_2024-04-03-23_59_Sentinel-2_L2A_True_color"
aldinga_img = "2024-04-03-00_00_2024-04-03-23_59_Sentinel-2_L2A_True_color"
whyalla_img = "2024-04-08-00_00_2024-04-08-23_59_Sentinel-2_L2A_True_color"

# Select one of the above images in the code below.
path_name = "whyalla"  # Subfolder name where the image is stored.
fname <- paste0("./", path_name, "/", whyalla_img, ".tiff")

# Check existence image/filename in current work directory.
if (file.exists(fname)) {
  print("File Found, and Continue.")
} else {
  # Do this for first time only when a train_file.rds does not exist.
  print("Filename Not found!, Check Current Folder Path and Filename!")
}

image_stack <- stack(fname)
image_stack
names(image_stack)  # Change the column names if necessary
names(image_stack) <- c("red", "green", "blue")
names(image_stack)

# Visualization of the stacked image as RGB Composite
par(col.axis = "white", col.lab = "white", tck = 0)
plotRGB(image_stack,
        r = 1, g = 2, b = 3,
        stretch = "lin",
        axes = TRUE,
        main = paste0("SA Region - ", toupper(path_name), "\n RGB (Red, Green, Blue)"))
box(col = "white")

# Save the image
writeRaster(image_stack, paste0("img_", path_name, ".tiff"), format = "GTiff", overwrite = TRUE)

# 1.2 Open Train File
# --------------------------------------------------------------------------
# Load the model from the file
train_file <- readRDS("train_file.rds")
dim(train_file)

# Quick summary of the RDS file
train_file %>% group_by(image) %>% dplyr::summarise(counts = n())
train_file %>% group_by(image, class) %>% dplyr::summarise(counts = n())

str(train_file)
train_file$id <- as.factor(train_file$id)
train_file$class <- as.factor(train_file$class)
str(train_file)

# 1.3 Open Current Image File to be Tested
# --------------------------------------------------------------------------
# Load the model from the current image file
df <- subset(train_file, image == path_name)
dim(df)
head(df)
str(df)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ******************* 3. Preparation Train Sets for ML Algorithms ********************
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Dataset to be split into Train & Test Sets
library(rpart)
library(rpart.plot)
library(scales)
library(stats)
library(C50)
library(caret)
library(MASS)
library(caTools)
library(e1071)

# 3.1 Compile Train File from Multiple Images
# --------------------------------------------------------------------------
# (Assuming the train_file has been compiled as shown in previous steps)

# 3.2 Derive Train Sets
# --------------------------------------------------------------------------
# Extract equal number of samples for each class
set.seed(1234)
train <- train_file %>% group_by(image, id) %>% sample_n(size = 4000)
head(train)
table(train$id)
train %>% group_by(image) %>% dplyr::summarise(counts = n())
train %>% group_by(image, id) %>% dplyr::summarise(counts = n())
train <- train[,c(1:3,6)]
dim(train)


# 3.3 Derive Test Sets
# --------------------------------------------------------------------------
# Test set derived from current image dataset
set.seed(1234)
test <- df[, c(1:3, 6:7)]
str(test)
head(test)
table(test$id)
test %>% group_by(id) %>% dplyr::summarise(counts = n())
test <- test[, c(1:4)]
dim(test)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ******************* 4. Decision Tree Classification for Satellite Image ************
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# 4.1 Cross Validation Train and Test Sets
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

# 4.2 Model Prediction
# --------------------------------------------------------------------------
model.prediction <- predict(dt_cv, test)
# Print confusion matrix and results
confusionMatrix(data = model.prediction, reference = test$id)

plot(varImp(dt_cv), main = "Decision Tree - Variable Importance")

# 4.3 Predict Entire Image Based on the Model
# --------------------------------------------------------------------------
# Save model file
saveRDS(dt_cv, paste0("dt_model_", path_name, ".rds"))
# Load the model from the file
dt_model <- readRDS(paste0("dt_model_", path_name, ".rds"))

# Predict all pixels/run classification
result <- predict(image_stack,
                  dt_model,
                  filename = "img_stack",
                  overwrite = TRUE)

par(mfrow = c(1, 2))
# Define colors for class representation
mycolors <- c("#FFFF00", "#393939")

# Plot Classification
plot(result,
     axes = FALSE,
     box = FALSE,
     main = paste0("Predicted Cloud - ", toupper(path_name), "\n[Yellow - Cloud];\n[Black - Non-Cloud]."),
     col = mycolors)

# Plotting the main satellite images
par(col.axis = "white", col.lab = "white", tck = 0)
plotRGB(image_stack,
        r = 1, g = 2, b = 3,
        stretch = "lin",
        axes = TRUE,
        main = paste0("Region - ", toupper(path_name), "\nRGB (Red, Green, Blue)\n"))
box(col = "white")

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ******************* 5. Random Forests Classification for Satellite Image ************
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# 5.1 Cross Validation Train and Test Sets
# --------------------------------------------------------------------------
library(randomForest)

set.seed(1234)
rf.control <- trainControl(method = "repeatedcv",
                           number = 10,
                           repeats = 3)

rf_cv <- train(id ~ .,
               data = train,
               method = "rf",
               trControl = rf.control)
rf_cv$results
rf_cv$bestTune

# 5.2 Model Prediction
# --------------------------------------------------------------------------
model.prediction <- predict(rf_cv, test)
# Print confusion matrix and results
confusionMatrix(data = model.prediction, reference = test$id)

plot(varImp(rf_cv), main = "Random Forests - Variable Importance")


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ****************** 6. KNN Classification for Satellite Image ************************
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# 6.1 Cross Validation Train and Test Sets
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

# 6.2 Model Prediction
# --------------------------------------------------------------------------
model.prediction <- predict(knn_cv, test)
# Print confusion matrix and results
confusionMatrix(data = model.prediction, reference = test$id)
plot(varImp(knn_cv), main = "KNN - Variable Importance")
knn_cv$bestTune

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ************ 7. SVM Radial Classification for Satellite Image ***********************
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# 7.1 Cross Validation Train and Test Sets
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

# 7.2 Model Prediction
# --------------------------------------------------------------------------
model.prediction <- predict(svm_rad_cv, test)
# Print confusion matrix and results
confusionMatrix(data = model.prediction, reference = test$id)
plot(varImp(svm_rad_cv), main = "SVM Radial - Variable Importance")
svm_rad_cv$bestTune


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ******* 8. Gradient Boosting Classification for Satellite Image *******************
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

library(gbm)
library(xgboost)


# 8.1 Cross Validation Train and Test Sets
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
gbm_cv


# 8.2 Model Prediction
# --------------------------------------------------------------------------
# Model prediction
model.prediction <- predict(gbm_cv, test)
# Print confusion matrix and results
confusionMatrix(data = model.prediction, reference = test$id)


plot(varImp(gbm_cv), main = "GBM - Variable Importance")
gbm_cv$bestTune


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ************************  9. Exploratory Data Analysis. ****************************
# ************************  Spectral profiles - Train File. ****************************
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


# 9.1 Selection Bands columns plus class variable.
# --------------------------------------------------------------------------
head(df)
dim(df)

df2 <- df[, c(1:3,5,7)] # Exclude polygon_no variable.
head(df2)

# 9.2 Calculating mean values by Class for each of four Bands.
# --------------------------------------------------------------------------

profiles = ddply(df2[,-5], .(class), colwise(mean))
profiles

# 9.3 Plotting the spectral profile in Line Plot.
# --------------------------------------------------------------------------

#convert data from wide to long format.
profiles_2 <- profiles %>% pivot_longer(cols=c('red', 'green', 'blue'),
                                       names_to='Bands',
                                       values_to='Reflectance')

# Plot of Mean Spectral Values.
# Plot of Mean Spectral Values.
regions = "Whyalla"
ggplot_1 <- ggplot(profiles_2) + 
  theme_light() +
  geom_line(aes(x = Bands, y = Reflectance, group = class, colour = class),size=1.3) +
  labs(x="Spectral Bands",
       y="Reflectance Values",
       title=paste0("SA Regions - ",toupper(regions),"\nSpectral Profile Satellite Imagery - RGB(Red, Green, Blue,)\nCloud v's Non-Cloud Pixels."),
       subtitle="(Mean Reflectance Values by Spectral Bands).",
       caption="(band 2=blue band);   (band 3=green band);   (band 4=red band).") +
  theme(plot.title=element_text(size=14, hjust=0.5, face="bold", colour="blue", vjust=-1)) +
  scale_y_continuous(breaks = seq(0.1, 1.0, by=0.05),labels = scales::comma) +
  scale_color_discrete(name = "Classification\nPixels") # Manual legend title
ggplot_1


# 9.4 Plotting the Histogram & Density Plots.
# --------------------------------------------------------------------------

# Pivot of df 'df_final2' to longer format, ie columns into rows.
profiles_3 <- df2 %>% pivot_longer(cols=c('red', 'green', 'blue'),
                                  names_to='Bands',
                                  values_to='Reflectance')
profiles_3



# Calculate mean values by class, cloud v non-cloud.
mu <- ddply(profiles_3, "class", summarise, grp.mean=mean(Reflectance))

mean_cloud <- round(mu[1,2],4)
mean_non_cloud <- round(mu[2,2],4)

# Histogram Spectral profiles with Mean values.
ggplot_2 <- ggplot(profiles_3, aes(x=Reflectance, group=class, fill=class)) + 
  geom_density(alpha = 0.40) + theme_classic() +
  geom_vline(data=mu, aes(xintercept=grp.mean),color=c("red","blue"),linetype="dotdash", size=1) +
  labs(x="Reflectance Values", 
       y="Density of Reflectance Values",
       title=paste0("SA Regions - ",toupper(regions)," \nDensity Histogram of Spectral Profiles - Reflectance\nCloud v's Non-Cloud Classification"),
       subtitle = "Vertical Lines = Mean Values Reflectance (Cloud v's Non-Cloud Class)",
       caption="*** Spectral Profile Satellite Image - RGB(Red, Green, Blue) ***") +
  scale_x_continuous(breaks = seq(0, 1, 0.10),labels = scales::comma) +
  scale_y_continuous(breaks = seq(0, 20, 1),labels = scales::comma) +
  annotate(x=mean_cloud,y=Inf,label=paste0("Cloud = ",mean_cloud),hjust=0.5,vjust=1.3,geom="label",size=3.5,color="red") +
  annotate(x=mean_non_cloud,y=Inf,label=paste0("Non-Cloud = ",mean_non_cloud),hjust=0.5,vjust=1.3,geom="label",size=3.5,color="blue") +
  theme(plot.title=element_text(size=14, hjust=0.5, face="bold", colour="blue", vjust=-1))  
ggplot_2


# 9.5 Plotting BoxPlots by Spectral Bands.
# --------------------------------------------------------------------------
cols <- c("#9FD6FF", "#9BFFC8", "#FFA18B")
ggplot_3 <- ggplot(data=profiles_3, aes(x=Bands, y=Reflectance,fill = Bands))  +
  geom_boxplot(alpha = 0.5,outlier.colour="red", outlier.shape=8,
               outlier.size=2)+ theme_classic() +
  coord_flip(ylim = c(0,1)) +
  theme(legend.position="none",plot.title = element_text(hjust=0.5)) +
  scale_fill_manual(values=cols) +
  labs(title = paste0("Regions SA - ",toupper(regions),"\nBoxplots of Reflectance Values for Pixels\nby Spectral Bands."),
       subtitle = "",
       caption="(band 2=blue band);   (band 3=green band);   (band 4=red band).",
       x = "Spectral Bands",
       y = "Reflectance Values") +
  theme(axis.text.x=element_text(angle=90, hjust=1, vjust=1,size=10)) + # x axis labels.
  scale_y_continuous(breaks=seq(0, 1, 0.1),labels = scales::comma) +
  theme(plot.title=element_text(size=14, hjust=0.5, face="bold", colour="blue", vjust=-1)) +
  theme(panel.spacing = unit(1.5, "lines"))
ggplot_3



# 9.6 Plotting BoxPlots by Class.
# --------------------------------------------------------------------------

ggplot_4 <- ggplot(data=profiles_3, aes(x=class, y=Reflectance, fill=class))  +
  geom_boxplot(alpha = 0.20,outlier.colour="red", outlier.shape=8,
               outlier.size=1)+ theme_classic() +
  coord_flip(ylim = c(0,1)) +
  theme(legend.position="none",plot.title = element_text(hjust=0.5)) +
  scale_fill_brewer(palette="Set1") +
  labs(title = paste0("SA Regions - ",toupper(regions),"\nBoxplots of Reflectance Values\nby Class Pixels."),
       subtitle = "",
       caption = "",
       x = "Pixels Class",
       y = "Reflectance Values") +
  theme(axis.text.x=element_text(angle=90, hjust=1, vjust=1,size=10)) + # x axis labels.
  scale_y_continuous(breaks=seq(0, 1, 0.1),labels = scales::comma) +
  theme(plot.title=element_text(size=14, hjust=0.5, face="bold", colour="blue", vjust=-1)) +
  theme(panel.spacing = unit(1.5, "lines"))
ggplot_4

