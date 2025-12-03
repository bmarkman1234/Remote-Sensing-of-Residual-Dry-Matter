# === Load Libraries ===
library(readxl)
library(dplyr)
library(randomForest)
library(ggplot2)
library(gridExtra)
library(scales)
library(caret)

# === Load Data ===
file_path <- "C:/Users/bmark/PycharmProjects/MS_Thesis/data/Dangermond Field Data.xlsx"
df <- read_excel(file_path)
df$study_area <- tolower(df$study_area)

# === Define Structure Groups ===
standing_sites <- c('cmt_ungrazed', 'jalama_horse')
mixed_sites <- c('jalachichi', 'steves_flat', 'jalama_bull', 'east_tinta')
laying_sites <- c('cojo_cow', 'jalama_mare')

df$structure <- case_when(
  df$study_area %in% standing_sites ~ "Standing",
  df$study_area %in% mixed_sites ~ "Mixed",
  df$study_area %in% laying_sites ~ "Laying",
  TRUE ~ NA_character_
)

df <- df %>% filter(structure %in% c("Standing", "Mixed", "Laying"))

# === LiDAR Predictors ===
lidar_predictors <- c("chm_max", "chm_range", "chm_mean", "chm_std", "chm_median", "chm_pct90")

# Drop rows with NA in predictors or target
df <- df[complete.cases(df[, c(lidar_predictors, "oven_weight")]), ]

# Setup color palette
palette <- c("Standing" = "red", "Mixed" = "darkgreen", "Laying" = "blue")

# Initialize containers
results <- list()
plot_list <- list()
importance_list <- list()

# === LOOCV Modeling and Plotting ===
for (structure in c("Standing", "Mixed", "Laying")) {
  df_struct <- df %>% filter(structure == !!structure)
  
  if (nrow(df_struct) < 2) {
    cat(paste0("Skipping ", structure, ": not enough data.\n"))
    next
  }
  
  X <- df_struct %>% select(all_of(lidar_predictors)) %>% scale()
  y <- df_struct$oven_weight
  n <- length(y)
  
  pred_lr <- numeric(n)
  pred_rf <- numeric(n)
  
  for (i in 1:n) {
    train_idx <- setdiff(1:n, i)
    X_train <- X[train_idx, , drop = FALSE]
    y_train <- y[train_idx]
    X_test <- X[i, , drop = FALSE]
    
    lr <- lm(y_train ~ ., data = as.data.frame(X_train))
    pred_lr[i] <- predict(lr, newdata = as.data.frame(X_test))
    
    rf <- randomForest(
      x = X_train, y = y_train,
      ntree = 1000,
      nodesize = 5,
      maxnodes = 30
    )
    pred_rf[i] <- predict(rf, newdata = X_test)
  }
  
  # Metrics
  r2_lr <- R2(pred_lr, y)
  mae_lr <- MAE(pred_lr, y)
  sd_lr <- sd(y - pred_lr)
  
  r2_rf <- R2(pred_rf, y)
  mae_rf <- MAE(pred_rf, y)
  
  # Full-model LR for p-value
  lr_full <- lm(y ~ ., data = as.data.frame(X))
  f_stat <- summary(lr_full)$fstatistic
  p_val_lr <- pf(f_stat[1], f_stat[2], f_stat[3], lower.tail = FALSE)
  p_val_str <- ifelse(p_val_lr < 0.05, "< 0.05", format(round(p_val_lr, 3), nsmall = 3))
  
  results[[structure]] <- data.frame(
    Structure = structure,
    `RF R2` = round(r2_rf, 2),
    `RF MAE` = round(mae_rf, 2),
    `LR R2` = round(r2_lr, 2),
    `LR MAE` = round(mae_lr, 2),
    `LR SD` = round(sd_lr, 2),
    `LR p-value` = p_val_str,
    Sites = paste(sort(unique(df_struct$study_area)), collapse = ", "),
    `Sample Size` = n
  )
  
  # === Prediction Plot ===
  df_plot <- data.frame(Actual = y, Pred_LR = pred_lr, Pred_RF = pred_rf)
  
  if (structure == "Laying") {
    legend_y_top <- min(y) - 0.15 * diff(range(y))
  } else {
    legend_y_top <- max(y)
  }
  
  p <- ggplot(df_plot, aes(x = Actual)) +
    geom_point(aes(y = Pred_RF), color = palette[structure], size = 4.5, shape = 16, alpha = 0.8) +
    geom_point(aes(y = Pred_LR), color = palette[structure], size = 4.5, shape = 1, alpha = 0.9) +
    geom_smooth(aes(y = Pred_LR), method = "lm", se = FALSE, color = "black", size = 1.5) +
    annotate("text",
             x = min(y),
             y = legend_y_top,
             hjust = 0, vjust = 1, size = 7,
             label = "○ Linear Regression\n● Random Forest",
             color = palette[structure]) +
    labs(
      title = paste(structure, "Vegetation"),
      x = "Actual RDM (g)",
      y = "Predicted RDM (g)"
    ) +
    theme_minimal(base_size = 24) +
    theme(
      plot.title = element_text(size = 26, face = "bold", hjust = 0.5),
      axis.title = element_text(size = 24, face = "bold"),
      axis.text = element_text(size = 22)
    )
  
  plot_list[[structure]] <- p
  
  # === Variable Importance Plot ===
  rf_full <- randomForest(
    x = X, y = y,
    ntree = 1000,
    nodesize = 5,
    maxnodes = 30,
    importance = TRUE
  )
  
  importance_df <- as.data.frame(importance(rf_full))
  importance_df$Feature <- rownames(importance_df)
  
  importance_df <- importance_df %>%
    arrange(desc(IncNodePurity)) %>%
    mutate(
      PercentImportance = 100 * IncNodePurity / sum(IncNodePurity),
      Feature = factor(Feature, levels = rev(Feature))
    )
  
  imp_plot <- ggplot(importance_df, aes(x = Feature, y = PercentImportance)) +
    geom_bar(stat = "identity", fill = palette[structure]) +
    coord_flip() +
    labs(title = paste(structure, "Vegetation"),
         x = "Feature",
         y = "Importance (% of Total)") +
    theme_minimal(base_size = 24) +
    theme(
      plot.title = element_text(size = 26, face = "bold", hjust = 0.5),
      axis.title.x = element_text(size = 20, face = "bold"),
      axis.title.y = element_text(size = 20, face = "bold"),
      axis.text.x = element_text(size = 20),
      axis.text.y = element_text(size = 22),
      plot.margin = margin(10, 20, 10, 20)
    )
  
  importance_list[[structure]] <- imp_plot
}

# === Print Model Performance Table ===
cat("\nModel Performance by Vegetation Structure (LiDAR Only, LOOCV Predictions Only):\n")
results_df <- do.call(rbind, results)
print(results_df)

# === Display Prediction Plots ===
grid.arrange(plot_list$Standing, plot_list$Mixed, plot_list$Laying, nrow = 1)

# === Display Importance Plots ===
grid.arrange(importance_list$Standing, importance_list$Mixed, importance_list$Laying, nrow = 1)

# === Laying Vegetation (Outlier Removed) ===
laying_df <- df %>% filter(structure == "Laying")
X_laying <- laying_df %>% select(all_of(lidar_predictors)) %>% scale()
y_laying <- laying_df$oven_weight
n_laying <- length(y_laying)

pred_lr_laying <- numeric(n_laying)
pred_rf_laying <- numeric(n_laying)

for (i in 1:n_laying) {
  train_idx <- setdiff(1:n_laying, i)
  X_train <- X_laying[train_idx, , drop = FALSE]
  y_train <- y_laying[train_idx]
  X_test <- X_laying[i, , drop = FALSE]
  
  lr <- lm(y_train ~ ., data = as.data.frame(X_train))
  pred_lr_laying[i] <- predict(lr, newdata = as.data.frame(X_test))
  
  rf <- randomForest(
    x = X_train, y = y_train,
    ntree = 1000,
    nodesize = 5,
    maxnodes = 30
  )
  pred_rf_laying[i] <- predict(rf, newdata = X_test)
}

df_laying_clean <- data.frame(
  Actual = y_laying,
  Pred_LR = pred_lr_laying,
  Pred_RF = pred_rf_laying
)

df_laying_filtered <- df_laying_clean %>% filter(Pred_LR > 0 & Pred_LR < 110)

r2_lr_clean <- R2(df_laying_filtered$Pred_LR, df_laying_filtered$Actual)
mae_lr_clean <- MAE(df_laying_filtered$Pred_LR, df_laying_filtered$Actual)

r2_rf_clean <- R2(df_laying_filtered$Pred_RF, df_laying_filtered$Actual)
mae_rf_clean <- MAE(df_laying_filtered$Pred_RF, df_laying_filtered$Actual)

ggplot(df_laying_filtered, aes(x = Actual)) +
  geom_point(aes(y = Pred_RF), color = palette["Laying"], size = 4.5, shape = 16, alpha = 0.8) +
  geom_point(aes(y = Pred_LR), color = palette["Laying"], size = 4.5, shape = 1, alpha = 0.9) +
  geom_smooth(aes(y = Pred_LR), method = "lm", se = FALSE, color = "black", size = 1.5) +
  labs(
    title = "Laying Vegetation (2 Outliers Removed)",
    x = "Actual RDM (g)",
    y = "Predicted RDM (g)"
  ) +
  annotate("text", x = min(df_laying_filtered$Actual), y = max(df_laying_filtered$Actual),
           hjust = 0, vjust = 1, size = 6,
           label = paste0(
             "Linear Regression:\n  R² = ", round(r2_lr_clean, 2),
             ", MAE = ", round(mae_lr_clean, 1),
             "\nRandom Forest:\n  R² = ", round(r2_rf_clean, 2),
             ", MAE = ", round(mae_rf_clean, 1)
           )) +
  theme_minimal(base_size = 24) +
  theme(
    plot.title = element_text(size = 26, face = "bold", hjust = 0.5),
    axis.title = element_text(size = 24, face = "bold"),
    axis.text = element_text(size = 22)
  )

