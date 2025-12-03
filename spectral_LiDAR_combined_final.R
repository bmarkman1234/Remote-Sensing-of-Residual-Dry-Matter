# Script Workflow:
#   - Loads field data and prepares predictors from spectral indices and LiDAR-derived metrics
#   - Defines three modeling scenarios:
#       • Spectral only predictors
#       • LiDAR only predictors
#       • Combined spectral+LiDAR predictors
#   - For each scenario:
#       • Selects relevant samples and predictor variables
#       • Scales predictors and splits by study area (selected sites for spectral/combined)
#       • Performs leave-one-out cross-validation (LOOCV) with:
#            - Linear regression (lm)
#            - Random forest (randomForest)
#       • Calculates and plots observed vs predicted values for both models
#       • Trains full random forest on all data for variable importance plotting
#       • Fits full linear model and saves tidy output (regression coefficients, stats)
#       • Computes and records performance metrics:
#           - R2
#           - MAE (Mean Absolute Error)
#           - SD of errors
#           - p-values (linear regression significance)
#       • Calculates Moran’s I on residuals per site to assess spatial autocorrelation
#   - Combines results into:
#       • Performance summary table (R2, MAE, SD, p-value, predictors, sample size)
#       • Table of linear regression coefficients and statistics (exported as CSV)
#       • Moran’s I boxplot comparison for model types/scenarios
#       • Feature importance barplots for RF models
#       • Observed vs predicted scatterplots for each model/scenario
#   - Final output includes:
#       • CSV table of linear regression coefficients
#       • Multiple summary and comparison plots (performance, importance, residual autocorrelation)
#       • Printed tables of metrics and coefficients for review
################## AUTHOR: Bruce Markman ###########################

library(readxl)
library(dplyr)
library(randomForest)
library(ggplot2)
library(gridExtra)
library(grid)
library(spdep)
library(caret)
library(broom)
library(readr)

df <- read_excel("C:/Users/bmark/pycharmProjects/MS_Thesis/data/Dangermond Field Data.xlsx") %>% 
  mutate(study_area = tolower(study_area))

spectral <- c("cai", "lcai", "ndli")
lidar <- c("chm_max", "chm_range", "chm_mean", "chm_std", "chm_median", "chm_pct90")
combined <- c(spectral, lidar)
target <- "oven_weight"
selected_areas <- c("steves_flat", "jalama_bull", "east_tinta", "cojo_cow")

model_specs <- list(
  list(name = "Spectral Only", predictors = spectral, use_selected = TRUE, label = "A)"),
  list(name = "LiDAR Only", predictors = lidar, use_selected = FALSE, label = "B)"),
  list(name = "Spectral + LiDAR", predictors = combined, use_selected = TRUE, label = "C)")
)

calc_loocv_preds <- function(X, y) {
  n <- nrow(X)
  preds <- data.frame(lr = numeric(n), rf = numeric(n))
  for (i in 1:n) {
    train_idx <- setdiff(1:n, i)
    lr <- lm(y[train_idx] ~ ., data = X[train_idx, ])
    rf <- randomForest(x = X[train_idx, ], y = y[train_idx], 
                       ntree = 1000, importance = TRUE, nodesize = 5, maxnodes = 30)
    preds$lr[i] <- predict(lr, X[i, ])
    preds$rf[i] <- predict(rf, X[i, ])
  }
  preds
}

calc_morans_i <- function(y, preds, areas, label) {
  results <- list()
  for (site in unique(areas)) {
    idx <- which(areas == site)
    if (length(idx) == 9) {
      w <- cell2nb(3, 3, type = "queen")
      lw <- nb2listw(w, style = "W", zero.policy = TRUE)
      for (m in names(preds)) {
        resids <- y[idx] - preds[idx, m]
        I <- moran.test(resids, lw, zero.policy = TRUE)$estimate[1]
        results[[length(results)+1]] <- data.frame(Model = m, Moran_I = I, Site = site, Type = label)
      }
    }
  }
  bind_rows(results)
}

make_lr_rf_scatter <- function(y, preds, model_name, show_legend = FALSE) {
  df <- data.frame(
    Actual = rep(y, 2),
    Predicted = c(preds$lr, preds$rf),
    Model = factor(rep(c("Linear Regression", "Random Forest"), each = length(y)))
  )
  
  ggplot(df, aes(x = Actual, y = Predicted)) +
    geom_point(aes(shape = Model), size = 4, stroke = 1.2, color = "black", fill = "black") +
    geom_smooth(method = "lm", se = FALSE, color = "black", size = 1.2) +
    scale_shape_manual(values = c("Linear Regression" = 1, "Random Forest" = 16)) +
    labs(
      title = model_name,
      x = "Actual RDM (g)",
      y = "Predicted RDM (g)",
      shape = NULL
    ) +
    theme_minimal(base_size = 22) +
    theme(
      plot.title = element_text(size = 24, face = "bold", hjust = 0.5),
      axis.title = element_text(size = 22, face = "bold"),
      axis.text = element_text(size = 20),
      legend.position = ifelse(show_legend, "bottom", "none"),
      legend.text = element_text(size = 18)
    )
}

make_rf_importance_plot <- function(rf_model, model_name, label) {
  imp <- importance(rf_model, type = 1)
  imp_df <- data.frame(
    Feature = rownames(imp),
    Importance = 100 * imp[, 1] / sum(imp[, 1]),
    Model = model_name,
    Label = label
  )
  imp_df$Feature <- factor(imp_df$Feature, levels = imp_df$Feature[order(imp_df$Importance)])
  
  ggplot(imp_df, aes(x = Importance, y = Feature)) +
    geom_bar(stat = "identity", fill = "black") +
    labs(
      title = paste(label, model_name),
      x = "Importance (% of Total)",
      y = "Feature"
    ) +
    theme_minimal(base_size = 18) +
    theme(
      plot.title = element_text(size = 20, face = "bold"),
      axis.title = element_text(size = 18, face = "bold"),
      axis.text = element_text(size = 16)
    )
}

all_morans <- list()
results <- data.frame()
pred_plots <- list()
imp_plots <- list()
lr_rf_scatter_plots <- list()
lr_tables <- list()

for (i in seq_along(model_specs)) {
  spec <- model_specs[[i]]
  data <- df %>%
    {if (spec$use_selected) filter(., study_area %in% selected_areas) else .} %>%
    select(all_of(c("study_area", spec$predictors, target))) %>% na.omit()
  
  X <- scale(data[, spec$predictors]) %>% as.data.frame()
  y <- data[[target]]
  areas <- data$study_area
  
  preds <- calc_loocv_preds(X, y)
  morans <- calc_morans_i(y, preds, areas, spec$name)
  all_morans[[spec$name]] <- morans
  
  rf_full <- randomForest(x = X, y = y, ntree = 1000, importance = TRUE, nodesize = 5, maxnodes = 30)
  lr_rf_scatter_plots[[spec$name]] <- make_lr_rf_scatter(y, preds, spec$name, show_legend = (i == 2))
  imp_plots[[spec$name]] <- make_rf_importance_plot(rf_full, spec$name, spec$label)
  
  lm_fit <- lm(y ~ ., data = X)
  lr_table <- tidy(lm_fit) %>% mutate(Model = spec$name)
  lr_tables[[spec$name]] <- lr_table
  
  r2_lr <- R2(preds$lr, y)
  r2_rf <- R2(preds$rf, y)
  mae_lr <- MAE(preds$lr, y)
  mae_rf <- MAE(preds$rf, y)
  sd_lr <- sd(y - preds$lr)
  sd_rf <- sd(y - preds$rf)
  p_val_lr <- pf(summary(lm(y ~ ., data = X))$fstatistic[1],
                 summary(lm(y ~ ., data = X))$fstatistic[2],
                 summary(lm(y ~ ., data = X))$fstatistic[3],
                 lower.tail = FALSE)
  p_str <- ifelse(p_val_lr < 0.05, "< 0.05", format(round(p_val_lr, 3), nsmall = 3))
  
  results <- bind_rows(results,
                       data.frame(Model = paste(spec$name, "(LR)"), R2 = round(r2_lr, 2), MAE = round(mae_lr, 2), SD_Error = round(sd_lr, 2), p_value = p_str, Predictors = length(spec$predictors), Sample_Size = nrow(X)),
                       data.frame(Model = paste(spec$name, "(RF)"), R2 = round(r2_rf, 2), MAE = round(mae_rf, 2), SD_Error = round(sd_rf, 2), p_value = "N/A", Predictors = length(spec$predictors), Sample_Size = nrow(X))
  )
}

lr_combined <- bind_rows(lr_tables) %>%
  rename(
    β = estimate,
    `STD.Error` = std.error,
    `t` = statistic,
    `p-value` = p.value
  ) %>%
  mutate(
    β = round(β, 2),
    `STD.Error` = round(`STD.Error`, 2),
    `t` = round(`t`, 2),
    `p-value` = formatC(`p-value`, digits = 2, format = "f")
  ) %>%
  select(term, Model, β, `STD.Error`, `t`, `p-value`)

write_csv(lr_combined, "C:/Users/bmark/OneDrive/Desktop/linear_regression_coefficients_combined.csv")

cat("\nLinear Regression Coefficients Tables:\n")
print(lr_combined)

moran_df <- bind_rows(all_morans)
moran_df$Model <- factor(moran_df$Model, levels = c("rf", "lr"), labels = c("Random Forest", "Linear Regression"))
moran_df$Type <- factor(moran_df$Type, levels = c("Spectral Only", "LiDAR Only", "Spectral + LiDAR"))

p_moran <- ggplot(moran_df, aes(x = Type, y = Moran_I, fill = Model)) +
  geom_boxplot(aes(color = Model), outlier.shape = NA, width = 0.6, position = position_dodge(0.75), linewidth = 0.8) +
  scale_fill_manual(values = c("Random Forest" = "gray35", "Linear Regression" = "gray85")) +
  scale_color_manual(values = c("Random Forest" = "black", "Linear Regression" = "black")) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
  labs(y = "Moran's I", x = NULL, fill = NULL, color = NULL) +
  theme_minimal(base_size = 22) +
  theme(
    legend.position = "right",
    axis.title.y = element_text(size = 22, face = "bold"),
    axis.text = element_text(size = 20),
    plot.title = element_text(size = 24, face = "bold", hjust = 0.5)
  )

grid.arrange(
  lr_rf_scatter_plots[["Spectral Only"]],
  lr_rf_scatter_plots[["LiDAR Only"]],
  lr_rf_scatter_plots[["Spectral + LiDAR"]],
  nrow = 1
)

print(p_moran)

grid.arrange(
  imp_plots[["Spectral Only"]],
  imp_plots[["LiDAR Only"]],
  imp_plots[["Spectral + LiDAR"]],
  nrow = 1
)

cat("\nModel Performance Summary (LOOCV):\n")
print(results %>% select(Model, R2, MAE, SD_Error, p_value, Predictors, Sample_Size))




