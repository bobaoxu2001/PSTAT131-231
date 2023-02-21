library(tidymodels)
library(discrim)
library(poissonreg)
library(corrr)
tidymodels_prefer()
suppressMessages(library(tidyquant))
library(ggplot2)
library(fitdistrplus)
library(reshape2)
library(patchwork)
library(parallel)
library(kableExtra)
#install.packages("kknn")
library(kknn)
set.seed(3435) # can be any number

iris <- read_csv(file = "iris.csv") %>%
  na.omit() %>%
  mutate(Species = case_when(Species == "Iris-setosa" ~ 0, 
                             Species == "Iris-versicolor" ~ 1,
                             Species == "Iris-virginica" ~ 2)) %>%
  clean_names()
iris$species <- as.factor(iris$species) 

iris_split <- iris %>% 
  initial_split(strata = species, prop = 0.8)
iris_train <- training(iris_split)
iris_test <- testing(iris_split)

iris_recipe <- 
  recipe(species ~ sepal_length_cm + sepal_width_cm + petal_length_cm + petal_width_cm, 
         data = iris) %>% 
  step_dummy(all_nominal_predictors()) %>% 
  step_center(all_predictors()) %>%  
  step_scale(all_predictors())

iris_folds <- vfold_cv(iris_train, strata = species, 
                       v = 5)

knn_spec <-nearest_neighbor() %>%
  set_mode("classification") %>%
  set_engine("kknn") %>% 
  set_args(neighbors = tune())

translate(knn_spec)

knn_workflow <- workflow() %>%
  add_recipe(iris_recipe) %>%
  add_model(knn_spec)

knn_params <- parameters(knn_spec)

knn_grid <- grid_regular(knn_params, levels = 5)

tune_res_knn <- tune_grid(
  knn_workflow,
  resamples = iris_folds,
  grid = knn_grid
)

autoplot(tune_res_knn)

# knn_roc <- collect_metrics(tune_res_knn) %>% 
  # dplyr::select(.metric, mean, std_err)

# knn_roc

bestknn <- select_best(tune_res_knn,metrix="roc_auc")
bestknn

knn_final <- finalize_workflow(knn_workflow, bestknn)

knn_final_fit <- fit(knn_final, data = iris_train)
