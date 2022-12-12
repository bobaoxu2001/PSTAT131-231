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

forest <- rand_forest() %>%
  set_engine("ranger", importance="impurity") %>%
  set_mode("classification") %>% 
  set_args(mtry = tune(),trees = tune(), min_n = tune())
forest

forest_workflow <- workflow() %>%
  add_model(forest %>% set_args(mtry = tune(), trees = tune(),
                                min_n = tune())) %>%
  add_recipe(iris_recipe)
#forest_workflow
param_grid2<-grid_regular(mtry(range = c(1, 4)),
                          trees(range = c(1,4)),
                          min_n(range = c(1,4)), 
                          levels = 4)
param_grid2

#install.packages("ranger")
tune_res_forest<-tune_grid(
  forest_workflow,
  resamples = iris_folds,
  grid = param_grid2,
  metrics = metric_set(roc_auc)
)

best_random<-select_best(tune_res_forest,metrix="roc_auc")
best_random

random_final<-finalize_workflow(forest_workflow, best_random)

random_final_fit <- fit(random_final, data = iris_train)
