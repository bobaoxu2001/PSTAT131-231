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

boost_spec <- boost_tree(trees = c(1,10), tree_depth = 4) %>%
  set_engine("xgboost") %>%
  set_mode("classification")
param_grid_boost <- grid_regular(trees(range = c(1, 10)),  levels = 5)

boost_workflow <- workflow() %>%
  add_model(boost_spec %>% set_args(trees = tune())) %>%
  add_recipe(iris_recipe)

tune_res_boost <- tune_grid(
  boost_workflow,
  resamples = iris_folds,
  grid = param_grid_boost,
  metrics = metric_set(roc_auc)
)

boost_preformance <- collect_metrics(tune_res_boost)

final_boost <- select_best(tune_res_boost, metric = "roc_auc")

final_workflow_boost <- finalize_workflow(boost_workflow, final_boost)

final_fit_boost <- fit(final_workflow_boost, data = iris_train)
