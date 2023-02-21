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

iris_lasso_spec <- multinom_reg(penalty = tune(), 
                                mixture = tune()) %>% 
  set_mode("classification") %>% 
  set_engine("glmnet")

iris_workflow <- workflow() %>% 
  add_recipe(iris_recipe) %>% 
  add_model(iris_lasso_spec)

iris_grid <- grid_regular(penalty(range = c(-5, 5)), 
                          mixture(range = c(0, 1)), levels = 10)

tune_res <- tune_grid(
  iris_workflow,
  resamples = iris_folds, 
  grid = iris_grid
)

bestpenalty <- select_best(tune_res,metrix="roc_auc")
bestpenalty

lasso_final<-finalize_workflow(iris_workflow,bestpenalty)

lasso_final_fit <- fit(lasso_final, data = iris_train)
