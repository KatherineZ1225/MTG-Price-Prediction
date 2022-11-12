```
|-- MTG Price Pirdection                  <- Project root level that is checked into github
  |--3. MTG Price                         <- Project folder
    |--README.md                          <- Top-level README for developers
    |--volume
    |   |--data
    |   |   |--external                   <- Data from third party sources
    |   |   |--interim                    <- Intermediate data that has been transformed
    |   |   |--processed                  <- The final model-ready data
    |   |   |--raw                        <- The original data dump
    |   |
    |   |--models                         <- Trained model files that can be read into R or Python
    |
    |--required
    |   |--requirements.txt               <- The required libraries for reproducing the Python environment
    |   |--requirements.r                 <- The required libraries for reproducing the R environment
    |
    |
    |--src
    |   |
    |   |--features                       <- Scripts for turning raw and external data into model-ready data
    |   |   |--feature_eng_v2.r
    |   |
    |   |--models                         <- Scripts for training and saving models
    |   |   |--glmnet_model.r
    |   |
    |
    |
    |
    |--.getignore                         <- List of files not to sync with github
```
