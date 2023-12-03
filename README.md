### Description
**Tools:** LightGBM, CatBoost, XGBoost, AutoML, Scikit-Learn, Pandas, Optuna, Shap  
**Concepts:** Ensembling, Cross-validation, Classifier calibration, Extracting business features, Feature importance analysis, Time Series Analysis    

*Topic: Customer churn prediction.*

This repository contains the **top-1** solution for Siberian Alpha Hack 2023. Our team implemented **ensemble of gradient boosting models**, tuned with optuna. Also we implemented decision tree on self generated "indicator-like" features as interpretable model for business.   
We performed calibration of our classifier based on synthetic example by applying different cost for FP/FN types errors (more info can be found in our presentation `DefendencePresentation.pdf`)      

### How to run inference:
In terminal execute the following command with specified arguments:
```sh
python inference.py --path_to_models_dir "models" --path_to_test_parquet "data/test.parquet"
```
