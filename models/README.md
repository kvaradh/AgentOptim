# models/

Drop the inherited XGBoost models here, then run the gate:

    python -m scripts.verify_models --expected expected_predictions.json

Filenames tried per task are listed in `_MODEL_FILES` in `core/admet.py`
(`solubility.json`, `bbb.json`, `herg.json` and a few aliases). `.json` is
loaded as an xgboost Booster, `.pkl` as a pickled sklearn-style estimator.

Until all three load, `core.admet.get_backend()` serves the labelled surrogate
and every score record says so.
