To run:
1. First, you need to process all datasets by running `python preprocess.py`
2. Then, you can define the hyper-parameters for the models in `params.json`
3. Finally, you can train a model by running `python baseline.py --learner lgb-rmse --fold MQ2008/Fold1 --param-space params.json --out-file out --log-file log` or equivalent