# Model evaluation statistics

`paired_stats.py` publishes the T-381 evaluation utility from Selective Hearing (source commit `61e5dd3032277059369749ca0f9762fbad19b6d8`). It summarizes mean and sample standard deviation across seeds and compares matched wrong-speaker outcomes with McNemar statistics. This is evaluation tooling, not a trained model or an accuracy claim.

Run `python3 tools/paired_stats.py --help` for commands and see its module documentation for input formats. No recordings, checkpoints, or private results are included. Sample standard deviation is not a confidence interval. Window-level p-values are exploratory when windows overlap or are correlated.
