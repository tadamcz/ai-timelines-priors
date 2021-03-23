This repository contains the models used in the Open Philanthropy report "[Semi-informative priors over AI timelines](https://www.openphilanthropy.org/semi-informative-priors)".

The three most important files are the following:
* `functions.py` contains the functions that correspond to
  the models used in the report. 
* `sections.py` runs the models with the particular inputs that are used in the report.
Each function in this file corresponds to a particular section or table in the report.
* `print_output.py` runs every function in `sections.py`. You can run this file to see
the results of all calculations in order.
  
`test_bayesian_frequentist_equivalence.py` was used to check the correctness of some
of the functions  in `functions.py`.

`webapp_frontend.py` contains the code for the web application at [aipriors.com](https://aipriors.com), which is a user-friendly interface to the most important parts of the model.