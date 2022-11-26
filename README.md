# pnml_for_dnn

Predictive Normalized Likelihood is the universal learner that minimizes the min-max regret for the individual data case.
However, this learner is intractable when the hypothesis class is with high capacity as deep neural networks.
In this repository, we combine pruning technique to effectively reduce the hypothesis class capacity and evaluate the pNML performance for it.

## Dependencies
The required packages are listed in the requirement.txt file.

## Running experimnets
```
# ERM baseline
cd src
python main_train_erm.py


# Train ERM -> Prune -> Train pNML
python main_pnml_from_pretrained.py

# Train pNML along with prunning
python main_pnml.py
```