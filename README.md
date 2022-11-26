# pnml_for_dnn

Predictive Normalized Likelihood is the universal learner that minimizes the min-max regret for the individual data case.
However, this learner is intercable when the hypoesis class is with high capacity as deep neural networks.
In this reposiroty, we combine pruning tecnique to effectivly reduce the hypotesis class capaccity and evalute the pNML perforamnce for it.


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