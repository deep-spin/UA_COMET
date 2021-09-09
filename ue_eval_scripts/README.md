<!-- Evaluation and Result Reproduction -->
# Getting Started

The scripts and data in this folder allow to evaluate the predictions of COMET (or any other model MCD/ensemble outputs) with respect to uncertainty, and to reproduce the results and figures shown in [paper URL]. COMET predictions and human annotations are provided in the data/model_outputs and data/human_scores folders respectively. The necessary format and structure of those documents are explained in separate README files in these folders.

# Experimental Setups and Evaluation metrics

## Single reference uncertainty evaluation

This is the main experimental setup used to evaluate Uncertainty-Aware COMET. It assumes access to a csv with segment-level human MT-quality annotations [] and a 
csv with multiple segment-level quality estimates produced by multiple MCD runs or ensembles for the uncertainty-aware system being evaluated. 

Running the `evaluate_kfold_uncertainty.py` file will calculate the following metrics for the selected system and dataset scoring:

1. **PPS**: Predictive Pearson Score: measures the predictive accuracy of the system by calculating the Pearson correlationr between the human quality scores and the average system predictions. 
2. **UPS**: Uncertainty Pearson Score: measures the alignment between the prediction errors and the uncertainty estimates 
3. **NLL**: Negative log-likelihood. 
4. **ECE**: Expected calibration error. 
5. **Sharpness**: Average predicted variance (sigma**2). 

The following arguments can be used with `evaluate_kfold_uncertainty.py` to evaluate different setups:

`--comet-setup-file` = path to comet setup outputs (csv) to evaluate.  
`--scores-file` = path to human quality scores (csv) to test against.  
`--norm` = set to True to calibrate the predicted quality std on the ECE.  
`--score-type`= Choose type of scores between da | mqm | hter.  
`--docs` = Select segment or document level eval.  
`--nruns` = For MCD: Select how many dropout runs to evaluate | For ensembles: Select how many ensemble checkpoints to use.  
`--baseline` = Select to evaluate the baseline only.  
`--ensemble` = Select if the comet setup outputs are from ensemble instead of MCD.  

### Examples:
We present below examples and outputs for each of the datasets used in the associated publication:

1. __DA scores from WMT 2020 task__

2. __MQM scores__

3. __HTER scores from QT21__


## Multi-reference uncertainty evaluation


# Reproducing Tables and Figures in []

## Main paper:

* For the results of Table 2, run:
* For the results of Table 3, run:
* For the results of Table 4, run:
* For the results of Table 5, run:
* For the results of Table 6, run:
* For reproducing Figure 2, run:
* For reproducing Figure 3, run:

## Appendix:
* For the results of Table 11, run:
* For reproducing Figure 4, run:
