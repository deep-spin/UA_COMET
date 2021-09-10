<!-- Evaluation and Result Reproduction -->
# Getting Started

The scripts and data in this folder allow to evaluate the predictions of COMET (or any other model MCD/ensemble outputs) with respect to uncertainty, and to reproduce the results and figures shown in [paper URL]. COMET predictions and human annotations are provided in the data/model_outputs and data/human_scores folders respectively. The necessary format and structure of those documents are explained in separate README files in these folders.


# Experimental Setups and Evaluation metrics
The following description refers to evaluating uncertainty predictions for MT quality evaluation. The evaluation scripts assume a system that estimates its confidence by considering multiple predictions for the same instance (e.g. by MC dropout or ensemble). Note that the metrics calculated are generic enough to be applied on different uncertainty estimation methods but the input processing would need to be adapted.

The evaluation is performed for each language pair (LP) and scoring metric separately.

## Single reference uncertainty evaluation

This is the main experimental setup used to evaluate Uncertainty-Aware COMET. It assumes access to a csv with segment-level human MT-quality annotations [] and a 
csv with multiple segment-level quality estimates produced by multiple MCD runs or ensembles for the uncertainty-aware system being evaluated. 

Running the `evaluate_kfold_uncertainty.py` file will calculate the following metrics for the selected system and dataset scoring:

1. **PPS**: Predictive Pearson Score: measures the predictive accuracy of the system by calculating the Pearson correlationr between the human quality scores and the average system predictions. 
2. **UPS**: Uncertainty Pearson Score: measures the alignment between the prediction errors and the uncertainty estimates 
3. **NLL**: Negative log-likelihood. 
4. **ECE**: Expected calibration error. 
5. **Sharpness**: Average predicted variance (sigma**2). 

The folder data/model_outputs/ contains COMET outputs that can be already used for this setup.   The following arguments can be used with `evaluate_kfold_uncertainty.py` to evaluate different setups:

`--comet-setup-file` = path to folder with model outputs for the setup to evaluate. Example:   
`--scores-file` = path to human quality scores (csv) to test against. Example:     
`--norm` = set to True to calibrate the predicted quality std on the ECE.  
`--score-type`= Choose type of scores between da | mqm | hter.  
`--docs` = Select segment or document level eval.  
`--nruns` = For MCD: Select how many dropout runs to evaluate | For ensembles: Select how many ensemble checkpoints to use.  
`--baseline` = Select to evaluate the baseline only.  
`--ensemble` = Select if the comet setup outputs are from ensemble instead of MCD.  

### Examples
We present below examples and outputs for each of the datasets used in the associated publication:

1. __DA scores from WMT 2020 task__

2. __MQM scores__

3. __HTER scores from QT21__


## Multi-reference uncertainty evaluation
In this experiment we evaluate the performance of uncertainty prediction when using more than oe references per segment. Evaluating this setup experiments requires running with a different script `evaluate_multi_ref.py` . 


This script requires formatting the setup repository differently: Run the system on each of the available references generating the model_output directory as in the previous experiments and then group these directories under a parent directory that will beprovided asinput to the script. The folder data/model_outputs/multi_ref contains such examples.

  The following arguments can be used with `evaluate_multi_ref.py` to evaluate different setups:

`--comet-setup-file` = path to folder with model outputs for the setup to evaluate. Example:   
`--scores-file` = path to human quality scores (csv) to test against. Example:     
`--norm` = set to True to calibrate the predicted quality std on the ECE.  
`--score-type`= Choose type of scores between da | mqm .  
`--docs` = Select segment or document level eval.  
`--nruns` = For MCD: Select how many dropout runs to evaluate | For ensembles: Select how many ensemble checkpoints to use.  
`--sample` = Set to sample one reference from a multi-reference set instead of averaging over them.  
`--numrefs`= Select over how many references to evaluate.  
`--paireval` = Set to sample pairs of references from a multi-reference set.  
`--lp` = Choose which language pair to evaluate over.  

### Examples:
We present below examples and outputs for each of the datasets used in the associated publication:

1. **Sample over 3 references**.  
    ```python3 evaluate_multi_ref.py --comet-setup-file /mnt/data-zeus1/chryssa/comet/setups/en_de/newstest2020/multi_human/ --scores-file /mnt/data-zeus1/chryssa/metrics-data/2020-mqm.csv --numrefs 3 --sample```

2. **Sample pair over 3 references**.  
    ```python3 evaluate_multi_ref.py --comet-setup-file /mnt/data-zeus1/chryssa/comet/setups/en_de/newstest2020/multi_human/ --scores-file /mnt/data-zeus1/chryssa/metrics-data/2020-mqm.csv --numrefs 3 --paireval```
    
3. **Average over 3 references**.  
    ```python3 evaluate_multi_ref.py --comet-setup-file /mnt/data-zeus1/chryssa/comet/setups/en_de/newstest2020/multi_human/ --scores-file /mnt/data-zeus1/chryssa/metrics-data/2020-mqm.csv --numrefs 3```


## Critical translation error retrieval

In this experiment we evaluate the efficiency of the uncertainty aware system in ranking segments by the quality and identifying the most erroneous cases. The experimental setup is evaluated using Precision@N and Recall@N comparing three methods:  

1. Use of the original COMET model quality predictions to rank the segments (COMET original)
2. Use the mean quality score calculated with MC dropout to rank the segments (mean COMET MCD)
3. Use the mean and std values calculated with MC dropout to calculate the probability that the quality is below a criticcal threshold, using the cumulative distribution function (CDF)

For this experiment we use PRISM translations of the source sentences instead of the orginal human references. We provide the translations for the WMT2020 English-German data in: []

To calculate and plot the Precision@N and Recall@N for critical errors (as defined above) run the `evaluate_critical_error_retrieval.py` script. The following flags can be used:

`--comet-setup-file` = path to folder with model outputs for the setup to evaluate. Example:   
`--scores-file` = path to human quality scores (csv) to test against. Example:   
`--comet-original-file` = path to folder with model outputs without MC dropout. Example:   
`--norm` = set to True to calibrate the predicted quality std on the ECE.  
`--score-type`= Choose type of scores between da | mqm .  
`--nruns` = For MCD: Select how many dropout runs to evaluate | For ensembles: Select how many ensemble checkpoints to use.  
`--lp` = Choose which language pair to evaluate over.  
`--dev-first` = select which half to be used as dev set.  
`--optimise` = Set to true to optimise the critical error threshold (on recall values).  
`--prefix` =  Set prefix of the plots to be saved. 
`--norm_len`= Set to normalize human scores by sentence length. 


# Reproducing Tables and Figures

We include below the commands needed to reproduce the results presented in []

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
