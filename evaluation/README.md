<!-- Evaluation and Result Reproduction -->
# Getting Started

The scripts and data in this folder allow to evaluate the predictions of COMET (or any other model MCD/ensemble outputs) with respect to uncertainty, and to reproduce the results and figures shown in [paper URL]. COMET predictions and human annotations are provided in the data/model_outputs and data/human_scores folders respectively. The necessary format and structure of those documents are explained in separate README files in these folders.


# Experimental Setups and Evaluation metrics
The following description refers to evaluating uncertainty predictions for MT quality evaluation. The evaluation scripts assume a system that estimates its confidence by considering multiple predictions for the same instance (e.g. by MC dropout or ensemble). Note that the metrics calculated are generic enough to be applied on different uncertainty estimation methods but the input processing would need to be adapted.

The evaluation is performed for each language pair (LP) and scoring metric separately.

## Single reference uncertainty evaluation

This is the main experimental setup used to evaluate Uncertainty-Aware COMET. It assumes access to a csv with segment-level human MT-quality annotations [] and a 
csv with multiple segment-level quality estimates produced by multiple MCD runs or ensembles for the uncertainty-aware system being evaluated. 

Running the `evaluate_segment_uncertainty.py` file will calculate the following metrics for the selected system and dataset scoring:

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
Executing:   
    `python3 evaluate_segment_uncertainty.py --comet-setup-file data/model_outputs/wmt20/en_de/newstest2020/d01_n100_nrefs1_1719 --scores-file data/human_scores/da/en-de_scores_da.csv --score-type da`

Should produce the following outputs:


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

To calculate and plot the Precision@N and Recall@N for critical errors (as defined above) run the `evaluate_critical_mistake_retrieval.py` script. The following flags can be used:

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

* For the results of Table 2, for each language pair xx-yy in the table run:  
    * For MCD: `python3 evaluate_segment_uncertainty.py --comet-setup-file data/model_outputs/wmt20/xx_yy/newstest2020/d01_n100_nrefs1_1719 --scores-file data/human_scores/da/xx-yy_scores_da.csv --score-type da`  
    * For Deep Ensembles: `python3 evaluate_segment_uncertainty.py --comet-setup-file data/model_outputs/wmt20/xx_yy/newstest2020/ensemble_1719/merged_ensemble_1719_da.csv --scores-file data/human_scores/da/xx-yy_scores_da.csv --score-type da --ensemble `    
    * For the baseline: `python3 evaluate_segment_uncertainty.py --comet-setup-file data/model_outputs/wmt20/xx_yy/newstest2020/d01_n100_nrefs1_1719 --scores-file data/human_scores/da/xx-yy_scores_da.csv --score-type da --baseline`   
* For the results of Table 3, for each language pair xx-yy in the table run:  
    * For MCD: `python3 evaluate_segment_uncertainty.py --comet-setup-file data/model_outputs/wmt20/xx_yy/newstest2020/d01_n100_nrefs1_1719 --scores-file data/human_scores/mqm/2020-mqm-xx-yy.csv --score-type mqm`  
    * For Deep Ensembles: `python3 evaluate_segment_uncertainty.py --comet-setup-file data/model_outputs/wmt20/xx_yy/newstest2020/ensemble_1719/merged_ensemble_1719_mqm.csv --scores-file data/human_scores/mqm/2020-mqm-xx-yy.csv --score-type mqm --ensemble ` 
    * For the baseline: `python3 evaluate_segment_uncertainty.py --comet-setup-file data/model_outputs/wmt20/xx_yy/newstest2020/d01_n100_nrefs1_1719 --scores-file data/human_scores/mqm/2020-mqm-xx-yy.csv --score-type mqm --baseline`   
* For the results of Table 4, run:
    * For MCD (smt system example): `python3 evaluate_segment_uncertainty.py --comet-setup-file data/model_outputs/qt21/merged/merged_xx_yy_hter.csv --scores-file data/model_outputs/qt21/merged/merged_xx_yy_hter.csv --score-type hter`
    * For Deep Ensembles: `python3 evaluate_segment_uncertainty.py --comet-setup-file data/model_outputs/qt21/xx_yy/ensemble/merged_ensemble_hter.csv --scores-file data/model_outputs/qt21/merged/merged_xx_yy_hter.csv --score-type hter --ensemble ` 
    * For the baseline: `python3 evaluate_segment_uncertainty.py --comet-setup-file data/model_outputs/qt21/merged/merged_baseline_xx_yy_hter.csv --scores-file data/model_outputs/qt21/merged/merged_xx_yy_hter.csv --score-type hter --baseline `
* For the results of Table 5, run:  
    * For 3 reference ABP set:  
        * S-1:  
        `python3 evaluate_multi_ref.py --comet-setup-file /mnt/data-zeus1/chryssa/comet/setups/xx_yy/newstest2020/humanABP  --scores-file data/human_scores/mqm/2020-mqm-xx-yy.csv --score-type mqm --numrefs 3 --sample `
        * S-2:  
        `python3 evaluate_multi_ref.py --comet-setup-file /mnt/data-zeus1/chryssa/comet/setups/xx_yy/newstest2020/humanABP  --scores-file data/human_scores/mqm/2020-mqm-xx-yy.csv --score-type mqm --numrefs 3 --paireval `
        * Mul:  
        `python3 evaluate_multi_ref.py --comet-setup-file /mnt/data-zeus1/chryssa/comet/setups/xx_yy/newstest2020/humanABP  --scores-file data/human_scores/mqm/2020-mqm-xx-yy.csv --score-type mqm --numrefs 3 `  
    * For 2 reference MN set:  
        * S-1:  
        `python3 evaluate_multi_ref.py --comet-setup-file /mnt/data-zeus1/chryssa/comet/setups/xx_yy/newstest2020/humanMN  --scores-file data/human_scores/mqm/2020-mqm-xx-yy.csv --score-type mqm --numrefs 2 --sample `
        * Mul:  
        `python3 evaluate_multi_ref.py --comet-setup-file /mnt/data-zeus1/chryssa/comet/setups/xx_yy/newstest2020/humanMN  --scores-file data/human_scores/mqm/2020-mqm-xx-yy.csv --score-type mqm --numrefs 2 `
* For the results of Table 6, for each human reference M run:  
   `python3 evaluate_multi_ref.py --comet-setup-file /mnt/data-zeus1/chryssa/comet/setups/xx_yy/newstest2020/humanM  --scores-file data/human_scores/mqm/2020-mqm-xx-yy.csv --score-type mqm --numrefs 1`
* For reproducing Figures 2 and 3 run:  
    `mkdir figures`  
    `python3 evaluate_critical_mistake_retrieval.py --comet-setup-file /mnt/data-zeus1/glushkovato/comet/setups/xx_yy/newstest2020/d01_n100_nrefs1_qe_1719  --scores-file data/human_scores/mqm/2020-mqm-xx-yy.csv  --score-type mqm --comet-original-file /mnt/data-zeus1/glushkovato/comet/setups/xx_yy/newstest2020/original_comet_setup_qe/  --optimise --prefix test --norm_len `

`

