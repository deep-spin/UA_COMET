<p align="center">
  <img src="https://raw.githubusercontent.com/Unbabel/COMET/master/docs/source/_static/img/COMET_lockup-dark.png">
  <br />
  <br />
  <a href="https://github.com/Unbabel/COMET/blob/master/LICENSE"><img alt="License" src="https://img.shields.io/github/license/Unbabel/COMET" /></a>
  <a href="https://github.com/Unbabel/COMET/stargazers"><img alt="GitHub stars" src="https://img.shields.io/github/stars/Unbabel/COMET" /></a>
  <a href=""><img alt="PyPI" src="https://img.shields.io/pypi/v/unbabel-comet" /></a>
  <a href="https://github.com/psf/black"><img alt="Code Style" src="https://img.shields.io/badge/code%20style-black-black" /></a>
</p>



#### This repository presents UA-COMET – an extension of the COMET metric implemented by Unbabel. 

#### It contains the code and data to reproduce the experiments in [Uncertainty-Aware Machine Translation Evaluation]().


## Quick Installation

We recommend python 3.6 to run COMET.

Detailed usage examples and instructions can be found in the [Full Documentation](https://unbabel.github.io/COMET/html/index.html).

Simple installation from PyPI

```bash
pip install unbabel-comet
```

To develop locally:
```bash
git clone https://github.com/Unbabel/COMET
pip install -r requirements.txt
pip install -e .
```

## Scoring MT outputs:

### Via Bash:

Examples from WMT20:

```bash
echo -e "Dem Feuer konnte Einhalt geboten werden\nSchulen und Kindergärten wurden eröffnet." >> src.de
echo -e "The fire could be stopped\nSchools and kindergartens were open" >> hyp.en
echo -e "They were able to control the fire.\nSchools and kindergartens opened" >> ref.en
```

```bash
comet score -s src.de -h hyp.en -r ref.en
```

You can export your results to a JSON file using the `--to_json` flag and select another model/metric with `--model`.

```bash
comet score -s src.de -h hyp.en -r ref.en --model wmt-large-hter-estimator --to_json segments.json
```

### Via Python:

```python
from comet.models import download_model
model = download_model("wmt-large-da-estimator-1719")
data = [
    {
        "src": "Dem Feuer konnte Einhalt geboten werden",
        "mt": "The fire could be stopped",
        "ref": "They were able to control the fire."
    },
    {
        "src": "Schulen und Kindergärten wurden eröffnet.",
        "mt": "Schools and kindergartens were open",
        "ref": "Schools and kindergartens opened"
    }
]
model.predict(data, cuda=True, show_progress=True)
```

## Scoring MT outputs with MCD runs

To run COMET with multiple MCD runs:

```bash
 #!/bin/bash
 
GPU_N=3

SCORES=/path/to/your/output/folder
DATA=/path/to/your/data/folder

N=100
D=0.1
N_REFS=1

SRC=src.txt
MT=mt.txt
REF=ref.txt

MODEL=wmt-large-da-estimator-1719

echo Starting the process...

CUDA_VISIBLE_DEVICES=$GPU_N comet score \
  -s $DATA/sources/$SRC \
  -h $DATA/system-outputs/$MT \
  -r $DATA/references/$REF \
  --to_json $SCORES/filename.json \
  --n_refs $N_REFS \
  --n_dp_runs $N \
  --d_enc $D \
  --d_pool $D \
  --d_ff1 $D \
  --d_ff2 $D \
  --model $MODEL 

```

This will run the model with a set of hyperparameters defined above. Here is the description of the main scoring arguments:

`-s`: Source segments.    
`-h`: MT outputs.    
`-r`: Reference segments.     
`--to_json`: Creates and exports model predictions to a JSON file.     
`--n_refs`: Number of references used during inference. [default=1]  
`--n_dp_runs`: Number of dropout runs at test time. [default=30]  
`--d_enc`: Dropout value for the encoder. [default=0.1]  
`--d_pool`: Dropout value for the layerwise pooling layer. [default=0.1]       
`--d_ff1`: Dropout value for the 1st feed forward layer. [default=0.1]       
`--d_ff2`: Dropout value for the 2nd feed forward layer. [default=0.1]        
`--model`: Name of the pretrained model OR path to a model checkpoint.     

To know more about the rest of the parameters and their default values, take a look at the ```comet/cli.py``` file.

## How to Reproduce and Evaluate Experiments

The ```evaluation``` sub-folder contains the scripts and data necessary to reproduce the experiments presented in [Uncertainty-Aware Machine Translation Evaluation]() and/or test new model outputs. See the README in that folder for more detailed instructions.

<!-- ### Simple Pythonic way to convert list or segments to model inputs:

```python
source = ["Dem Feuer konnte Einhalt geboten werden", "Schulen und Kindergärten wurden eröffnet."]
hypothesis = ["The fire could be stopped", "Schools and kindergartens were open"]
reference = ["They were able to control the fire.", "Schools and kindergartens opened"]

data = {"src": source, "mt": hypothesis, "ref": reference}
data = [dict(zip(data, t)) for t in zip(*data.values())]

model.predict(data, cuda=True, show_progress=True)
```

**Note:** Using the python interface you will get a list of segment-level scores. You can obtain the corpus-level score by averaging the segment-level scores -->

## Model Zoo:

The COMET models used for uncertainty-aware MT evaluation experiments are: 
* `wmt-large-da-estimator-1719`  for the WMT20 dataset (DA/MQM scores)
* `wmt-large-hter-estimator` for the QT21 dataset (HTER scores)  

Available and compatible models are:

| Model              |               Description                        |
| :--------------------- | :------------------------------------------------ |
| ↑`wmt-large-da-estimator-1719` | **RECOMMENDED:** Estimator model build on top of XLM-R (large) trained on DA from WMT17, WMT18 and WMT19 |
| ↑`wmt-base-da-estimator-1719` | Estimator model build on top of XLM-R (base) trained on DA from WMT17, WMT18 and WMT19 |
| ↓`wmt-large-hter-estimator` | Estimator model build on top of XLM-R (large) trained to regress on HTER. |
| ↓`wmt-base-hter-estimator` | Estimator model build on top of XLM-R (base) trained to regress on HTER. |


## Train your own Metric: 

Instead of using pretrained models your can train your own COMET model with the following command:
```bash
comet train -f {config_file_path}.yaml
```
For more information check: [COMET's documentation](https://unbabel.github.io/COMET/html/training.html).

Alternatively, it is possible to train a different metric and compare performance using the scripts in the evaluation sub-folder. In this case, ensure the metric output files maintain the same structure as described in ```evaluation/data/README.md```.



## Publications

```
@inproceedings{rei-etal-2020-comet,
    title = "{COMET}: A Neural Framework for {MT} Evaluation",
    author = "Rei, Ricardo  and
      Stewart, Craig  and
      Farinha, Ana C  and
      Lavie, Alon",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.emnlp-main.213",
    pages = "2685--2702",
}
```

```
@inproceedings{rei-EtAl:2020:WMT,
  author    = {Rei, Ricardo  and  Stewart, Craig  and  Farinha, Ana C  and  Lavie, Alon},
  title     = {Unbabel's Participation in the WMT20 Metrics Shared Task},
  booktitle      = {Proceedings of the Fifth Conference on Machine Translation},
  month          = {November},
  year           = {2020},
  address        = {Online},
  publisher      = {Association for Computational Linguistics},
  pages     = {909--918},
}
```

```
@inproceedings{stewart-etal-2020-comet,
    title = "{COMET} - Deploying a New State-of-the-art {MT} Evaluation Metric in Production",
    author = "Stewart, Craig  and
      Rei, Ricardo  and
      Farinha, Catarina  and
      Lavie, Alon",
    booktitle = "Proceedings of the 14th Conference of the Association for Machine Translation in the Americas (Volume 2: User Track)",
    month = oct,
    year = "2020",
    address = "Virtual",
    publisher = "Association for Machine Translation in the Americas",
    url = "https://www.aclweb.org/anthology/2020.amta-user.4",
    pages = "78--109",
}
```
