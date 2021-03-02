import json
import os
import yaml
import argparse
from comet.corpora import corpus2download, download_corpus
from comet.models import download_model, load_checkpoint, model2download, str2model
from comet.trainer import TrainerConfig, build_trainer
from pytorch_lightning import seed_everything

def download(data, model, saving_path):
    for d in data:
        download_corpus(d, saving_path)

    for m in model:
        download_model(m, saving_path)


def score(model, source, hypothesis, reference, cuda, batch_size):
    with open(source,'r') as src:
        source = [s.strip() for s in src.readlines()]
    with open(hypothesis, 'r') as hyp:
        hypothesis = [s.strip() for s in hyp.readlines()]
    with open(reference, 'r') as ref:
        reference = [s.strip() for s in ref.readlines()]
    data = {"src": source, "mt": hypothesis, "ref": reference}
    data = [dict(zip(data, t)) for t in zip(*data.values())]

    model = load_checkpoint(model) if os.path.exists(model) else download_model(model)
    data, scores = model.predict(data, cuda, show_progress=True, batch_size=batch_size)

    for i in range(len(scores)):
        print("Segment {} score: {:.3f}".format(i, scores[i]))
    print("COMET system score: {:.3f}".format(sum(scores) / len(scores)))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Quick COMET dropout tester."
    )
    parser.add_argument(
        "--checkpoint",
        default="_ckpt_epoch_1.ckpt",
        help="Path to the Model checkpoint we want to test.",
        type=str,
    )
    parser.add_argument(
        "--model",
        default="wmt-large-da-estimator-1719",
        help="trained model to test",
        type=str,
    )
    parser.add_argument(
        "--src",
        default="src.de",
        help="source",
        type = str,
    )
    parser.add_argument(
        "--mt",
        default="hyp.en",
        help="MT hypothesis",
        type=str,
    )
    parser.add_argument(
        "--ref",
        default="ref.en",
        help="Reference",
        type=str,
    )
    parser.add_argument(
        "--batch_size",
        default=-1,
        help="Batch size used during inference. By default uses the same batch size used during training.",
        type=int,
    )
    parser.add_argument(
        "--cuda", default=False, help="Uses cuda.", action="store_true",
    )
    args = parser.parse_args()
    #model = load_checkpoint(args.checkpoint)


    score(args.model, args.src, args.mt, args.ref, args.cuda, batch_size=-1)


