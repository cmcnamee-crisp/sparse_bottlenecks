"""Process results, producing a tsv for each test.
"""
import glob

import numpy as np
import pandas as pd


def collect(FOLDER: str) -> pd.DataFrame:
    """Loads all tsv in the given folder into a single dataframe. """
    out = []
    for f in glob.glob(f"{FOLDER}/*.tsv"):
        tbl = pd.read_table(f)
        out.append(tbl)
    return pd.concat(out)


def model_name(x: pd.Series) -> str:
    """Generate model name used for matplotlib figures. """
    if x.pretrain_model == "default":
        return {"cnn": "$CNN_{NoPre}$", "RN50NOPRE": "$RN_{NoPre}$"}[x.finetune_model]
    else:
        return {"RN50": r"${RN}_{CLIP}$", "OrigRes": r"${RN}_{IMG}$"}[x.pretrain_model]


def t1(FOLDER: str) -> pd.DataFrame:
    """This is slow as the t1 script did not aggregate results. """
    df = collect(FOLDER)
    df["correct"] = df["classname"] == df["prediction_name"]
    df["model"] = df.apply(model_name, axis=1)
    df = df.groupby(["batch_size", "data_path", "seed", "model"]).mean().reset_index()
    return df


def t2(FOLDER: str) -> pd.DataFrame:
    df = collect(FOLDER)
    df["model"] = df.apply(model_name, axis=1)
    return df


def t3(FOLDER: str) -> pd.DataFrame:
    df = collect(FOLDER)
    df["model"] = df.apply(model_name, axis=1)
    return df


def t4(FOLDER: str) -> pd.DataFrame:
    def _collect_confusion_mtx(FOLDER: str) -> pd.DataFrame:
        """Loads the confusion matrices specifically.
        (Previously, there were summarized tsvs in the same folder.)
        """
        out = []
        for f in glob.glob(f"./results/{FOLDER}/confu*.tsv"):
            tbl = pd.read_table(f)
            out.append(tbl)
        return pd.concat(out)

    CLASSES = [
        "column-oval-clean",
        "column-oval-fuzzy",
        "column-poly-clean",
        "column-poly-fuzzy",
        "column-rectangle-clean",
        "column-rectangle-fuzzy",
        "ring-oval-clean",
        "ring-oval-fuzzy",
        "ring-poly-clean",
        "ring-poly-fuzzy",
        "ring-rectangle-clean",
        "ring-rectangle-fuzzy",
        "row-oval-clean",
        "row-oval-fuzzy",
        "row-poly-clean",
        "row-poly-fuzzy",
        "row-rectangle-clean",
        "row-rectangle-fuzzy",
    ]
    CAT2IDX = {"layout": 0, "shape": 1, "stroke": 2}

    def _get_concept_acc(x: pd.Series, concept: str) -> float:
        concept_counts = sum([x[clss] for clss in CLASSES if concept in clss])
        d = 1000  #  sum(x[classes]) # 100
        return concept_counts / d

    def _get_rm_acc(x: pd.Series) -> float:
        category_idx = CAT2IDX[x.dataspec_analysis]
        concept = x.classname.split("-")[category_idx]
        return _get_concept_acc(x, concept)

    def _get_alt_acc(x: pd.Series) -> float:
        category_idx = CAT2IDX[x.dataspec_analysis]
        return np.mean(
            [
                _get_concept_acc(x, concept)
                for i, concept in enumerate(x.classname.split("-"))
                if i != category_idx
            ]
        )

    df = _collect_confusion_mtx(FOLDER)

    df["model"] = df.apply(model_name, axis=1)
    df["others"] = df.apply(_get_alt_acc, axis=1)
    df["target"] = df.apply(_get_rm_acc, axis=1)

    return df


if __name__ == "__main__":
    DATE = "2021-06-23"

    t1(f"results/t1-{DATE}").to_csv("t1.tsv", index=False, sep="\t")
    t2(f"results/t2-{DATE}").to_csv("t2.tsv", index=False, sep="\t")
    t3(f"results/t3-{DATE}").to_csv("t3.tsv", index=False, sep="\t")
    t4(f"results/t4-{DATE}").to_csv("t4.tsv", index=False, sep="\t")
