"""Collects all results and formats them into `.tsv.zip`s.
"""
import glob
import tqdm

import pandas as pd
import numpy as np

import sklearn
import sklearn.metrics


LAYERS_DATE = "instancewise"
T1_DATE = "final"
T2_DATE = "final"
T3T4_DATE = "final"


def main():
    li = collect(f"layers-{LAYERS_DATE}")
    # out = layers_details(li)
    # out.to_csv(
    #     "./figures/tbls/layers_details.tsv.zip",
    #     sep="\t",
    #     index=False,
    #     compression="zip",
    # )

    out = layers(li)
    out.to_csv(
        "./figures/tbls/layers.tsv.zip", sep="\t", index=False, compression="zip"
    )
    del li, out

    t1_df = collect(f"t1-{T1_DATE}")
    out = t1(t1_df)
    out.to_csv("./figures/tbls/t1.tsv.zip", sep="\t", index=False, compression="zip")
    del t1_df, out

    t2_df = collect(f"train-{T2_DATE}")
    out = t2(t2_df)
    out.to_csv("./figures/tbls/t2.tsv.zip", sep="\t", index=False, compression="zip")
    del t2_df, out

    t3t4_df = collect(f"causal-{T3T4_DATE}")
    out = t3(t3t4_df)
    out.to_csv("./figures/tbls/t3.tsv.zip", sep="\t", index=False, compression="zip")
    del out

    t3t4_df = collect(f"causal-{T3T4_DATE}")
    out = t4(t3t4_df)
    out.to_csv("./figures/tbls/t4.tsv.zip", sep="\t", index=False, compression="zip")
    del out


def collect(FOLDER):
    out = []
    for f in glob.glob(f"./results/{FOLDER}/*.tsv"):
        tbl = pd.read_table(f)
        out.append(tbl)
    return pd.concat(out)


classes = [
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


def get_layers(m):
    return {
        "ClipRes": [
            "conv_layer1",
            "conv_layer2",
            "conv_layer3",
            "layer1",
            "layer2",
            "layer3",
            "layer4",
            "attnpool",
        ],
        "ClipVit": [
            "conv_embedding",
            "transformer_0",
            "transformer_1",
            "transformer_2",
            "transformer_3",
            "transformer_4",
            "transformer_5",
            "transformer_6",
            "transformer_7",
            "transformer_8",
            "transformer_9",
            "transformer_10",
            "transformer_11",
        ],
        "OrigRes": ["conv_layer0", "layer1", "layer2", "layer3", "layer4"],
        "RawRes": ["conv_layer0", "layer1", "layer2", "layer3", "layer4"],
        "RawCnn": ["conv_layer0", "layer1", "layer2", "layer3", "fc"],
    }[m]


def layers_details(df):
    detail_results = []

    with tqdm.tqdm(
        total=len(df.holdout_many.unique())
        * len(df.seed.unique())
        * len(df.pretrain_model.unique()),
        desc="Generating",
    ) as pbar:

        for h in df.holdout_many.unique():
            for s in df.seed.unique():
                for pm in df.pretrain_model.unique():
                    t_pm = df.query(
                        "pretrain_model == @pm & seed == @s & holdout_many == @h"
                    )
                    for l in t_pm.layer.unique():
                        t = t_pm.query("layer == @l")
                        detail_results.append(
                            {
                                "value": t.layout_downstream_correct.mean(),
                                "target": "downstream",
                                "concept": "layout",
                                "pretrain_model": pm,
                                "layer": l,
                                "holdout_many": h,
                            }
                        )
                        detail_results.append(
                            {
                                "value": t.shape_downstream_correct.mean(),
                                "target": "downstream",
                                "concept": "shape",
                                "pretrain_model": pm,
                                "layer": l,
                                "holdout_many": h,
                            }
                        )
                        detail_results.append(
                            {
                                "value": t.stroke_downstream_correct.mean(),
                                "target": "downstream",
                                "concept": "stroke",
                                "pretrain_model": pm,
                                "layer": l,
                                "holdout_many": h,
                            }
                        )
                        detail_results.append(
                            {
                                "value": t.layout_correct.mean(),
                                "target": "probing",
                                "concept": "layout",
                                "pretrain_model": pm,
                                "layer": l,
                                "holdout_many": h,
                            }
                        )
                        detail_results.append(
                            {
                                "value": t.shape_correct.mean(),
                                "target": "probing",
                                "concept": "shape",
                                "pretrain_model": pm,
                                "layer": l,
                                "holdout_many": h,
                            }
                        )
                        detail_results.append(
                            {
                                "value": t.stroke_correct.mean(),
                                "target": "probing",
                                "concept": "stroke",
                                "pretrain_model": pm,
                                "layer": l,
                                "holdout_many": h,
                            }
                        )
                        detail_results.append(
                            {
                                "value": sklearn.metrics.normalized_mutual_info_score(
                                    t.layout_pred, t.layout_downstream
                                ),
                                "target": "nmi",
                                "concept": "layout",
                                "pretrain_model": pm,
                                "layer": l,
                                "holdout_many": h,
                            }
                        )
                        detail_results.append(
                            {
                                "value": sklearn.metrics.normalized_mutual_info_score(
                                    t.shape_pred, t.shape_downstream
                                ),
                                "target": "nmi",
                                "concept": "shape",
                                "pretrain_model": pm,
                                "layer": l,
                                "holdout_many": h,
                            }
                        )
                        detail_results.append(
                            {
                                "value": sklearn.metrics.normalized_mutual_info_score(
                                    t.stroke_pred, t.stroke_downstream
                                ),
                                "target": "nmi",
                                "concept": "stroke",
                                "pretrain_model": pm,
                                "layer": l,
                                "holdout_many": h,
                            }
                        )
                        detail_results.append(
                            {
                                "value": t[
                                    t.layout_correct.astype(bool)
                                ].layout_downstream_correct.mean(),
                                "target": "downstream | probing",
                                "concept": "layout",
                                "pretrain_model": pm,
                                "layer": l,
                                "holdout_many": h,
                            }
                        )
                        detail_results.append(
                            {
                                "value": t[
                                    t.shape_correct.astype(bool)
                                ].shape_downstream_correct.mean(),
                                "target": "downstream | probing",
                                "concept": "shape",
                                "pretrain_model": pm,
                                "layer": l,
                                "holdout_many": h,
                            }
                        )
                        detail_results.append(
                            {
                                "value": t[
                                    t.stroke_correct.astype(bool)
                                ].stroke_downstream_correct.mean(),
                                "target": "downstream | probing",
                                "concept": "stroke",
                                "pretrain_model": pm,
                                "layer": l,
                                "holdout_many": h,
                            }
                        )
                        detail_results.append(
                            {
                                "value": t[~t.layout_correct.astype(bool)]
                                .layout_downstream_correct.apply(lambda x: 1 - x)
                                .mean(),
                                "target": "!downstream | !probing",
                                "concept": "layout",
                                "pretrain_model": pm,
                                "layer": l,
                                "holdout_many": h,
                            }
                        )
                        detail_results.append(
                            {
                                "value": t[~t.shape_correct.astype(bool)]
                                .shape_downstream_correct.apply(lambda x: 1 - x)
                                .mean(),
                                "target": "!downstream | !probing",
                                "concept": "shape",
                                "pretrain_model": pm,
                                "layer": l,
                                "holdout_many": h,
                            }
                        )
                        detail_results.append(
                            {
                                "value": t[~t.stroke_correct.astype(bool)]
                                .stroke_downstream_correct.apply(lambda x: 1 - x)
                                .mean(),
                                "target": "!downstream | !probing",
                                "concept": "stroke",
                                "pretrain_model": pm,
                                "layer": l,
                                "holdout_many": h,
                            }
                        )
                    pbar.update()

    return pd.DataFrame(detail_results)


def layers(df):
    results = []

    with tqdm.tqdm(
        total=len(df.holdout_many.unique())
        * len(df.seed.unique())
        * len(df.pretrain_model.unique()),
        desc="Generating",
    ) as pbar:
        for h in df.holdout_many.unique():
            for s in df.seed.unique():
                for pm in df.pretrain_model.unique():
                    t_pm = df.query(
                        "pretrain_model == @pm & seed == @s & holdout_many == @h"
                    )

                    for l in t_pm.layer.unique():
                        t = t_pm.query("layer == @l")
                        results.append(
                            {
                                "value": sklearn.metrics.accuracy_score(
                                    t["probing_pred"], t["downstream_label"]
                                ),
                                "target": "probing",
                                "pretrain_model": pm,
                                "layer": l,
                                "holdout_many": h,
                            }
                        )
                        results.append(
                            {
                                "value": sklearn.metrics.accuracy_score(
                                    t["downstream_pred"], t["downstream_label"]
                                ),
                                "target": "downstream",
                                "pretrain_model": pm,
                                "layer": l,
                                "holdout_many": h,
                            }
                        )
                        results.append(
                            {
                                "value": sklearn.metrics.accuracy_score(
                                    t["shape_pred"], t["shape_pred"]
                                ),
                                "target": "shape",
                                "pretrain_model": pm,
                                "layer": l,
                                "holdout_many": h,
                            }
                        )
                        results.append(
                            {
                                "value": sklearn.metrics.accuracy_score(
                                    t["layout_pred"], t["layout_label"]
                                ),
                                "target": "layout",
                                "pretrain_model": pm,
                                "layer": l,
                                "holdout_many": h,
                            }
                        )
                        results.append(
                            {
                                "value": sklearn.metrics.accuracy_score(
                                    t["stroke_pred"], t["stroke_label"]
                                ),
                                "target": "stroke",
                                "pretrain_model": pm,
                                "layer": l,
                                "holdout_many": h,
                            }
                        )
                        results.append(
                            {
                                "value": sklearn.metrics.normalized_mutual_info_score(
                                    t.probing_pred, t.downstream_pred
                                ),
                                "target": "nmi",
                                "pretrain_model": pm,
                                "layer": l,
                                "holdout_many": h,
                            }
                        )
                    pbar.update()
    return pd.DataFrame(results)


def t1(df):
    t = (
        df.groupby(["data_path", "pretrain_model", "layer", "seed", "classname"])
        .mean()
        .reset_index()
    )
    # We just look at the last layer here.
    t = t[t.apply(lambda x: x.layer == get_layers(x.pretrain_model)[-1], 1)]
    return t


def t2(df):
    df = df.query("dataspec != 'classes' & data_path == 'default'")
    t = (
        df.groupby(
            [
                "data_path",
                "pretrain_model",
                "dataspec",
                "layer",
                "seed",
                "classname",
                "holdout",
                "holdout_many",
            ]
        )
        .mean()
        .reset_index()
    )
    t = t[t.apply(lambda x: x.layer == get_layers(x.pretrain_model)[-1], 1)]
    t = t[t.holdout == "unseen"]
    for_chart = t
    for_chart["holdout_condition"] = for_chart.holdout_many.map(
        {False: "N - 1 Slice", True: "1 Slice"}
    )
    for_chart = (
        for_chart.groupby(["pretrain_model", "holdout_condition", "dataspec"])
        .accuracy.mean()
        .reset_index()
    )
    return for_chart


def t3(df):
    df = df.query(
        "data_path == 'default' & holdout == 'unseen' & dataspec_downstream != 'classes' & dataspec_intervene != 'classes'"
    ).copy()
    df["holdout_condition"] = df.holdout_many.map(
        {False: "N - 1 Slice", True: "1 Slice"}
    )
    df["condition"] = df.apply(
        lambda x: {True: "ablated", False: "others"}[
            x.dataspec_downstream == x.dataspec_intervene
        ],
        1,
    )
    return df


def t4(df):
    t = df.query(
        "data_path == 'default' & holdout == 'unseen' & dataspec_downstream == 'classes' & dataspec_intervene != 'classes'"
    ).copy()
    t["ablated"] = t.apply(get_rm_acc, axis=1)
    t["others"] = t.apply(get_alt_acc, axis=1)
    t = t[
        [
            "pretrain_model",
            "classname",
            "dataspec_intervene",
            "holdout_many",
            "seed",
            "ablated",
            "others",
        ]
    ]
    t = pd.melt(
        t,
        id_vars=[
            "pretrain_model",
            "classname",
            "dataspec_intervene",
            "holdout_many",
            "seed",
        ],
        value_vars=["ablated", "others",],
        var_name="condition",
        value_name="accuracy",
    )
    return t


def get_acc(x):
    classname = x["classname"]
    n = x[classname]
    d = sum(x[classes])  # 100
    return n / d


def _get_concept_acc(x, concept):
    counts = sum(x[classes])
    concept_counts = sum([x[clss] for clss in classes if concept in clss])
    return concept_counts / counts


def get_rm_acc(x):
    category_idx = CAT2IDX[x.dataspec_intervene]
    concept = x.classname.split("-")[category_idx]
    return _get_concept_acc(x, concept)


def get_alt_acc(x):
    category_idx = CAT2IDX[x.dataspec_intervene]
    return np.mean(
        [
            _get_concept_acc(x, concept)
            for i, concept in enumerate(x.classname.split("-"))
            if i != category_idx
        ]
    )


if __name__ == "__main__":
    main()
