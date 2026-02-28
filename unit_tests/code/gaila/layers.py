import torch as th
from pytorch_lightning import seed_everything

import pandas as pd
import os
import time
import tqdm

import flags
import helpers

from models import linear

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("times", exist_ok=True)

label2class = {
    0: "column-oval-clean",
    1: "column-oval-fuzzy",
    4: "column-rectangle-clean",
    5: "column-rectangle-fuzzy",
    2: "column-poly-clean",
    3: "column-poly-fuzzy",
    12: "row-oval-clean",
    13: "row-oval-fuzzy",
    16: "row-rectangle-clean",
    17: "row-rectangle-fuzzy",
    14: "row-poly-clean",
    15: "row-poly-fuzzy",
    6: "ring-oval-clean",
    7: "ring-oval-fuzzy",
    10: "ring-rectangle-clean",
    11: "ring-rectangle-fuzzy",
    8: "ring-poly-clean",
    9: "ring-poly-fuzzy",
}
class2label = {
    "column-oval-clean": 0,
    "column-oval-fuzzy": 1,
    "column-rectangle-clean": 4,
    "column-rectangle-fuzzy": 5,
    "column-poly-clean": 2,
    "column-poly-fuzzy": 3,
    "row-oval-clean": 12,
    "row-oval-fuzzy": 13,
    "row-rectangle-clean": 16,
    "row-rectangle-fuzzy": 17,
    "row-poly-clean": 14,
    "row-poly-fuzzy": 15,
    "ring-oval-clean": 6,
    "ring-oval-fuzzy": 7,
    "ring-rectangle-clean": 10,
    "ring-rectangle-fuzzy": 11,
    "ring-poly-clean": 8,
    "ring-poly-fuzzy": 9,
}

class2shapelabel = {
    "column-oval-clean": 0,
    "column-oval-fuzzy": 0,
    "column-rectangle-clean": 2,
    "column-rectangle-fuzzy": 2,
    "column-poly-clean": 1,
    "column-poly-fuzzy": 1,
    "row-oval-clean": 0,
    "row-oval-fuzzy": 0,
    "row-rectangle-clean": 2,
    "row-rectangle-fuzzy": 2,
    "row-poly-clean": 1,
    "row-poly-fuzzy": 1,
    "ring-oval-clean": 0,
    "ring-oval-fuzzy": 0,
    "ring-rectangle-clean": 2,
    "ring-rectangle-fuzzy": 2,
    "ring-poly-clean": 1,
    "ring-poly-fuzzy": 1,
}
class2layoutlabel = {
    "column-oval-clean": 0,
    "column-oval-fuzzy": 0,
    "column-rectangle-clean": 0,
    "column-rectangle-fuzzy": 0,
    "column-poly-clean": 0,
    "column-poly-fuzzy": 0,
    "row-oval-clean": 2,
    "row-oval-fuzzy": 2,
    "row-rectangle-clean": 2,
    "row-rectangle-fuzzy": 2,
    "row-poly-clean": 2,
    "row-poly-fuzzy": 2,
    "ring-oval-clean": 1,
    "ring-oval-fuzzy": 1,
    "ring-rectangle-clean": 1,
    "ring-rectangle-fuzzy": 1,
    "ring-poly-clean": 1,
    "ring-poly-fuzzy": 1,
}
class2strokelabel = {
    "column-oval-clean": 0,
    "column-oval-fuzzy": 1,
    "column-rectangle-clean": 0,
    "column-rectangle-fuzzy": 1,
    "column-poly-clean": 0,
    "column-poly-fuzzy": 1,
    "row-oval-clean": 0,
    "row-oval-fuzzy": 1,
    "row-rectangle-clean": 0,
    "row-rectangle-fuzzy": 1,
    "row-poly-clean": 0,
    "row-poly-fuzzy": 1,
    "ring-oval-clean": 0,
    "ring-oval-fuzzy": 1,
    "ring-rectangle-clean": 0,
    "ring-rectangle-fuzzy": 1,
    "ring-poly-clean": 0,
    "ring-poly-fuzzy": 1,
}

pred2layout = {
    0: "column",
    1: "ring",
    2: "row",
}
pred2shape = {0: "oval", 1: "poly", 2: "rectangle"}
pred2stroke = {0: "clean", 1: "fuzzy"}


def shape_label(label):
    # 0 oval, 1 poly, 2 rectangle
    classname = label2class[label]
    return class2shapelabel[classname]


def layout_label(label):
    # 0 column, 1 ring, 2 row
    classname = label2class[label]
    return class2layoutlabel[classname]


def stroke_label(label):
    # 0 clean, 1 stroke
    classname = label2class[label]
    return class2strokelabel[classname]


def probing_prediction(row):
    return class2label[
        f"{pred2layout[row.layout_pred]}-{pred2shape[row.shape_pred]}-{pred2stroke[row.stroke_pred]}"
    ]


def layout_downstream(row):
    return layout_label(row.downstream_pred)


def shape_downstream(row):
    return shape_label(row.downstream_pred)


def stroke_downstream(row):
    return stroke_label(row.downstream_pred)


def main(FLAGS):
    seed_everything(FLAGS.seed)
    device = "cpu"
    out = []
    for layer in tqdm.tqdm(flags.get_layers(FLAGS), desc="Layers"):

        FLAGS.dataspec = "shape"
        train_id = flags.train_layerwise_id(FLAGS, layer)
        shape_model = linear.Linear.load_from_checkpoint(
            checkpoint_path=f"checkpoints/{train_id}.ckpt"
        )

        FLAGS.dataspec = "stroke"
        train_id = flags.train_layerwise_id(FLAGS, layer)
        stroke_model = linear.Linear.load_from_checkpoint(
            checkpoint_path=f"checkpoints/{train_id}.ckpt"
        )

        FLAGS.dataspec = "layout"
        train_id = flags.train_layerwise_id(FLAGS, layer)
        layout_model = linear.Linear.load_from_checkpoint(
            checkpoint_path=f"checkpoints/{train_id}.ckpt"
        )

        FLAGS.dataspec = "classes"
        holdout_many = FLAGS.holdout_many
        FLAGS.holdout_many = False
        train_id = flags.train_layerwise_id(FLAGS, layer)
        datadesc, dataspec = helpers.load_info(FLAGS)
        downstream_model = linear.Linear.load_from_checkpoint(
            checkpoint_path=f"checkpoints/{train_id}.ckpt"
        )
        FLAGS.holdout_many = holdout_many
        datamodule, _ = helpers.get_datamodule_layer(
            flags.transformed_id(FLAGS),
            datadesc,
            dataspec,
            FLAGS.batch_size,
            device,
            layer=layer,
            flatten="avg",
        )
        out.append(
            instancelevel_predictions(
                FLAGS,
                shape_model,
                stroke_model,
                layout_model,
                downstream_model,
                datamodule,
                info={"layer": layer},
            )
        )
    train_id = flags.train_id(FLAGS)
    pd.concat(out).to_csv(
        f"results/{FLAGS.jobname}/{train_id}.tsv",
        sep="\t",
        index=False,
    )


def instancelevel_predictions(
    FLAGS,
    shape_model,
    stroke_model,
    layout_model,
    downstream_model,
    datamodule,
    info=None,
):
    if info is None:
        info = {}
    labels = []
    shape_pred = []
    stroke_pred = []
    layout_pred = []
    downstream_pred = []

    with th.no_grad():
        shape_model.eval()
        stroke_model.eval()
        layout_model.eval()
        for X, y in datamodule.train_dataloader():
            labels.append(y)
            shape_pred.append(shape_model(X).argmax(1))
            stroke_pred.append(stroke_model(X).argmax(1))
            layout_pred.append(layout_model(X).argmax(1))
            downstream_pred.append(downstream_model(X).argmax(1))

    labels = th.cat(labels).numpy()
    shape_pred = th.cat(shape_pred).numpy()
    stroke_pred = th.cat(stroke_pred).numpy()
    layout_pred = th.cat(layout_pred).numpy()
    downstream_pred = th.cat(downstream_pred).numpy()
    out = pd.DataFrame(
        {
            "downstream_label": labels,
            "downstream_pred": downstream_pred,
            "shape_pred": shape_pred,
            "stroke_pred": stroke_pred,
            "layout_pred": layout_pred,
        }
    )
    for k, v in info.items():
        out[k] = v
    for k, v in flags.namespace_to_dict(FLAGS).items():
        out[k] = v
    out["layout_label"] = out.downstream_label.map(layout_label)
    out["shape_label"] = out.downstream_label.map(shape_label)
    out["stroke_label"] = out.downstream_label.map(stroke_label)
    out["layout_correct"] = (out["layout_pred"] == out["layout_label"]).astype(int)
    out["shape_correct"] = (out["shape_pred"] == out["shape_label"]).astype(int)
    out["stroke_correct"] = (out["stroke_pred"] == out["stroke_label"]).astype(int)
    out["downstream_correct"] = (
        out["downstream_pred"] == out["downstream_label"]
    ).astype(int)

    out["probing_pred"] = out.apply(probing_prediction, 1)
    out["layout_downstream"] = out.apply(layout_downstream, 1)
    out["shape_downstream"] = out.apply(shape_downstream, 1)
    out["stroke_downstream"] = out.apply(stroke_downstream, 1)

    out["probing_pred_correct"] = out.probing_pred == out.downstream_label
    out["layout_downstream_correct"] = out.layout_downstream == out.layout_label
    out["shape_downstream_correct"] = out.shape_downstream == out.shape_label
    out["stroke_downstream_correct"] = out.stroke_downstream == out.stroke_label
    return out


if __name__ == "__main__":
    tick = time.time()
    FLAGS = flags.get_flags().parse_args()
    main(flags.get_flags().parse_args())
    tock = time.time()
    train_id = flags.train_id(FLAGS)
    pd.DataFrame([{"id": train_id, "script": "layers", "seconds": tock - tick}]).to_csv(
        f"times/{train_id}.tsv", index=False, sep="\t"
    )
