"""Writes tikz plot commands.
"""
import pandas as pd


def main():
    fig_t1()
    fig_t2()
    fig_t3()
    fig_t4()
    fig_layers()


MODEL_ORDER = ["ClipVit", "ClipRes", "OrigRes", "RawRes", "RawCnn"]
MODEL_LABELS = ["\\Me", "\\Md", "\\Mc", "\\Mb", "\\Ma"]
CONCEPT_ORDER = ["layout", "shape", "stroke"]


def fig_layers():
    df = pd.read_table("./figures/tbls/layers.tsv.zip")
    order = ["layout", "shape", "stroke", "nmi", "downstream", "probing"]

    with open("./figures/tikz/layers.tex", "w") as f:
        f.write(
            """
\\begin{tikzpicture}
\\begin{groupplot}[group    style={columns=5},
    view={0}{90},
    width=2.25cm,
    height=2.25cm,
    scale only axis,
axis y line*=left, % only one y axis on the left, * switches off the arrow tip
axis x line*=bottom,
    group style={
    group size=5 by 1,
    horizontal sep=5pt,
    vertical sep=5pt,
    x descriptions at=edge bottom,
    y descriptions at=edge left},
    ymin=0, ymax=1.1,
legend,
    legend style={at={(1.1,0)},anchor=south west,font=\\footnotesize},
        legend cell align={left},
    ]        
"""
        )

        for j, pretrain_model in enumerate(MODEL_ORDER):
            layers = get_layers_short(pretrain_model)
            rl = get_layers_readable(pretrain_model)
            layers = layers[: len(rl)]

            ticks = ",".join([str(i + 1) for i in range(len(layers))])
            ticklabels = ",".join(rl)

            if j == 0:
                f.write(
                    f"""
        \\nextgroupplot[
            ylabel=Accuracy,
            title={MODEL_LABELS[j]},
            xticklabel style = {{rotate=90}},
        xmin=.5,xmax={len(layers)+.5},
        xtick={{ {ticks} }},
                    xticklabels={{ {ticklabels} }}]"""
                )
            else:
                f.write(
                    f"""
        \\nextgroupplot[
            title={MODEL_LABELS[j]},
            xticklabel style = {{rotate=90}},
        xmin=.5,xmax={len(layers)+.5},
        xtick={{ {ticks} }},
                    xticklabels={{ {ticklabels} }}]"""
                )

            t = df.query("pretrain_model == @pretrain_model & holdout_many == False")
            condition_0 = "N - 1 Slice"

            lines = []
            for i, layer in enumerate(layers):
                acc = t.query(f"layer == @layer & target == '{order[0]}'").value.mean()
                lines.append(f"({i+1}, {acc:.2f})")
            coords1 = "".join(lines)

            lines = []
            for i, layer in enumerate(layers):
                acc = t.query(f"layer == @layer & target == '{order[1]}'").value.mean()
                lines.append(f"({i+1}, {acc:.2f})")
            coords2 = "".join(lines)

            lines = []
            for i, layer in enumerate(layers):
                acc = t.query(f"layer == @layer & target == '{order[2]}'").value.mean()
                lines.append(f"({i+1}, {acc:.2f})")
            coords3 = "".join(lines)

            lines = []
            for i, layer in enumerate(layers):
                acc = t.query(f"layer == @layer & target == '{order[3]}'").value.mean()
                lines.append(f"({i+1}, {acc:.2f})")
            coords4 = "".join(lines)

            lines = []
            for i, layer in enumerate(layers):
                acc = t.query(f"layer == @layer & target == '{order[4]}'").value.mean()
                lines.append(f"({i+1}, {acc:.2f})")
            coords5 = "".join(lines)

            lines = []
            for i, layer in enumerate(layers):
                acc = t.query(f"layer == @layer & target == '{order[5]}'").value.mean()
                lines.append(f"({i+1}, {acc:.2f})")
            coords6 = "".join(lines)

            f.write(
                f"""
            \\addplot+[color=C1, mark = triangle*, mark options={{fill=C1}}] coordinates {{ {coords1} }};
            \\addplot+[color=C2, mark = square*, mark options={{fill=C2}}] coordinates {{ {coords2} }};
            \\addplot+[color=C3, mark = diamond*, mark options={{fill=C3}}] coordinates {{ {coords3} }};
            \\addplot+[color=C4, mark = pentagon*, mark options={{fill=C4}}] coordinates {{ {coords4} }};
            \\addplot+[color=C5, mark = *, mark options={{fill=C5}}] coordinates {{ {coords5} }};
            \\addplot+[color=C6,dashed, mark=star, mark options={{fill=C6}}] coordinates {{ {coords6} }};
            """
            )
        f.write(
            """
\\end{groupplot}
\\end{tikzpicture}
"""
        )


def fig_t1():
    df = pd.read_table("./figures/tbls/t1.tsv.zip")
    markers = ["triangle", "square", "diamond", "pentagon", "o", "star"]

    order = ["default", "colors_0.0556", "colors_0.9", "colors_0.99", "colors_1.0"]
    df = df.groupby(["pretrain_model", "data_path"]).mean()[["accuracy"]].reset_index()

    with open("./figures/tikz/t1.tex", "w") as f:
        f.write(
            """
\\begin{figure}[!ht]
    \\centering
    \\begin{tikzpicture}
        \\begin{axis}[
            xtick={1,2,3,4,5},
            xticklabels={default, $\\textrm{RAND}$, $90\%$, $99\%$, $100\%$},
            legend columns=3,
            legend style={at={(1,-0.3)},anchor=north east},
            width=\\columnwidth,
            height=1.5in,
            ylabel=Accuracy
        ]
        """
        )
        for j, model in enumerate(MODEL_ORDER):
            out = []
            for i, data_path in enumerate(order):
                acc = df.query(
                    "data_path == @data_path & pretrain_model == @model"
                ).accuracy.iloc[0]
                out.append(f"({i+1}, {acc:.2f})")
            coords = " ".join(out)
            f.write(
                f"""
    \\addplot+[color=C{j+1}, mark = {markers[j]}, mark options={{fill=C{j+1}}}] coordinates {{ {coords} }};
"""
            )

        f.write(
            """
    \\addplot+[color=C6,dashed, mark=star, mark options={fill=C6}] coordinates { (1, 0.06) (2, 0.06) (3, 0.06) (4, 0.06) (5, 0.06) };
        """
        )

        f.write(
            """
\\legend{\\Me, \\Md, \\Mc, \\Mb, \\Ma, chance};
\\end{axis}
\\end{tikzpicture}
\\end{figure}
"""
        )


def fig_t2():
    df = pd.read_table("./figures/tbls/t2.tsv.zip")
    with open("./figures/tikz/t2.tex", "w") as f:
        f.write(
            """
\\begin{tikzpicture}
\\begin{groupplot}[group    
    style={columns=5},
    view={0}{90},
    width=2.75cm,
    /pgf/bar width=.3cm,
    height=1.5cm,
    scale only axis,
    xtick={1,2,3},
    axis y line*=left, % only one y axis on the left, * switches off the arrow tip
    axis x line*=bottom,
    group style={
    group size=5 by 1,
    horizontal sep=5pt,
    vertical sep=5pt,
    x descriptions at=edge bottom,
    y descriptions at=edge left},
    xticklabels={Layout, Shape,{Stroke}},
    xmin=.5,xmax=3.5,
    ymin=0, ymax=1.1,
    ybar legend,
    legend pos=south west,
    legend style={font=\\footnotesize, at={(0,0)}},
    legend cell align={left},
    xlabel={Tested Concept}
]        
"""
        )

        for j, pretrain_model in enumerate(MODEL_ORDER):
            if j == 0:
                f.write(
                    f"""
\\nextgroupplot[ylabel=Accuracy,title={MODEL_LABELS[j]}]
"""
                )
            else:
                f.write(
                    f"""
\\nextgroupplot[title={MODEL_LABELS[j]}]
"""
                )

            t = df.query("pretrain_model == @pretrain_model")
            condition_0 = "N - 1 Slice"
            lines = []
            for i, intervene in enumerate(CONCEPT_ORDER):
                acc = t.query(
                    "dataspec == @intervene & holdout_condition == @condition_0"
                ).accuracy.mean()
                lines.append(f"({i+1}, {acc:.2f})")
            coords = "".join(lines)
            f.write(
                f"""
    \\addplot[ybar, color=C1, fill=C1, bar shift=-0.15cm, fill opacity=0.7, pattern=crosshatch, pattern color=C1] coordinates {{ {coords} }};"
"""
            )

            lines = []
            condition_1 = "1 Slice"
            for i, intervene in enumerate(CONCEPT_ORDER):
                acc = t.query(
                    "dataspec == @intervene  & holdout_condition == @condition_1"
                ).accuracy.mean()
                lines.append(f"({i+1}, {acc:.2f})")
            coords = "".join(lines)
            f.write(
                f"""
    \\addplot[ybar, color=C2, fill=C2, bar shift=0.15cm, fill opacity=0.7, pattern=north east lines, pattern color=C2] coordinates {{ {coords} }};
"""
            )

            f.write(
                """
    \draw [->, very thick] (axis cs:1,0.75) -- (axis cs:1,1.0);
    \draw [-, very thick] (axis cs:.7,0.75) -- (axis cs:1.3,0.75);

    \draw [->, very thick] (axis cs:2,0.75) -- (axis cs:2,1.0);
    \draw [-, very thick] (axis cs:1.7,0.75) -- (axis cs:2.3,0.75);

    \draw [->, very thick] (axis cs:3,0.75) -- (axis cs:3,1.0);
    \draw [-, very thick] (axis cs:2.7,0.75) -- (axis cs:3.3,0.75);
                """
            )
            if j == 0:
                f.write(
                    """
    \\legend{N - 1 Slices, 1 Slice}
"""
                )
        f.write(
            """
\\end{groupplot}
\\end{tikzpicture}     
"""
        )


def fig_t3():
    df = pd.read_table("./figures/tbls/t3.tsv.zip")
    with open("./figures/tikz/t3.tex", "w") as f:
        condition_labels = [
            "\\sout{Layout}",
            "\\sout{Shape}",
            "\\sout{Stroke}",
        ]
        condition_labels = [
            "Others",
            "Ablated",
        ]
        for j, condition in enumerate(["others", "ablated"]):
            for i, pretrain_model in enumerate(MODEL_ORDER):
                if i == 0 and j == 0:
                    f.write(
                        f"\\nextgroupplot[ylabel={condition_labels[j]},title={MODEL_LABELS[i]}]"
                    )
                elif j == 0:
                    f.write(f"\\nextgroupplot[title={MODEL_LABELS[i]}]")
                elif i == 0:
                    f.write(f"\\nextgroupplot[ylabel={condition_labels[j]}]")
                else:
                    f.write(f"\\nextgroupplot")
                t = df.query(
                    "pretrain_model == @pretrain_model & condition == @condition"
                )
                condition_0 = "N - 1 Slice"
                lines = []
                for i, intervene in enumerate(CONCEPT_ORDER):
                    acc = t.query(
                        "dataspec_intervene == @intervene & holdout_condition == @condition_0"
                    ).accuracy.mean()
                    lines.append(f"({i+1}, {acc:.2f})")
                coords = "".join(lines)
                f.write(
                    f"\\addplot[ybar, color=C1, fill=C1, bar shift=-0.15cm, fill opacity=0.7, pattern=crosshatch, pattern color=C1] coordinates {{ {coords} }};"
                )

                lines = []
                condition_1 = "1 Slice"
                for i, intervene in enumerate(CONCEPT_ORDER):
                    acc = t.query(
                        "dataspec_intervene == @intervene  & holdout_condition == @condition_1"
                    ).accuracy.mean()
                    lines.append(f"({i+1}, {acc:.2f})")
                coords = "".join(lines)
                f.write(
                    f"\\addplot[ybar, color=C2, fill=C2, bar shift=0.15cm, fill opacity=0.7, pattern=north east lines, pattern color=C2] coordinates {{ {coords} }};"
                )

                if condition == "others":
                    f.write(
                        """
            \draw [->, very thick] (axis cs:1,0.75) -- (axis cs:1,1.0);
            \draw [-, very thick] (axis cs:.7,0.75) -- (axis cs:1.3,0.75);

            \draw [->, very thick] (axis cs:2,0.75) -- (axis cs:2,1.0);
            \draw [-, very thick] (axis cs:1.7,0.75) -- (axis cs:2.3,0.75);

            \draw [->, very thick] (axis cs:3,0.75) -- (axis cs:3,1.0);
            \draw [-, very thick] (axis cs:2.7,0.75) -- (axis cs:3.3,0.75);
                    """
                    )
                elif condition == "ablated":
                    f.write(
                        """
            \draw [->, very thick] (axis cs:1,0.33) -- (axis cs:1,.0);
            \draw [-, very thick] (axis cs:.7,0.33) -- (axis cs:1.3,0.33);

            \draw [->, very thick] (axis cs:2,0.33) -- (axis cs:2,.0);
            \draw [-, very thick] (axis cs:1.7,0.33) -- (axis cs:2.3,0.33);

            \draw [->, very thick] (axis cs:3,0.5) -- (axis cs:3,0.0);
            \draw [-, very thick] (axis cs:2.7,0.5) -- (axis cs:3.3,0.5);
                    """
                    )


def fig_t4():
    df = pd.read_table("./figures/tbls/t4.tsv.zip")
    df["holdout_condition"] = df.holdout_many.map(
        {False: "N - 1 Slice", True: "1 Slice"}
    )

    with open("./figures/tikz/t4.tex", "w") as f:
        df = (
            df.groupby(
                [
                    "pretrain_model",
                    "condition",
                    "holdout_condition",
                    "dataspec_intervene",
                ]
            )
            .accuracy.mean()
            .reset_index()
        )
        condition_labels = [
            "\\sout{Layout}",
            "\\sout{Shape}",
            "\\sout{Stroke}",
        ]
        condition_labels = [
            "Others",
            "Ablated",
        ]

        for j, condition in enumerate(["others", "ablated"]):
            for i, pretrain_model in enumerate(MODEL_ORDER):
                if i == 0 and j == 0:
                    f.write(
                        f"\\nextgroupplot[ylabel={condition_labels[j]},title={MODEL_LABELS[i]}]"
                    )
                elif j == 0:
                    f.write(f"\\nextgroupplot[title={MODEL_LABELS[i]}]")
                elif i == 0:
                    f.write(f"\\nextgroupplot[ylabel={condition_labels[j]}]")
                else:
                    f.write(f"\\nextgroupplot")
                t = df.query(
                    "pretrain_model == @pretrain_model & condition == @condition"
                )
                condition_0 = "N - 1 Slice"
                lines = []
                for i, intervene in enumerate(CONCEPT_ORDER):
                    acc = t.query(
                        "dataspec_intervene == @intervene & holdout_condition == @condition_0"
                    ).accuracy.mean()
                    lines.append(f"({i+1}, {acc:.2f})")
                coords = "".join(lines)
                f.write(
                    f"\\addplot[ybar, color=C1, fill=C1, bar shift=-0.15cm, fill opacity=0.7, pattern=crosshatch, pattern color=C1] coordinates {{ {coords} }};"
                )

                lines = []
                condition_1 = "1 Slice"
                for i, intervene in enumerate(CONCEPT_ORDER):
                    acc = t.query(
                        "dataspec_intervene == @intervene  & holdout_condition == @condition_1"
                    ).accuracy.mean()
                    lines.append(f"({i+1}, {acc:.2f})")
                coords = "".join(lines)
                f.write(
                    f"\\addplot[ybar, color=C2, fill=C2, bar shift=0.15cm, fill opacity=0.7, pattern=north east lines, pattern color=C2] coordinates {{ {coords} }};"
                )

                if condition == "others":
                    f.write(
                        """
            \draw [->, very thick] (axis cs:1,0.75) -- (axis cs:1,1.0);
            \draw [-, very thick] (axis cs:.7,0.75) -- (axis cs:1.3,0.75);

            \draw [->, very thick] (axis cs:2,0.75) -- (axis cs:2,1.0);
            \draw [-, very thick] (axis cs:1.7,0.75) -- (axis cs:2.3,0.75);

            \draw [->, very thick] (axis cs:3,0.75) -- (axis cs:3,1.0);
            \draw [-, very thick] (axis cs:2.7,0.75) -- (axis cs:3.3,0.75);
                    """
                    )
                elif condition == "ablated":
                    f.write(
                        """
            \draw [->, very thick] (axis cs:1,0.33) -- (axis cs:1,.0);
            \draw [-, very thick] (axis cs:.7,0.33) -- (axis cs:1.3,0.33);

            \draw [->, very thick] (axis cs:2,0.33) -- (axis cs:2,.0);
            \draw [-, very thick] (axis cs:1.7,0.33) -- (axis cs:2.3,0.33);

            \draw [->, very thick] (axis cs:3,0.5) -- (axis cs:3,0.0);
            \draw [-, very thick] (axis cs:2.7,0.5) -- (axis cs:3.3,0.5);
                    """
                    )


def get_layers_short(m):
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
        ],
        "OrigRes": ["conv_layer0", "layer1", "layer2", "layer3", "layer4"],
        "RawRes": ["conv_layer0", "layer1", "layer2", "layer3", "layer4"],
        "RawCnn": ["conv_layer0", "layer1", "layer2", "layer3", "fc"],
    }[m]


def get_layers_readable(m):
    return {
        "ClipRes": [
            "conv 1",
            "conv 2",
            "conv 3",
            "layer 1",
            "layer 2",
            "layer 3",
            "layer 4",
            "attnpool",
        ],
        "ClipVit": [
            "conv",
            "layer 1",
            "layer 2",
            "layer 3",
            "layer +",
        ],
        "OrigRes": ["conv", "layer 1", "layer 2", "layer 3", "avgpool"],
        "RawRes": ["conv", "layer 1", "layer 2", "layer 3", "avgpool"],
        "RawCnn": ["conv", "layer 1", "layer 2", "layer 3", "dense"],
    }[m]


if __name__ == "__main__":
    main()
