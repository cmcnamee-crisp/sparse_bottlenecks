"""Transforms raw images with pre-trained encoders. Run after the datasets have been generated.
"""
import os
import argparse

from collections import defaultdict
from pyexpat import model

import tqdm
import pandas as pd
import torch as th
import numpy as np

from PIL import Image
from pytorch_lightning import seed_everything
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import torchvision

import clip
import flags
import helpers


def main(FLAGS):
    seed_everything(FLAGS.seed)
    BATCH_SIZE = FLAGS.batch_size
    device = "cuda" if th.cuda.is_available() else "cpu"
    datadesc = pd.read_table(f"data/datadesc_{FLAGS.data_path}_{FLAGS.seed}.tsv")

    os.makedirs(f"data/encode/", exist_ok=True)

    if FLAGS.ClipRes:
        CLIP_RES_MODEL = "RN50"
        CLIP_RES_MODEL_STR = "ClipRes"
        os.makedirs(f"data/{FLAGS.data_path}_{CLIP_RES_MODEL}", exist_ok=True)
        clip_res_model, clip_res_preprocess = clip.load(
            CLIP_RES_MODEL, device=device, jit=False
        )
        if FLAGS.layerwise:
            transform_layerwise(
                datadesc,
                f"./data/images/{FLAGS.data_path}",
                f"./data/encode/{FLAGS.data_path}-{CLIP_RES_MODEL_STR}-{FLAGS.seed}",
                clip_res_model.visual,
                clip_res_preprocess,
                device,
                BATCH_SIZE,
                get_activations_fn=get_clip_resnet_activations,
            )
        else:
            transform(
                datadesc,
                f"./data/images/{FLAGS.data_path}",
                f"./data/encode/{FLAGS.data_path}-{CLIP_RES_MODEL_STR}-{FLAGS.seed}",
                clip_res_model,
                clip_res_preprocess,
                device,
                BATCH_SIZE,
            )

    if FLAGS.ClipVit:
        CLIP_VIT_MODEL = "ViT-B/32"
        CLIP_VIT_MODEL_STR = "ClipVit"
        os.makedirs(f"data/{FLAGS.data_path}_{CLIP_VIT_MODEL_STR}", exist_ok=True)
        clip_vit_model, clip_vit_preprocess = clip.load(
            CLIP_VIT_MODEL, device=device, jit=False
        )
        if FLAGS.layerwise:
            transform_layerwise(
                datadesc,
                f"./data/images/{FLAGS.data_path}",
                f"./data/encode/{FLAGS.data_path}-{CLIP_VIT_MODEL_STR}-{FLAGS.seed}",
                clip_vit_model.visual,
                clip_vit_preprocess,
                device,
                BATCH_SIZE,
                get_activations_fn=get_vit_activations,
            )
        else:
            transform(
                datadesc,
                f"./data/images/{FLAGS.data_path}",
                f"./data/encode/{FLAGS.data_path}-{CLIP_VIT_MODEL_STR}-{FLAGS.seed}",
                clip_vit_model,
                clip_vit_preprocess,
                device,
                BATCH_SIZE,
            )

    if FLAGS.OrigRes:
        RES_MODEL = "resnet50"
        RES_MODEL_STR = "OrigRes"
        os.makedirs(f"data/{FLAGS.data_path}_{RES_MODEL}", exist_ok=True)
        if FLAGS.layerwise:
            print(RES_MODEL)
            res_model, res_preprocess = load_resmodel_unwrapped(RES_MODEL, device)
            transform_layerwise(
                datadesc,
                f"./data/images/{FLAGS.data_path}",
                f"./data/encode/{FLAGS.data_path}-{RES_MODEL_STR}-{FLAGS.seed}",
                res_model,
                res_preprocess,
                device,
                BATCH_SIZE,
                get_activations_fn=get_resnet_activations,
            )
        else:
            res_model, res_preprocess = load_resmodel(RES_MODEL, device)
            transform(
                datadesc,
                f"./data/images/{FLAGS.data_path}",
                f"./data/encode/{FLAGS.data_path}-{RES_MODEL_STR}-{FLAGS.seed}",
                res_model,
                res_preprocess,
                device,
                BATCH_SIZE,
            )

    if FLAGS.data_path == "t1colors":
        model_data_paths = ["colors_0.0556", "colors_0.9", "colors_0.99", "colors_1.0"]
    elif FLAGS.data_path == "t1":
        model_data_paths = [FLAGS.model_data_path]
    else:
        model_data_paths = [FLAGS.data_path]

    for model_data_path in model_data_paths:
        if FLAGS.RawRes:
            RES_MODEL_STR = "RawRes"
            if FLAGS.data_path in ["t1colors", "t1"]:
                output_path = f"./data/encode/{FLAGS.data_path}-{model_data_path}-{RES_MODEL_STR}-{FLAGS.seed}"
            else:
                output_path = (
                    f"./data/encode/{FLAGS.data_path}-{RES_MODEL_STR}-{FLAGS.seed}"
                )
            res_model, res_preprocess = load_raw_res(device, model_data_path)
            if FLAGS.layerwise:
                transform_layerwise(
                    datadesc,
                    f"./data/images/{FLAGS.data_path}",
                    output_path,
                    res_model,
                    res_preprocess,
                    device,
                    BATCH_SIZE,
                    get_activations_fn=get_resnet_activations,
                )
            else:
                transform(
                    datadesc,
                    f"./data/images/{FLAGS.data_path}",
                    output_path,
                    res_model,
                    res_preprocess,
                    device,
                    BATCH_SIZE,
                )

        if FLAGS.pretrain_model == "RawCnn":
            RAWCNN_MODEL_STR = "RawCnn"
            if FLAGS.data_path in ["t1colors", "t1"]:
                output_path = f"./data/encode/{FLAGS.data_path}-{model_data_path}-{RAWCNN_MODEL_STR}-{FLAGS.seed}"
            else:
                output_path = (
                    f"./data/encode/{FLAGS.data_path}-{RAWCNN_MODEL_STR}-{FLAGS.seed}"
                )
            if FLAGS.layerwise:
                res_model, res_preprocess = load_raw_cnn(FLAGS, device, model_data_path)
                transform_layerwise(
                    datadesc,
                    f"./data/images/{FLAGS.data_path}",
                    output_path,
                    res_model,
                    res_preprocess,
                    device,
                    BATCH_SIZE,
                    get_activations_fn=get_cnn_activations,
                )
            else:
                res_model, res_preprocess = load_raw_cnn(FLAGS, device, model_data_path)
                transform(
                    datadesc,
                    f"./data/images/{FLAGS.data_path}",
                    output_path,
                    res_model,
                    res_preprocess,
                    device,
                    BATCH_SIZE,
                )
    # if FLAGS.default:
    #     os.makedirs(f"data/{FLAGS.data_path}_{FLAGS.default_dim}", exist_ok=True)
    #     default_preprocess = Compose(
    #         [
    #             Resize(256),
    #             CenterCrop(224),
    #             Resize(FLAGS.default_dim),
    #             ToTensor(),
    #             stripalpha,
    #             # Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #         ]
    #     )

    #     for filename_pre in tqdm.tqdm(datadesc.filename, desc="collect"):
    #         # if not os.path.exists(filename_res):
    #         flat_encoding = (
    #             default_preprocess(
    #                 Image.open(f"./data/{FLAGS.data_path}/{filename_pre}")
    #             )
    #             .squeeze()
    #             .detach()
    #             .cpu()
    #             .numpy()
    #         )
    #         filename_post = f"./data/{FLAGS.data_path}_{FLAGS.default_dim}/{filename_pre}".replace(
    #             ".png", ".npy"
    #         )
    #         np.save(filename_post, flat_encoding)
    #         # Ensure memory stays A-ok.
    #         del flat_encoding


# Saves one file per image.
# def transform(datadesc, input_path, output_path, model, process, device, BATCH_SIZE):
#     input_names = []
#     output_names = []
#     for filename in tqdm.tqdm(datadesc.filename, desc="collect"):
#         input_names.append(filename)
#         output_names.append(filename.replace(".png", ".npy"))
#     for idx in tqdm.tqdm(range(0, len(input_names), BATCH_SIZE), desc="encode"):
#         batch = []
#         for filename in input_names[idx : idx + BATCH_SIZE]:
#             try:
#                 x = Image.open(f"{input_path}/{filename}")
#             except:
#                 print(f"{input_path}/{filename}")
#                 break
#             inp = process(x).unsqueeze(0).to(device)
#             batch.append(inp)
#         if len(batch) < 1:
#             print("Finished.")
#             continue
#         batch = th.cat(batch)
#         batch_size = len(batch)  # for the last batch, the size may be smaller.
#         with th.no_grad():
#             batched_encodings = model.encode_image(batch)
#         for filename, encoding in zip(
#             output_names[idx : idx + batch_size], batched_encodings.unbind(0)
#         ):
#             flat_encoding = encoding.squeeze().cpu().numpy()
#             np.save(f"{output_path}/{filename}", flat_encoding)
#         # Ensure memory stays A-ok.
#         del batch, batched_encodings
#     del model, process


def transform(
    datadesc, input_path, output_path, model, process, device, BATCH_SIZE,
):
    """Saves a single file instead of many.
    """
    input_names = []
    output_names = []
    for filename in tqdm.tqdm(datadesc.filename, desc="collect"):
        input_names.append(filename)
        output_names.append(filename.replace(".png", ".npy"))
    output = []
    for idx in tqdm.tqdm(range(0, len(input_names), BATCH_SIZE), desc="encode"):
        batch = []
        for filename in input_names[idx : idx + BATCH_SIZE]:
            try:
                x = Image.open(f"{input_path}/{filename}")
            except:
                print(f"{input_path}/{filename}")
                break
            inp = process(x).unsqueeze(0).to(device)
            batch.append(inp)
        if len(batch) < 1:
            print("Finished.")
            continue
        batch = th.cat(batch)
        with th.no_grad():
            batched_encodings = model.encode_image(batch)
        output.append(batched_encodings)
        # Ensure memory stays A-ok.
        del batch

    np.save(
        f"{output_path}.npy", th.cat(output).squeeze().cpu().numpy(),
    )
    del model, process


def transform_layerwise(
    datadesc,
    input_path,
    output_path,
    model_body,
    process,
    device,
    BATCH_SIZE,
    get_activations_fn,
):
    """Saves a single file instead of many.
    """
    input_names = []
    output_names = []
    for filename in tqdm.tqdm(datadesc.filename, desc="collect"):
        input_names.append(filename)
        output_names.append(filename.replace(".png", ".npy"))

    output = defaultdict(list)
    for idx in tqdm.tqdm(range(0, len(input_names), BATCH_SIZE), desc="encode"):
        batch = []
        for filename in input_names[idx : idx + BATCH_SIZE]:
            try:
                x = Image.open(f"{input_path}/{filename}")
            except:
                print(f"{input_path}/{filename}")
                break
            batch.append(process(x).unsqueeze(0).to(device))
        if len(batch) < 1:
            print("Finished.")
            continue
        batch = th.cat(batch)
        with th.no_grad():
            batched_encodings = get_activations_fn(model_body, batch)
        for name, activations in batched_encodings.items():
            output[name].append(activations)
        # Ensure memory stays A-ok.
        del batch
    for layer, activation_list in output.items():
        # If we shuffle the datadesc, we can maintain the original index of the image.
        activations = th.cat(activation_list).squeeze().cpu().numpy()
        np.save(f"{output_path}_{layer}.npy", activations)
    del model_body, process


def layerwise_encode(network, x):
    activations = {}
    for idx, (layer_name, layer) in enumerate(network.named_children()):
        # Encode batch through the models layers, one by one.
        with th.no_grad():
            if idx == 0:
                x = layer(x.half())
            else:
                x = layer(x)

            y = x.squeeze().cpu()
            batch_size = y.shape[0]
            # its possible in the future to use a mixture of experts instead
            # of flattening.
            y = y.reshape(batch_size, -1)
        if y.shape[1] > 10_000:
            activations[layer_name] = y
    return activations


def stripalpha(x):
    return x[:3, :, :]


class PlaceHolder:
    def __init__(self, encoder) -> None:
        self.encoder = encoder

    def encode_image(self, x):
        return self.encoder(x)


def load_resmodel(_model, device):
    assert _model == "resnet50", "50 is hardcoded atm."
    ## RESNET! - resnet50
    # https://colab.research.google.com/github/pytorch/pytorch.github.io/blob/master/assets/hub/pytorch_vision_resnet.ipynb#scrollTo=X8W_aWqFsHQV
    resnet_preprocess = Compose(
        [
            Resize(256),
            CenterCrop(224),
            ToTensor(),
            stripalpha,
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    resnet = torchvision.models.resnet50(pretrained=True)
    resnet.eval()
    resnet = th.nn.Sequential(*list(resnet.children())[:-1])
    resnet.to(device)
    resnet.eval()
    resnet_placeholder = PlaceHolder(resnet)
    return resnet_placeholder, resnet_preprocess


def load_resmodel_unwrapped(_model, device):
    assert _model == "resnet50", "50 is hardcoded atm."
    ## RESNET! - resnet50
    # https://colab.research.google.com/github/pytorch/pytorch.github.io/blob/master/assets/hub/pytorch_vision_resnet.ipynb#scrollTo=X8W_aWqFsHQV
    resnet_preprocess = Compose(
        [
            Resize(256),
            CenterCrop(224),
            ToTensor(),
            stripalpha,
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    resnet = torchvision.models.resnet50(pretrained=True)
    resnet.eval()
    # resnet = th.nn.Sequential(*list(resnet.children())[:-1])
    resnet.to(device)
    resnet.eval()
    return resnet, resnet_preprocess


def load_raw_res(device, data_path):
    # https://colab.research.google.com/github/pytorch/pytorch.github.io/blob/master/assets/hub/pytorch_vision_resnet.ipynb#scrollTo=X8W_aWqFsHQV
    preprocess = Compose(
        [
            Resize(256),
            CenterCrop(224),
            ToTensor(),
            stripalpha,
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    from models import resnet

    model = resnet.ResNetWrapper.load_from_checkpoint(
        checkpoint_path=f"checkpoints/RawRes-{data_path}.ckpt"
    )
    model.eval()
    # resnet = th.nn.Sequential(*list(resnet.children())[:-1])
    model.to(device)
    model.eval()
    return model, preprocess


def load_raw_cnn(FLAGS, device, data_path):
    # https://colab.research.google.com/github/pytorch/pytorch.github.io/blob/master/assets/hub/pytorch_vision_resnet.ipynb#scrollTo=X8W_aWqFsHQV
    preprocess = Compose(
        [
            Resize(256),
            CenterCrop(224),
            ToTensor(),
            stripalpha,
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    from models import cnn

    # Temporarily override data_path to the one where the model was trained
    orig_data_path = FLAGS.data_path
    FLAGS.data_path = data_path
    checkpoint_id = flags.pretrain_id(FLAGS)
    FLAGS.data_path = orig_data_path
    
    model = cnn.CNN.load_from_checkpoint(
        checkpoint_path=f"checkpoints/{checkpoint_id}.ckpt"
    )
    model.eval()
    # resnet = th.nn.Sequential(*list(resnet.children())[:-1])
    model.to(device)
    model.eval()
    return model, preprocess


def get_cnn_activations(self, x):
    """
    Rationale: After trying a few different approaches, for now, to get the activations of each
    layer I rewrite the original forward function of each of the models.
    """
    out = {}
    layers = self.encoder
    x = layers[:2](x)
    out["conv_layer0"] = _format_layer(x)
    x = layers[2:5](x)
    out["layer1"] = _format_layer(x)
    x = layers[5:8](x)
    out["layer2"] = _format_layer(x)
    x = layers[8:11](x)
    out["layer3"] = _format_layer(x)
    x = layers[11:](x)
    out["fc"] = _format_layer(x)
    return out


def get_resnet_activations(self, x):
    out = {}
    x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
    out["conv_layer0"] = _format_layer(x)
    x = self.layer1(x)
    out["layer1"] = _format_layer(x)
    x = self.layer2(x)
    out["layer2"] = _format_layer(x)
    x = self.layer3(x)
    out["layer3"] = _format_layer(x)
    x = self.layer4(x)
    x = self.avgpool(x)
    x = th.flatten(x, 1)
    out["layer4"] = _format_layer(x)
    # x = self.fc(x)
    # out["fc"] = _format_layer(x)
    return out


def get_clip_resnet_activations(self, x):
    out = {}
    x = x.type(self.conv1.weight.dtype)
    x = self.relu(self.bn1(self.conv1(x)))
    out["conv_layer1"] = _format_layer(x)
    x = self.relu(self.bn2(self.conv2(x)))
    out["conv_layer2"] = _format_layer(x)
    x = self.relu(self.bn3(self.conv3(x)))
    out["conv_layer3"] = _format_layer(x)

    x = self.avgpool(x)
    x = self.layer1(x)
    out["layer1"] = _format_layer(x)
    x = self.layer2(x)
    out["layer2"] = _format_layer(x)
    x = self.layer3(x)
    out["layer3"] = _format_layer(x)
    x = self.layer4(x)
    out["layer4"] = _format_layer(x)
    x = self.attnpool(x)
    out["attnpool"] = _format_layer(x)

    return out


def get_vit_activations(self, x):
    out = {}
    x = self.conv1(x.half())  # shape = [*, width, grid, grid]
    x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
    x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
    x = th.cat(
        [
            self.class_embedding.to(x.dtype)
            + th.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
            x,
        ],
        dim=1,
    )  # shape = [*, grid ** 2 + 1, width]
    x = x + self.positional_embedding.to(x.dtype)
    x = self.ln_pre(x)
    out["conv_embedding"] = _format_layer(x)
    x = x.permute(1, 0, 2)
    for name, layer in self.transformer.resblocks.named_children():
        x = layer(x)
        out[f"transformer_{name}"] = _format_layer(x.permute(1, 0, 2))
    x = x.permute(1, 0, 2)  # LND -> NLD
    x = self.ln_post(x[:, 0, :])
    if self.proj is not None:
        x = x @ self.proj
    return out


def _format_layer(x: th.tensor) -> th.tensor:
    """This means across the channels dimension. In the future we could think
    about handling this in a different way.
    """
    if x.ndim > 2:
        x = th.mean(x, dim=1)
        x = th.flatten(x, 1)
    return x.to("cpu")


if __name__ == "__main__":
    parser = flags.get_flags()
    parser.add_argument("--model_data_path", default="default", type=str)
    parser.add_argument("--layerwise", type=bool, default=False)

    parser.add_argument("--ClipRes", default=False, type=bool)
    parser.add_argument("--ClipVit", default=False, type=bool)
    parser.add_argument("--OrigRes", default=False, type=bool)
    parser.add_argument("--RawRes", default=False, type=bool)
    # Note: --RawCnn is now handled by --pretrain_model RawCnn

    parser.add_argument("--default", default=True, type=bool)
    parser.add_argument("--default_dim", default=64, type=int)

    FLAGS = parser.parse_args()

    import time

    tick = time.time()
    main(FLAGS)
    tock = time.time()
    id = f"transform-{FLAGS.data_path}"
    pd.DataFrame([{"id": id, "script": "transform", "seconds": tock - tick}]).to_csv(
        f"times/{id}.tsv", index=False, sep="\t"
    )
