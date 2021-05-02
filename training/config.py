EMBEDDINGS = {
    "bert-base": "bert-base-cased",
    "bert-large": "bert-large-cased",
    "roberta-base": "xlm-roberta-base",
    "roberta-large": "xlm-roberta-large"
}

TOP = {
    "linear": "linear",
    "cos_sim": "cosine_similarity"
}

def get_config(model_description="roberta-large-cos_sim-relu-only_wic"):
    model_description = model_description.split("-")

    model_config = {"embeddings": EMBEDDINGS["-".join(model_description[:2])], "top": TOP[model_description[2]]}

    feature = model_description[3]

    if feature == "cls":
        model_config["use_cls"] = True
    elif feature == "no_cls":
        model_config["use_cls"] = False
    else:
        model_config["activation"] = feature

    if len(model_description) == 4:
        model_config["only_wic"] = False
        model_config["use_default_datasets"] = False
        return model_config

    dataset_feature = model_description[4]

    if dataset_feature == "only_wic":
        model_config["only_wic"] = True
        model_config["use_default_datasets"] = False
    elif dataset_feature == "use_default_datasets":
        model_config["only_wic"] = False
        model_config["use_default_datasets"] = True

    return model_config
