import config
import dataset
import os
import engine
import torch
import utils
import params
import datetime
import random
import pandas as pd
import torch.nn as nn
import numpy as np
from torch.optim import lr_scheduler
from model import TweetModel
from sklearn import model_selection
from sklearn import metrics
import transformers
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from functools import partial
from apex import amp


def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


def run(fold):

    seed_everything(100)

    dfx = pd.read_csv(config.TRAINING_FILE)

    df_train = dfx[dfx.kfold != fold].reset_index(drop=True)
    df_valid = dfx[dfx.kfold == fold].reset_index(drop=True)
    print(df_train.shape)
    print(df_valid.shape)

    device = torch.device("cuda")

    train_dataset = dataset.TweetDataset(
        tweet=df_train.text.values,
        sentiment=df_train.sentiment.values,
        selected_text=df_train.selected_text.values,
    )

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.TRAIN_BATCH_SIZE, num_workers=4
    )

    valid_dataset = dataset.TweetDataset(
        tweet=df_valid.text.values,
        sentiment=df_valid.sentiment.values,
        selected_text=df_valid.selected_text.values,
    )

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=config.VALID_BATCH_SIZE, num_workers=2
    )

    model_config = transformers.RobertaConfig.from_pretrained(config.MODEL_DIR)
    model_config.output_hidden_states = True
    model = TweetModel(conf=model_config)
    model.to(device)

    num_train_steps = int(len(df_train) / config.TRAIN_BATCH_SIZE * config.EPOCHS)
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.001,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_parameters, lr=config.LEARNING_RATE)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
    )
    # model, optimizer = amp.initialize(model, optimizer, opt_level="O1", verbosity=0)

    es = utils.EarlyStopping(patience=2, mode="max")
    model_date = datetime.datetime.today().strftime("%m%d")
    print(f"Training is Starting for fold={fold}")
    best_jac = 0
    for epoch in range(config.EPOCHS):
        engine.train_fn(
            train_data_loader, model, optimizer, device, scheduler=scheduler
        )
        jaccard = engine.eval_fn(valid_data_loader, model, device)
        print(f"Jaccard Score = {jaccard}")
        if jaccard > best_jac:
            print("🔥")
            es(
                jaccard,
                model,
                model_path=f"{config.model_type}_{fold}_{model_date}_{jaccard:0.4f}.bin",
            )
            jaccard = best_jac
        if es.early_stop:
            print("Early stopping")
            break


if __name__ == "__main__":
    FOLD = 0
    run(fold=FOLD)
