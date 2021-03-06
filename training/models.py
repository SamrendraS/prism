import torch

from pytorch_lightning import LightningModule
from pytorch_lightning.metrics import Accuracy
from transformers import BertModel, XLMRobertaModel, AdamW, get_linear_schedule_with_warmup
from torch import nn

from prism.util.util import get_tokens_embeddings
from prism.data.constants import TOTAL_STEPS
from prism.training.config import get_config


class GeneralBertClassifier(LightningModule):
    def __init__(self, model_path):
        super(GeneralBertClassifier, self).__init__()

        if model_path == "bert-base-cased" or model_path == "bert-large-cased":
            self.model = BertModel.from_pretrained(model_path)
        elif model_path == "xlm-roberta-large" or model_path == "xlm-roberta-base":
            self.model = XLMRobertaModel.from_pretrained(model_path)
        else:
            raise RuntimeError("Specify correct embeddings: " + model_path)

        self.embedding_dim = self.model.get_input_embeddings().embedding_dim

        self.loss = nn.BCELoss()

        self.save_hyperparameters()

        self.valid_accuracy = Accuracy()
        self.test_accuracy = Accuracy()
        self.predictions_proba = torch.Tensor()

    def _get_embeddings(self, input_ids, attention_mask, word_indices, add_cls):
        sentence_outputs = self.model(input_ids, attention_mask).last_hidden_state
        tokens_embeddings = get_tokens_embeddings(sentence_outputs, word_indices)
        word_embedding = torch.max(tokens_embeddings, 1)[0]

        if not add_cls:
            return word_embedding

        cls_embedding = sentence_outputs[:, 0, :]
        return word_embedding, cls_embedding

    def forward(self, input_ids, attention_mask, word_indices):
        raise RuntimeError("Override me")

    def training_step(self, batch, _):
        inputs, attn, word_indices, labels = batch
        outputs = self(inputs, attn, word_indices)
        return self.loss(outputs, labels)

    def _get_logits(self, outputs):
        raise RuntimeError("Override me")

    def validation_step(self, batch, _):
        inputs, attn, word_indices, labels = batch
        outputs = self(inputs, attn, word_indices)

        logits = self._get_logits(outputs)

        self.valid_accuracy.update(logits, labels.int())
        self.log("val_acc", self.valid_accuracy)

        loss = self.loss(outputs, labels)
        self.log("val_loss", loss, prog_bar=True)

    def validation_epoch_end(self, _):
        self.log("val_acc_epoch", self.valid_accuracy.compute(), prog_bar=True)

    def on_test_epoch_start(self):
        self.predictions_proba = torch.Tensor()

    def test_step(self, batch, _):
        inputs, attn, word_indices, labels = batch
        outputs = self(inputs, attn, word_indices)
        self.predictions_proba = torch.cat((self.predictions_proba, outputs.detach().cpu()))

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=1e-5)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=TOTAL_STEPS)
        return [optimizer], [scheduler]


class CosineSimilarityClassifier(GeneralBertClassifier):
    def __init__(self, model_path, activation, threshold):
        super(CosineSimilarityClassifier, self).__init__(model_path)

        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        else:
            raise RuntimeError("Only relu or sigmoid can be use as an activation")

        self.threshold = threshold
        self.cos = nn.CosineSimilarity(dim=1)

    def _get_logits(self, outputs):
        return (outputs > self.threshold).float()

    def forward(self, input_ids, attention_mask, word_indices):
        first_word_embedding = self._get_embeddings(input_ids[0], attention_mask[0], word_indices[0], add_cls=False)
        second_word_embedding = self._get_embeddings(input_ids[1], attention_mask[1], word_indices[1], add_cls=False)

        outputs = self.cos(first_word_embedding, second_word_embedding)
        outputs = self.activation(outputs)
        return outputs


def get_model():
    model_config = get_config()

    model_path = model_config["embeddings"]
    top = model_config["top"]

    if top == "cosine_similarity":
        BASELINE_THRESHOLD = 0.5195
        activation = model_config["activation"]
        return CosineSimilarityClassifier(model_path, activation, BASELINE_THRESHOLD)