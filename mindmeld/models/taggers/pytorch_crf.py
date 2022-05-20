import logging
import os
import random
import uuid
from collections import Counter
from copy import copy
from itertools import zip_longest
from random import randint

import numpy as np
import torch
import torch.nn as nn
from sklearn.feature_extraction import DictVectorizer, FeatureHasher
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torchcrf import CRF

from ...exceptions import MindMeldError
from ...path import USER_CONFIG_DIR

logger = logging.getLogger(__name__)

DEFAULT_PYTORCH_CRF_ER_CONFIG = {
    "feat_type": "hash",  # ["hash", "dict"]
    "feat_num": 50000,
    "stratify": True,
    "drop_input": 0.2,
    "train_batch_size": 8,
    "patience": 3,
    "epochs": 100,
    "train_dev_split": 0.15,
    "optimizer_type": "sgd",  # ["sgd", "adam"]
}


class TaggerDataset(Dataset):
    def __init__(self, inputs, seq_lens, labels=None):
        self.inputs = inputs
        self.labels = labels
        self.seq_lens = seq_lens
        self.max_seq_length = max(seq_lens)

    def __len__(self):
        return len(self.seq_lens)

    def __getitem__(self, index):
        mask_list = [1] * self.seq_lens[index] + [0] * (self.max_seq_length - self.seq_lens[index])

        mask = torch.as_tensor(mask_list, dtype=torch.bool)
        if self.labels:
            return self.inputs[index], mask, self.labels[index]

        return self.inputs[index], mask


def custom_coo_cat(tensors):
    assert len(tensors) > 0

    rows = []
    cols = []
    values = []
    sparse_sizes = [0, 0]

    nnz = 0
    for tensor in tensors:
        tensor = tensor.coalesce()
        row, col = tensor.indices()[0], tensor.indices()[1]
        if row is not None:
            rows.append(row + sparse_sizes[0])

        cols.append(col + sparse_sizes[1])

        value = tensor.values()
        if value is not None:
            values.append(value)

        sparse_sizes[0] += tensor.shape[0]
        sparse_sizes[1] += tensor.shape[1]
        nnz += tensor._nnz()

    row = None
    if len(rows) == len(tensors):
        row = torch.cat(rows, dim=0)

    col = torch.cat(cols, dim=0)

    value = None
    if len(values) == len(tensors):
        value = torch.cat(values, dim=0)

    return torch.sparse_coo_tensor(indices=torch.stack([row, col]), values=value, size=sparse_sizes).coalesce()


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.01)


def custom_collate(sequence):
    if len(sequence[0]) == 3:
        sparse_mats, masks, labels = zip(*sequence)
        return custom_coo_cat(sparse_mats), torch.stack(masks), torch.stack(labels)
    elif len(sequence[0]) == 2:
        sparse_mats, masks = zip(*sequence)
        return custom_coo_cat(sparse_mats), torch.stack(masks)


class Encoder:
    def __init__(self, feature_extractor="hash", num_feats=50000):

        self.feat_extractor = DictVectorizer(dtype=np.float32) if feature_extractor == "dict" else FeatureHasher(
            n_features=num_feats, dtype=np.float32)
        self.label_encoder = LabelEncoder()
        self.feat_extract_type = feature_extractor
        self.num_classes = None
        self.classes = None
        self.num_feats = num_feats
        self.fit_done = False

    def get_tensor_data(self, feat_dicts, labels=None, fit=False):
        if labels is None:
            labels = []
        if fit:
            if self.feat_extract_type == "dict":
                comb_dict_list = [x for seq in feat_dicts for x in seq]
                self.feat_extractor.fit(comb_dict_list)
                self.num_feats = len(self.feat_extractor.get_feature_names())
            if labels:
                self.label_encoder.fit([x for l in labels for x in l])
                self.pad_index = len(self.label_encoder.classes_) - 1
                self.classes = self.label_encoder.classes_
                self.num_classes = len(self.label_encoder.classes_)

            self.fit_done = True
        feats = []
        encoded_labels = []
        seq_lens = [len(x) for x in feat_dicts]
        max_seq_len = max(seq_lens)

        for i, (x, y) in enumerate(zip_longest(feat_dicts, labels)):

            padded_x = x + [{}] * (max_seq_len - seq_lens[i])
            sparse_feat = self.feat_extractor.transform(padded_x).tocoo()
            sparse_feat_tensor = torch.sparse_coo_tensor(
                indices=torch.as_tensor(np.stack([sparse_feat.row, sparse_feat.col])),
                values=torch.as_tensor(sparse_feat.data), size=sparse_feat.shape)
            feats.append(sparse_feat_tensor)

            if y:
                transformed_label = self.label_encoder.transform(y)
                transformed_label = np.pad(transformed_label, pad_width=(0, max_seq_len - seq_lens[i]),
                                           constant_values=self.pad_index)
                label_tensor = torch.as_tensor(transformed_label, dtype=torch.long)
                encoded_labels.append(label_tensor)
        return (feats, encoded_labels, seq_lens) if encoded_labels else (feats, seq_lens)


# pylint: disable=too-many-instance-attributes
class TorchCRF(nn.Module):
    def __init__(self):
        super().__init__()
        self.optimizer = None
        self.encoder = None
        self.best_model_save_path = os.path.join(USER_CONFIG_DIR, "tmp", str(uuid.uuid4()), "best_crf_model.pt")
        os.makedirs(os.path.dirname(self.best_model_save_path), exist_ok=True)

    def set_random_states(self):
        torch.manual_seed(self.random_state)
        random.seed(self.random_state + 1)
        np.random.seed(self.random_state + 2)

    def validate_params(self):
        if self.optimizer_type not in ["sgd", "adam"]:
            raise MindMeldError(
                f"Optimizer type {self.optimizer_type} not supported. Supported options are ['sgd', 'adam']")
        elif self.feat_type not in ["hash", "dict"]:
            raise MindMeldError(f"Feature type {self.feat_type} not supported. Supported options are ['hash', 'dict']")
        elif not 0 < self.train_dev_split < 1:
            raise MindMeldError("Train-dev split should be a value between 0 and 1.")
        elif not 0 <= self.drop_input < 1:
            raise MindMeldError("Drop Input should be a value between 0 and 1. (inclusive)")

        for x, y in zip([self.feat_num, self.train_batch_size, self.patience, self.epochs],
                        ["Number of features", "Train Batch size", "Patience", "Number of epochs"]):
            if not isinstance(x, int):
                raise MindMeldError(f"{y} should be am integer value.")

    def build_params(self, num_features, num_classes):
        self.W = nn.Parameter(torch.nn.init.xavier_normal_(torch.empty(size=(num_features, num_classes))),
                              requires_grad=True)
        self.b = nn.Parameter(torch.nn.init.constant_(torch.empty(size=(num_classes,)), val=0.01),
                              requires_grad=True)
        self.crf_layer = CRF(num_classes, batch_first=True)
        self.crf_layer.apply(init_weights)
        self.num_classes = num_classes

    def forward(self, inputs, targets, mask, drop_input=0.0):
        if drop_input:
            dp_mask = (torch.FloatTensor(inputs.values().size()).uniform_() > drop_input)
            inputs.values()[:] = inputs.values() * dp_mask
        dense_W = torch.tile(self.W, dims=(mask.shape[0], 1))
        out_1 = torch.addmm(self.b, inputs, dense_W)
        crf_input = out_1.reshape((mask.shape[0], -1, self.num_classes))
        if targets is None:
            return self.crf_layer.decode(crf_input, mask=mask)
        loss = - self.crf_layer(crf_input, targets, mask=mask)
        return loss

    # The below implementation is borrowed from https://github.com/kmkurn/pytorch-crf/pull/37

    def _compute_log_alpha(self, emissions, mask, run_backwards):
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)

        assert emissions.dim() == 3 and mask.dim() == 2
        assert emissions.size()[:2] == mask.size()
        assert emissions.size(2) == self.crf_layer.num_tags
        assert all(mask[0].data)

        seq_length = emissions.size(0)
        mask = mask.float()
        broadcast_transitions = self.crf_layer.transitions.unsqueeze(0)  # (1, num_tags, num_tags)
        emissions_broadcast = emissions.unsqueeze(2)
        seq_iterator = range(1, seq_length)

        if run_backwards:
            # running backwards, so transpose
            broadcast_transitions = broadcast_transitions.transpose(1, 2)  # (1, num_tags, num_tags)
            emissions_broadcast = emissions_broadcast.transpose(2, 3)

            # the starting probability is end_transitions if running backwards
            log_prob = [self.crf_layer.end_transitions.expand(emissions.size(1), -1)]

            # iterate over the sequence backwards
            seq_iterator = reversed(seq_iterator)
        else:
            # Start transition score and first emission
            log_prob = [emissions[0] + self.crf_layer.start_transitions.view(1, -1)]

        for i in seq_iterator:
            # Broadcast log_prob over all possible next tags
            broadcast_log_prob = log_prob[-1].unsqueeze(2)  # (batch_size, num_tags, 1)
            # Sum current log probability, transition, and emission scores
            score = broadcast_log_prob + broadcast_transitions + emissions_broadcast[
                i]  # (batch_size, num_tags, num_tags)
            # Sum over all possible current tags, but we're in log prob space, so a sum
            # becomes a log-sum-exp
            score = torch.logsumexp(score, dim=1)
            # Set log_prob to the score if this timestep is valid (mask == 1), otherwise
            # copy the prior value
            log_prob.append(score * mask[i].unsqueeze(1) +
                            log_prob[-1] * (1. - mask[i]).unsqueeze(1))

        if run_backwards:
            log_prob.reverse()

        return torch.stack(log_prob)

    def compute_marginal_probabilities(self, inputs, mask):
        # SWITCHING FOR BATCH FIRST DEFAULT
        dense_W = torch.tile(self.W, dims=(mask.shape[0], 1))
        out_1 = torch.addmm(self.b, inputs, dense_W)
        emissions = out_1.reshape((mask.shape[0], -1, self.num_classes))
        emissions = emissions.transpose(0, 1)
        mask = mask.transpose(0, 1)
        alpha = self._compute_log_alpha(emissions, mask, run_backwards=False)
        beta = self._compute_log_alpha(emissions, mask, run_backwards=True)
        z = torch.logsumexp(alpha[alpha.size(0) - 1] + self.crf_layer.end_transitions, dim=1)
        prob = alpha + beta - z.view(1, -1, 1)
        return torch.exp(prob).transpose(0, 1)

    def set_params(self, **params):
        self.feat_type = params.get('feat_type', DEFAULT_PYTORCH_CRF_ER_CONFIG['feat_type']).lower()
        self.feat_num = params.get('feat_num', DEFAULT_PYTORCH_CRF_ER_CONFIG['feat_num'])
        self.stratify = params.get('stratify', DEFAULT_PYTORCH_CRF_ER_CONFIG['stratify'])
        self.drop_input = params.get('drop_input', DEFAULT_PYTORCH_CRF_ER_CONFIG['drop_input'])
        self.train_batch_size = params.get('train_batch_size', DEFAULT_PYTORCH_CRF_ER_CONFIG['train_batch_size'])
        self.patience = params.get('patience', DEFAULT_PYTORCH_CRF_ER_CONFIG['patience'])
        self.epochs = params.get('epochs', DEFAULT_PYTORCH_CRF_ER_CONFIG['epochs'])
        self.train_dev_split = params.get('train_dev_split', DEFAULT_PYTORCH_CRF_ER_CONFIG['train_dev_split'])
        self.optimizer_type = params.get('optimizer_type', DEFAULT_PYTORCH_CRF_ER_CONFIG['optimizer_type']).lower()
        self.random_state = params.get('random_state', randint(1, 10000001))

        self.validate_params()

        logger.debug("Random state for torch-crf is %s", self.random_state)
        if self.feat_type == "dict" and "feat_num" in params:
            logger.warning(
                "WARNING: Number of features is compatible with only `hash` feature type. This value is ignored with `dict` setting", )

    # pylint: disable=too-many-locals
    def fit(self, X, y):
        self.set_random_states()
        self.encoder = Encoder(feature_extractor=self.feat_type, num_feats=self.feat_num)
        stratify_tuples = None
        if self.stratify:
            stratify_tuples = [tuple(sorted(list(set(label)))) for label in y]
            # If we have a label class that is only 1 in number, duplicate it, otherwise train_test_split throws error when using stratify!
            cnt = Counter(stratify_tuples)
            last_one = -1
            while cnt.most_common()[last_one][-1] < 2:
                lone_idx = stratify_tuples.index(cnt.most_common()[last_one][0])
                stratify_tuples.append(cnt.most_common()[last_one][0])
                y.append(copy(y[lone_idx]))
                X.append(copy(X[lone_idx]))
                last_one -= 1
        train_X, dev_X, train_y, dev_y = train_test_split(X, y, test_size=self.train_dev_split,
                                                          stratify=stratify_tuples, random_state=self.random_state)
        # pylint: disable=unbalanced-tuple-unpacking
        train_inputs, encoded_train_labels, train_seq_lens = self.encoder.get_tensor_data(train_X, train_y, fit=True)
        train_dataset = TaggerDataset(train_inputs, train_seq_lens, encoded_train_labels)
        # pylint: disable=unbalanced-tuple-unpacking
        dev_inputs, encoded_dev_labels, dev_seq_lens = self.encoder.get_tensor_data(dev_X, dev_y, fit=False)
        dev_dataset = TaggerDataset(dev_inputs, dev_seq_lens, encoded_dev_labels)

        train_dataloader = DataLoader(train_dataset, batch_size=self.train_batch_size, shuffle=True,
                                      collate_fn=custom_collate)

        dev_dataloader = DataLoader(dev_dataset, batch_size=512, shuffle=True, collate_fn=custom_collate)

        best_dev_score, best_dev_epoch = -np.inf, -1
        _patience_counter = 0

        self.build_params(self.encoder.num_feats, self.encoder.num_classes)
        if self.optimizer_type == "sgd":
            self.optimizer = optim.SGD(self.parameters(), lr=0.01, momentum=0.9, nesterov=True, weight_decay=1e-5)
        elif self.optimizer_type == "adam":
            self.optimizer = optim.Adam(self.parameters(), weight_decay=1e-5)

        for epoch in range(self.epochs):
            if _patience_counter >= self.patience:
                break
            self.train_one_epoch(train_dataloader)
            dev_f1_score = self.run_predictions(dev_dataloader, calc_f1=True)
            logger.debug("Epoch %s finished. Dev F1: %s", epoch, dev_f1_score)

            if dev_f1_score <= best_dev_score:
                _patience_counter += 1
            else:
                _patience_counter = 0
                best_dev_score, best_dev_epoch = dev_f1_score, epoch
                torch.save(self.state_dict(), self.best_model_save_path)
                logger.debug("Model weights saved for best dev epoch %s.", best_dev_epoch)

    def train_one_epoch(self, train_dataloader):
        self.train()
        train_loss = 0
        for batch_idx, (inputs, mask, labels) in enumerate(train_dataloader):
            self.optimizer.zero_grad()
            loss = self.forward(inputs, labels, mask, drop_input=self.drop_input)
            train_loss += loss.item()
            loss.backward()
            self.optimizer.step()
            if batch_idx % 20 == 0:
                logger.debug("Batch: %s Mean Loss: %s", batch_idx,
                             (train_loss / (batch_idx + 1)))

    def run_predictions(self, dataloader, calc_f1=False):
        self.eval()
        predictions = []
        targets = []
        with torch.no_grad():
            for inputs, *mask_and_labels in dataloader:
                if calc_f1:
                    mask, labels = mask_and_labels
                    targets.extend(torch.masked_select(labels, mask).tolist())
                else:
                    mask = mask_and_labels.pop()
                preds = self.forward(inputs, None, mask)
                predictions.extend([x for lst in preds for x in lst] if calc_f1 else preds)
        if calc_f1:
            dev_score = f1_score(targets, predictions, average='weighted')
            return dev_score
        else:
            return predictions

    def predict_marginals(self, X):
        self.load_state_dict(torch.load(self.best_model_save_path))
        inputs, seq_lens = self.encoder.get_tensor_data(X)
        torch_dataset = TaggerDataset(inputs, seq_lens)

        dataloader = DataLoader(torch_dataset, batch_size=512, shuffle=False, collate_fn=custom_collate)
        marginals_dict = []
        self.eval()
        with torch.no_grad():
            for inputs, mask in dataloader:
                probs = self.compute_marginal_probabilities(inputs, mask).tolist()
                mask = mask.tolist()
                # If anyone has any suggestions on a cleaner way to do this, I am all ears!
                marginals_dict.extend([[dict(zip(self.encoder.classes, token_probs)) \
                                        for (token_probs, valid_token) in zip(seq, mask_seq) if valid_token] \
                                       for seq, mask_seq in zip(probs, mask)])

        return marginals_dict

    def predict(self, X):
        self.load_state_dict(torch.load(self.best_model_save_path))
        inputs, seq_lens = self.encoder.get_tensor_data(X)
        torch_dataset = TaggerDataset(inputs, seq_lens)

        dataloader = DataLoader(torch_dataset, batch_size=512, shuffle=False, collate_fn=custom_collate)
        preds = self.run_predictions(dataloader, calc_f1=False)
        return [self.encoder.label_encoder.inverse_transform(x).tolist() for x in preds]
