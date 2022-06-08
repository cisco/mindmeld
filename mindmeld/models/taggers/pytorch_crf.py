import logging
import os
import random
import uuid
from collections import Counter
from copy import copy
from itertools import chain
from random import randint
import gc

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


def diag_concat_coo_tensors(tensors):
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


def collate_tensors_and_masks(sequence):
    if len(sequence[0]) == 3:
        sparse_mats, masks, labels = zip(*sequence)
        return diag_concat_coo_tensors(sparse_mats), torch.stack(masks), torch.stack(labels)
    if len(sequence[0]) == 2:
        sparse_mats, masks = zip(*sequence)
        return diag_concat_coo_tensors(sparse_mats), torch.stack(masks)


class Encoder:
    def __init__(self, feature_extractor="hash", num_feats=50000):

        if feature_extractor == "dict":
            self.feat_extractor = DictVectorizer(dtype=np.float32)
        else:
            self.feat_extractor = FeatureHasher(n_features=num_feats, dtype=np.float32)

        self.label_encoder = LabelEncoder()
        self.num_classes = None
        self.classes = None
        self.num_feats = num_feats

    def get_input_tensors(self, feat_dicts, fit=False):
        if fit:
            if isinstance(self.feat_extractor, DictVectorizer):
                comb_dict_list = [x for seq in feat_dicts for x in seq]
                self.feat_extractor.fit(comb_dict_list)
                self.num_feats = len(self.feat_extractor.get_feature_names())

        pass

    def get_padded_transformed_tensors(self, inputs_or_labels, seq_lens, is_label):
        if inputs_or_labels is None:
            return None
        encoded_tensors = []
        max_seq_len = max(seq_lens)

        for i, x in enumerate(inputs_or_labels):
            if not is_label:
                padded_encoded_tensor = self.encode_padded_input(seq_lens[i], max_seq_len, x)
            else:
                padded_encoded_tensor = self.encode_padded_label(seq_lens[i], max_seq_len, x)
            encoded_tensors.append(padded_encoded_tensor)
        return encoded_tensors

    def get_tensor_data(self, feat_dicts, labels=None, fit=False):
        if fit:
            if isinstance(self.feat_extractor, DictVectorizer):
                flattened_feat_dicts = list(chain.from_iterable(feat_dicts))
                self.feat_extractor.fit(flattened_feat_dicts)
                self.num_feats = len(self.feat_extractor.get_feature_names())
            if labels is not None:
                flattened_labels = list(chain.from_iterable(labels))
                self.label_encoder.fit(flattened_labels)
                self.classes, self.num_classes = self.label_encoder.classes_, len(self.label_encoder.classes_)

        seq_lens = [len(x) for x in feat_dicts]

        encoded_tensor_inputs = self.get_padded_transformed_tensors(feat_dicts, seq_lens, is_label=False)
        encoded_tensor_labels = self.get_padded_transformed_tensors(labels, seq_lens, is_label=True)

        return encoded_tensor_inputs, seq_lens, encoded_tensor_labels

    def encode_padded_input(self, current_seq_len, max_seq_len, x):
        padded_x = x + [{}] * (max_seq_len - current_seq_len)
        sparse_feat = self.feat_extractor.transform(padded_x).tocoo()
        sparse_feat_tensor = torch.sparse_coo_tensor(
            indices=torch.as_tensor(np.stack([sparse_feat.row, sparse_feat.col])),
            values=torch.as_tensor(sparse_feat.data), size=sparse_feat.shape)
        return sparse_feat_tensor

    def encode_padded_label(self, current_seq_len, max_seq_len, y):
        transformed_label = self.label_encoder.transform(y)
        transformed_label = np.pad(transformed_label, pad_width=(0, max_seq_len - current_seq_len),
                                   constant_values=(self.num_classes - 1))
        label_tensor = torch.as_tensor(transformed_label, dtype=torch.long)
        return label_tensor


# pylint: disable=too-many-instance-attributes
class TorchCrfModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.optim = None
        self.encoder = None
        self.W = None
        self.b = None
        self.crf_layer = None
        self.num_classes = None

        self.feat_type = None
        self.feat_num = None
        self.stratify_train_val_split = None
        self.drop_input = None
        self.batch_size = None
        self.patience = None
        self.number_of_epochs = None
        self.dev_split_ratio = None
        self.optimizer = None
        self.random_state = None

        self.best_model_save_path = os.path.join(USER_CONFIG_DIR, "tmp", str(uuid.uuid4()), "best_crf_model.pt")
        os.makedirs(os.path.dirname(self.best_model_save_path), exist_ok=True)

    def set_random_states(self):
        torch.manual_seed(self.random_state)
        random.seed(self.random_state + 1)
        np.random.seed(self.random_state + 2)

    def validate_params(self):
        if self.optimizer not in ["sgd", "adam"]:
            raise MindMeldError(
                f"Optimizer type {self.optimizer_type} not supported. Supported options are ['sgd', 'adam']")
        if self.feat_type not in ["hash", "dict"]:
            raise MindMeldError(f"Feature type {self.feat_type} not supported. Supported options are ['hash', 'dict']")
        if not 0 < self.dev_split_ratio < 1:
            raise MindMeldError("Train-dev split should be a value between 0 and 1.")
        if not 0 <= self.drop_input < 1:
            raise MindMeldError("Drop Input should be a value between 0 (inclusive) and 1.")
        if not isinstance(self.patience, int):
            raise MindMeldError("Patience should be an integer value.")
        if not isinstance(self.number_of_epochs, int):
            raise MindMeldError("Number of epochs should be am integer value.")

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

    def set_params(self, feat_type="hash", feat_num=50000, stratify_train_val_split=True, drop_input=0.2, batch_size=8,
                   patience=3, number_of_epochs=100, dev_split_ratio=0.2, optimizer="sgd",
                   random_state=randint(1, 10000001)):

        self.feat_type = feat_type  # ["hash", "dict"]
        self.feat_num = feat_num
        self.stratify_train_val_split = stratify_train_val_split
        self.drop_input = drop_input
        self.batch_size = batch_size
        self.patience = patience
        self.number_of_epochs = number_of_epochs
        self.dev_split_ratio = dev_split_ratio
        self.optimizer = optimizer  # ["sgd", "adam"]
        self.random_state = random_state

        self.validate_params()

        logger.debug("Random state for torch-crf is %s", self.random_state)
        if self.feat_type == "dict":
            logger.warning(
                "WARNING: Number of features is compatible with only `hash` feature type. This value is ignored with `dict` setting")

    def get_dataloader(self, X, y, is_train):
        tensor_inputs, input_seq_lens, tensor_labels = self.encoder.get_tensor_data(X, y, fit=is_train)
        tensor_dataset = TaggerDataset(tensor_inputs, input_seq_lens, tensor_labels)
        torch_dataloader = DataLoader(tensor_dataset, batch_size=self.batch_size if is_train else 512, shuffle=is_train,
                                      collate_fn=collate_tensors_and_masks)
        return torch_dataloader

    def fit(self, X, y):
        self.set_random_states()
        self.encoder = Encoder(feature_extractor=self.feat_type, num_feats=self.feat_num)
        stratify_tuples = None
        if self.stratify_train_val_split:
            stratify_tuples = self.stratify_input(X, y)

        # TODO: Rewrite our own train_test_split function to handle FileBackedList and avoid duplicating unique labels
        train_X, dev_X, train_y, dev_y = train_test_split(X, y, test_size=self.dev_split_ratio,
                                                          stratify=stratify_tuples, random_state=self.random_state)

        train_dataloader = self.get_dataloader(train_X, train_y, is_train=True)
        dev_dataloader = self.get_dataloader(dev_X, dev_y, is_train=False)

        # desperate attempt to save some memory
        del X, y, train_X, train_y, dev_X, dev_y, stratify_tuples
        gc.collect()

        self.build_params(self.encoder.num_feats, self.encoder.num_classes)

        if self.optimizer == "sgd":
            self.optim = optim.SGD(self.parameters(), lr=0.01, momentum=0.9, nesterov=True, weight_decay=1e-5)
        if self.optimizer == "adam":
            self.optim = optim.Adam(self.parameters(), weight_decay=1e-5)

        self.training_loop(train_dataloader, dev_dataloader)

    def training_loop(self, train_dataloader, dev_dataloader):

        best_dev_score, best_dev_epoch = -np.inf, -1
        _patience_counter = 0

        for epoch in range(self.number_of_epochs):
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

    def stratify_input(self, X, y):
        def get_unique_tuple(label):
            return tuple(sorted(list(set(label))))

        stratify_tuples = [get_unique_tuple(label) for label in y]
        # If we have a label class that is only 1 in number, duplicate it, otherwise train_test_split throws error when using stratify!
        cnt = Counter(stratify_tuples)

        for label, count in cnt.most_common()[::-1]:
            if count > 1:
                break
            idx = stratify_tuples.index(label)
            X.append(copy(X[idx]))
            y.append(copy(y[idx]))
            stratify_tuples.append(label)
        return stratify_tuples

    def train_one_epoch(self, train_dataloader):
        self.train()
        train_loss = 0
        for batch_idx, (inputs, mask, labels) in enumerate(train_dataloader):
            self.optim.zero_grad()
            loss = self.forward(inputs, labels, mask, drop_input=self.drop_input)
            train_loss += loss.item()
            loss.backward()
            self.optim.step()
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

        dataloader = self.get_dataloader(X, None, is_train=False)
        marginals_dict = []
        self.eval()
        with torch.no_grad():
            for inputs, mask in dataloader:
                probs = self.compute_marginal_probabilities(inputs, mask).tolist()
                mask = mask.tolist()

                # This is basically to create a nested list-dict structure in which we have the probability values
                # for each token for each sequence.
                for seq, mask_seq in zip(probs, mask):
                    one_seq_list = []
                    for (token_probs, valid_token) in zip(seq, mask_seq):
                        if valid_token:
                            one_seq_list.append(dict(zip(self.encoder.classes, token_probs)))
                    marginals_dict.append(one_seq_list)

        return marginals_dict

    def predict(self, X):
        self.load_state_dict(torch.load(self.best_model_save_path))
        dataloader = self.get_dataloader(X, None, is_train=False)

        preds = self.run_predictions(dataloader, calc_f1=False)
        return [self.encoder.label_encoder.inverse_transform(x).tolist() for x in preds]
