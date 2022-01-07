# -*- coding: utf-8 -*-
#
# Copyright (c) 2015 Cisco Systems, Inc. and others.  All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This module consists of encoders that serve as input to pytorch modules
"""
import json
import logging
import os
from abc import abstractmethod, ABC
from itertools import chain
from typing import Dict, List, Union, Any, Tuple

from .helpers import BatchData, TokenizerType
from .._util import _get_module_or_attr
from ..containers import HuggingfaceTransformersContainer

try:
    import torch
except ImportError:
    pass

try:
    from tokenizers import normalizers
    from tokenizers.trainers import Trainer
    from tokenizers.normalizers import Lowercase, NFD, StripAccents
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.models import BPE, WordPiece
    from tokenizers.trainers import BpeTrainer, WordPieceTrainer
    from tokenizers.processors import TemplateProcessing

    NO_TOKENIZERS_MODULE = False
except ImportError:
    NO_TOKENIZERS_MODULE = True
    pass

logger = logging.getLogger(__name__)


class AbstractEncoder(ABC):
    """
    Defines a stateful tokenizer. Unlike the tokenizer in the text_preperation_pipeline, tokenizers
    derived from this abstract class have a state such a vocabulary or a trained/pretrained model
    that is used for encoding an input textual string into sequence of ids or a sequence of
    embeddings. These outputs are used by the initial layers of neural nets.
    """

    def __init__(self, **_kwargs):
        ununsed_kwargs = ','.join([f'{k}:{v}' for k, v in _kwargs.items()])
        msg = f"The following keyword arguments are not used while initializing " \
              f"{self.__class__.__name__}: {ununsed_kwargs}"
        logger.debug(msg)

    @abstractmethod
    def prepare(self, examples: List[str]):
        """
        Method that fits the tokenizer and creates a state that can be dumped or used for encoding

        Args:
            examples: List of text strings that will be used for creating the state of the tokenizer
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def dump(self, path: str):
        """
        Method that dumps the state (if any) of the tokenizer

        Args:
            path: The folder where the state has to be dumped
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def load(self, path: str):
        """
        Method that dumps the state (if any) of the tokenizer

        Args:
            path: The folder where the dumped state can be found. Not all tokenizers dump with same
                file names, hence we use a folder name rather than filename.
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def _tokenize(self, text: str) -> List[str]:
        """
        Method that converts a peice of text into a sequence of strings

        Args:
            text (str): Input text.

        Returns:
            tokens (List[str]): List of tokens.
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def batch_encode(
        self, examples: List[str], padding_length: int = None, add_terminals: bool = False, **kwargs
    ) -> BatchData:
        """
        Method that encodes a list of texts into a list of sequence of ids

        Args:
            examples: List of text strings that will be encoded as a batch
            padding_length: The maximum length of each encoded input. Sequences less than this
                length are padded to padding_length, longer sequences are trimmed. If not specified,
                the max length of examples upon tokenization is used as padding_length.
            add_terminals: A boolean flag that determines if terminal special tokens are to be added
                to the tokenized examples or not.

        Returns:
            BatchData: A dictionary consisting of tensor encodings, lengths of inputs and
                special tokens used while encoding, etc. (useful for batching labels in case of
                token classification). Typically, it includes the following keys
                (description follows):
                - seq_lengths: Number of tokens in each example before adding padding tokens. It
                    also includes terminal tokens as well, if they are added. If using an encoder
                    that splits words in sub-words, then seq_lengths implies number of words instead
                    of number of sub-words, along with any added terminal tokens. This number is
                    useful in case of token classifiers which require token-level (aka.
                    word-level) outputs as well as in sequence classifiers models such as LSTM.
                - split_lengths: The length of each subgroup (i.e. group of sub-words) in each
                    example. Due to its definition, it obviously does not include any terminal
                    tokens in its counts. This can be seen as a fine-grained information to
                    seq_lengths values for the encoders with sub-word tokenization. This is again
                    useful in cases of token classifiers to flexibly choose between representations
                    of first sub-word or mean/max pool of sub-words' representations in order to
                    obtain the word-level representations. For lookup table based encoders where
                    words are not broken into sub-words, `split_lengths` is simply a sequence of
                    ones whose sum indicates the number of words w/o terminal & padding tokens.
                - seq_ids (in case non-pretrained models that require training an embedding layer):
                    The encoded ids useful for embedding lookup, including terminal special tokens
                    if asked for, and with padding.
                - attention_masks (only in case of huggingface trainable encoders): Boolean flags
                    corresponding to each id in seq_ids, set to 0 if padding token else 1.
                - hgf_encodings (only in huggingface pretrained encoders): A dict of outputs from a
                    Pretrained Language Model encoder from Huggingface (shortly dubbed as hgf).
                - char_seq_ids (only in dual tokenizers): Similar to seq_ids but from a char
                    tokenizer in case of dual tokenization
                - char_seq_lengths (only in dual tokenizers): Similar to seq_lengths but from a char
                    tokenizer in case of dual tokenization. Like seq_lengths, this also includes
                    terminal special tokens from char vocab in the length count whenever added.

        Special note on `add_terminals` when using for sequence classification:
            This flag can be True or False in general. Setting it to False will lead to errors in
            case of Huggingface tokenizers as they are generally built to include terminals along
            with pad tokens. Hence, the default value for `add_terminals` is False in case of
            encoders built on top of AbstractVocabLookupEncoder and True for Hugginface ones. This
            value can be True or False for encoders based on AbstractVocabLookupEncoder for sequence
            classification.
        Special note on `add_terminals` when using for token classification:
            When using a CRF layer, `add_terminals` must be set to False and hence Huggingface
            encoders cannot be used in such scenarios. If not using a CRF layer, one can use any
            encoders by setting it to True but care must be taken to modify labels accordingly.
            Hence, it is advisable to use `split_lengths` for padding labels in case of token
            classification instead of `seq_lengths`
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def get_vocab(self) -> Dict:
        """Returns a dictionary of vocab tokens as keys and their ids as values"""
        raise NotImplementedError("Subclasses must implement this method")

    def get_pad_token_idx(self) -> Union[None, int]:
        """
        If there exists a padding token's index in the vocab, it is returned; useful while
        initializing an embedding layer. Else returns a None.
        """
        if not hasattr(self, "pad_token_idx"):
            return None
        return getattr(self, "pad_token_idx")


class AbstractVocabLookupEncoder(AbstractEncoder):
    """
    Abstract class wrapped around AbstractEncoder that has a vocabulary lookup as the state.
    """

    SPECIAL_TOKENS_DICT = {
        "pad_token": "<PAD>",
        "unk_token": "<UNK>",
        "start_token": "<START>",
        "end_token": "<END>",
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.token2id = {}

    @property
    def id2token(self):
        return {i: t for t, i in self.token2id.items()}

    def prepare(self, examples: List[str]):
        examples = [ex.strip() for ex in examples]
        all_tokens = dict.fromkeys(chain.from_iterable([self._tokenize(text) for text in examples]))
        self.token2id = {t: i for i, t in enumerate(all_tokens)}

        for name, token in self.__class__.SPECIAL_TOKENS_DICT.items():
            self.token2id.update({token: len(self.token2id)})
            setattr(self, f"{name}", token)
            setattr(self, f"{name}_idx", self.token2id[token])

    def dump(self, path: str):
        filename = self._get_filename(path)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w") as opfile:
            for token in self.token2id:
                opfile.write(f"{token}\n")
        with open(os.path.join(path, "special_vocab.json"), "w") as fp:
            json.dump(self.__class__.SPECIAL_TOKENS_DICT, fp, indent=4)
        msg = f"The state of {self.__class__.__name__} is successfully dumped at '{filename}'"
        logger.info(msg)

    def load(self, path: str):
        filename = self._get_filename(path)
        with open(filename, "r") as opfile:
            tokens = [line.strip() for line in opfile]
            self.token2id = {t: i for i, t in enumerate(tokens)}
        with open(os.path.join(path, "special_vocab.json"), "r") as fp:
            special_tokens_dict = json.load(fp)
            for name, token in special_tokens_dict.items():
                setattr(self, f"{name}", token)
                setattr(self, f"{name}_idx", self.token2id[token])

        msg = f"The state of {self.__class__.__name__} is successfully loaded from '{filename}'"
        logger.info(msg)

    def _get_filename(self, path: str):
        if not os.path.isdir(path):
            msg = f"The dump method of {self.__class__.__name__} only accepts diretory as the " \
                  f"path argument."
            logger.error(msg)
            raise ValueError(msg)
        filename = os.path.join(path, "vocab.txt")
        return filename

    @abstractmethod
    def _tokenize(self, text: str) -> List[str]:
        raise NotImplementedError("Subclasses must implement this method")

    def batch_encode(
        self, examples: List[str], padding_length: int = None,
        add_terminals: bool = False, _return_tokenized_examples: bool = False, **kwargs
    ) -> BatchData:

        # convert to tokens and obtain sequence lengths
        tokenized_examples = [self._tokenize(text) for text in examples]
        sequence_lengths = [len(seq) for seq in tokenized_examples]

        # obtain padding length
        # if padding_length is None, it is computed as max(length of all seqs from inputted text)
        # account for start and end tokens respectively if add_terminals is True
        curr_max = max(sequence_lengths) + 2 if add_terminals else max(sequence_lengths)
        padding_length = min(padding_length, curr_max) if padding_length else curr_max

        # batchify by truncating or padding for each example
        seq_ids, sequence_lengths, list_of_list_of_tokens = zip(*[
            self._encode_text(example, padding_length, add_terminals)
            for example in tokenized_examples
        ])

        return BatchData(**{
            "seq_lengths": torch.as_tensor(sequence_lengths, dtype=torch.long),
            "split_lengths": [[1] * len(list_of_t) for list_of_t in list_of_list_of_tokens],
            "seq_ids": torch.as_tensor(seq_ids, dtype=torch.long),
            **(
                {"_seq_tokens": [" ".join(list_of_t) for list_of_t in list_of_list_of_tokens]}
                if _return_tokenized_examples else {}
            ),
        })

    def _encode_text(self, list_of_tokens: List[str], padding_length: int, add_terminals: bool):
        """
        Encodes a list of tokens in to a list of ids based on vocab and special token ids.

        Args:
            list_of_tokens (List[str]): List of words or sub-words that are to be encoded
            padding_length (int): Maximum length of the encoded sequence; sequences shorter than
                this length are padded with a pad index while longer sequences are trimmed
            add_terminals (bool): Whether terminal start and end tokens are to be added or not to
                the encoded sequence

        Returns:
            list_of_ids (List[int]): Sequence of ids corresponding to the input tokens
            seq_length (int): The length of sequence upon encoding (and before any padding)
            list_of_tokens (List[str]): Similar to inputted list_of_tokens but without adding
                special tokens
        """
        list_of_tokens = (
            list_of_tokens[:padding_length - 2] if add_terminals else
            list_of_tokens[:padding_length]
        )
        list_of_tokens_with_terminals = (
            [getattr(self, "start_token")] +
            list_of_tokens +
            [getattr(self, "end_token")]
        ) if add_terminals else list_of_tokens
        list_of_tokens_with_terminals_and_padding = (
            list_of_tokens_with_terminals +
            [getattr(self, "pad_token")] * (padding_length - len(list_of_tokens_with_terminals))
        )
        list_of_ids = [
            self.token2id.get(token, getattr(self, "unk_token_idx"))
            for token in list_of_tokens_with_terminals_and_padding
        ]
        return list_of_ids, len(list_of_tokens_with_terminals), list_of_tokens

    def get_vocab(self) -> Dict:
        return self.token2id


class WhitespaceEncoder(AbstractVocabLookupEncoder):
    """
    Encoder that tokenizes at whitespace. Not useful for languages such as Chinese.
    """

    def _tokenize(self, text: str) -> List[str]:
        return text.strip("\n").split(" ")


class CharEncoder(AbstractVocabLookupEncoder):
    """
    A simple tokenizer that tokenizes at character level
    """

    def _tokenize(self, text: str) -> List[str]:
        return list(text.strip("\n"))


class WhitespaceAndCharDualEncoder(WhitespaceEncoder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.char_token2id = {}

    SPECIAL_CHAR_TOKENS_DICT = {
        "char_pad_token": "<CHAR_PAD>",
        "char_unk_token": "<CHAR_UNK>",
        "char_start_token": "<CHAR_START>",
        "char_end_token": "<CHAR_END>",
    }

    @property
    def char_id2token(self):
        return {i: t for t, i in self.char_token2id.items()}

    @staticmethod
    def char_tokenize(text: str) -> List[str]:
        return list(text.strip("\n"))

    def prepare(self, examples: List[str]):
        super().prepare(examples)

        examples = [ex.strip() for ex in examples]
        all_tokens = dict.fromkeys(
            chain.from_iterable([self.char_tokenize(text) for text in examples])
        )
        self.char_token2id = {t: i for i, t in enumerate(all_tokens)}

        for name, token in self.__class__.SPECIAL_CHAR_TOKENS_DICT.items():
            self.char_token2id.update({token: len(self.char_token2id)})
            setattr(self, f"{name}", token)
            setattr(self, f"{name}_idx", self.char_token2id[token])

    def dump(self, path: str):
        super().dump(path)

        filename = self._get_char_filename(path)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w") as opfile:
            for token in self.char_token2id:
                opfile.write(f"{token}\n")
        with open(os.path.join(path, "special_char_vocab.json"), "w") as fp:
            json.dump(self.__class__.SPECIAL_CHAR_TOKENS_DICT, fp, indent=4)
        msg = f"The state of {self.__class__.__name__} is successfully dumped at '{filename}'"
        logger.info(msg)

    def load(self, path: str):
        super().load(path)

        filename = self._get_char_filename(path)
        with open(filename, "r") as opfile:
            tokens = [line.strip() for line in opfile]
            self.char_token2id = {t: i for i, t in enumerate(tokens)}
        with open(os.path.join(path, "special_char_vocab.json"), "r") as fp:
            special_char_tokens_dict = json.load(fp)
            for name, token in special_char_tokens_dict.items():
                setattr(self, f"{name}", token)
                setattr(self, f"{name}_idx", self.char_token2id[token])

        msg = f"The state of {self.__class__.__name__} is successfully loaded from '{filename}'"
        logger.info(msg)

    def _get_char_filename(self, path: str):
        if not os.path.isdir(path):
            msg = f"The dump method of {self.__class__.__name__} only accepts diretory as the " \
                  f"path argument."
            logger.error(msg)
            raise ValueError(msg)
        filename = os.path.join(path, "char_vocab.txt")
        return filename

    def batch_encode(
        self, examples: List[str], char_padding_length: int = None, char_add_terminals=True,
        add_terminals=False, _return_tokenized_examples: bool = False, **kwargs
    ) -> BatchData:

        if add_terminals:
            msg = f"The argument 'add_terminals' must be False to encode a batch using " \
                  f"{self.__class__.__name__}."
            logger.error(msg)
            raise ValueError(msg)

        batch_data = super().batch_encode(
            examples=examples, add_terminals=False, _return_tokenized_examples=True, **kwargs
        )

        # use tokenized examples to obtain tokens for char tokenization
        seq_tokens = batch_data.pop("_seq_tokens")
        char_seq_ids, char_seq_lengths = [], []
        for _seq_tokens in seq_tokens:
            # compute padding length for character sequences
            _curr_max = max([len(word) for word in _seq_tokens])
            _curr_max = _curr_max + 2 if char_add_terminals else _curr_max
            char_padding_length = (
                min(char_padding_length, _curr_max) if char_padding_length else _curr_max
            )
            _char_seq_ids, _char_seq_lengths = zip(*[
                self._encode_chars(list(word), char_padding_length, char_add_terminals)
                for word in _seq_tokens
            ])
            char_seq_ids.append(_char_seq_ids)
            char_seq_lengths.append(_char_seq_lengths)

        batch_data.update({
            "char_seq_ids": [torch.as_tensor(_ids, dtype=torch.long) for _ids in char_seq_ids],
            "char_seq_lengths": [torch.as_tensor(_ls, dtype=torch.long) for _ls in char_seq_lengths]
        })
        return batch_data

    def _encode_chars(self, list_of_tokens: List[str], padding_length: int, add_terminals: bool):
        """
        Encodes a list of tokens in to a list of character ids based on the encoder's char vocab

        Args:
            list_of_tokens (List[str]): List of chars that are to be encoded
            padding_length (int): Maximum length of the encoded sequence; sequences shorter than
                this length are padded with a pad index while longer sequences are trimmed
            add_terminals (bool): Whether terminal start and end tokens are to be added or not to
                the encoded sequence

        Returns:
            list_of_ids (List[int]): Sequence of ids corresponding to the input chars
            seq_length (int): The length of sequence upon encoding (before padding)
        """
        list_of_chars = (
            list_of_tokens[:padding_length - 2] if add_terminals else
            list_of_tokens[:padding_length]
        )
        list_of_chars_with_terminals = (
            [getattr(self, "char_start_token")] +
            list_of_chars +
            [getattr(self, "char_end_token")]
        ) if add_terminals else list_of_chars
        list_of_chars_with_terminals_and_padding = (
            list_of_chars_with_terminals +
            [getattr(self, "char_pad_token")] * (
                padding_length - len(list_of_chars_with_terminals))
        )
        list_of_ids = [
            self.char_token2id.get(token, getattr(self, "char_unk_token_idx"))
            for token in list_of_chars_with_terminals_and_padding
        ]
        return list_of_ids, len(list_of_chars_with_terminals)

    def get_char_vocab(self) -> Dict:
        return self.char_token2id

    def get_char_pad_token_idx(self) -> Union[None, int]:
        """
        If there exists a char padding token's index in the vocab, it is returned; useful while
        initializing an embedding layer. Else returns a None.
        """
        if not hasattr(self, "char_pad_token_idx"):
            return None
        return getattr(self, "char_pad_token_idx")


def _trim_list_of_subtokens_groups(
    x: List[List[Any]],
    max_len: int,
    y: List[Any] = None
) -> Union[Tuple[List[Any], List], List[Any]]:
    """
    Given a sequence of sub-tokens groups upon tokenization step, this method identifies the
    first N sub-groups to be used so that the max_len of the sequence is not violated. If a
    sequence of labels are passed-in through the argument y, their list is also trimmed
    corresponding to the list of sub-token groups.

    Args:
        x: List of groups of sub-words, obtained upon whitespace pre-tokenization and word-level
            tokenization
        max_len: The maximum length of ravelled output expected. If given a value greater than the
            number of all sub-words inputted, it is clipped to number of all sub-words.
        y: Labels accompanying each group in x
    """
    max_len = min(max_len, sum([len(_x) for _x in x]))
    curr_len = 0
    if y:
        new_x, new_y = [], []
        for _x, _y in zip(x, y):
            if curr_len >= max_len:
                return new_x, new_y
            if curr_len + len(_x) > max_len:
                new_x.append(_x[:max_len - curr_len])
                new_y.append(_y)
                curr_len = max_len
            elif curr_len + len(_x) <= max_len:
                new_x.append(_x)
                new_y.append(_y)
                curr_len += len(_x)
        return new_x, new_y
    else:
        new_x = []
        for _x in x:
            if curr_len >= max_len:
                return new_x
            if curr_len + len(_x) > max_len:
                new_x.append(_x[:max_len - curr_len])
                curr_len = max_len
            elif curr_len + len(_x) <= max_len:
                new_x.append(_x)
                curr_len += len(_x)
        return new_x


class AbstractHuggingfaceTrainableEncoder(AbstractEncoder):
    """
    Abstract class wrapped around AbstractEncoder that is based on Huggingface's tokenizers library
    for creating state model.

    reference:
    https://huggingface.co/docs/tokenizers/python/latest/pipeline.html
    """

    SPECIAL_TOKENS = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = None
        self.trainer = Trainer

        if NO_TOKENIZERS_MODULE:
            msg = "Must install extra [transformers] by running " \
                  "'pip install mindmeld[transformers]'"
            raise ImportError(msg)

    def prepare(self, examples: List[str]):
        """
        references:
        - Huggingface: tutorials/python/training_from_memory.html @ https://tinyurl.com/6hxrtspa
        - https://huggingface.co/docs/tokenizers/python/latest/index.html
        """

        self._prepare_pipeline()
        trainer = self.trainer(
            # vocab_size=30000,
            special_tokens=self.__class__.SPECIAL_TOKENS
        )
        self.tokenizer.train_from_iterator(examples, trainer=trainer, length=len(examples))

    def _prepare_pipeline(self):
        self.tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])
        self.tokenizer.pre_tokenizer = Whitespace()
        self.tokenizer.post_processor = TemplateProcessing(
            single="[CLS] $A [SEP]",
            pair="[CLS] $A [SEP] $B:1 [SEP]:1",
            special_tokens=[
                ("[CLS]", 1),
                ("[SEP]", 2),
            ],
        )
        self.tokenizer.enable_padding(
            pad_id=self.__class__.SPECIAL_TOKENS.index("[PAD]"),
            pad_token="[PAD]"
        )

    def dump(self, path: str):
        filename = self._get_filename(path)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        self.tokenizer.save(filename)
        msg = f"The state of {self.__class__.__name__} is successfully dumped at '{filename}'"
        logger.info(msg)

    def load(self, path: str):
        filename = self._get_filename(path)
        self.tokenizer = _get_module_or_attr("tokenizers", "Tokenizer").from_file(filename)
        self._prepare_pipeline()
        msg = f"The state of {self.__class__.__name__} is successfully loaded from '{filename}'"
        logger.info(msg)

    def _get_filename(self, path: str):
        if not os.path.isdir(path):
            msg = f"The dump method of {self.__class__.__name__} only accepts diretory as the " \
                  f"path argument."
            logger.error(msg)
            raise ValueError(msg)
        filename = os.path.join(path, "tokenizer.json")
        return filename

    def _tokenize(self, text: str) -> List[str]:
        """
        Example:
        --------
        output = tokenizer.encode("Hello, y'all! How are you 😁 ?")
        print(output.tokens)
        # ["Hello", ",", "y", "'", "all", "!", "How", "are", "you", "[UNK]", "?"]
        """
        output = self.tokenizer.encode(text, add_special_tokens=False)
        # By disabling add_special_tokens, one can expect tokenized outputs without any terminal
        # [CLS] and [SEP] tokens
        return output.tokens

    def batch_encode(
        self, examples: List[str], padding_length: int = None, add_terminals=True, **kwargs
    ) -> BatchData:
        """
        Example:
        --------
        output = tokenizer.encode_batch(["Hello, y'all!", "How are you 😁 ?"])
        print(output[1].tokens)
        # ["[CLS]", "How", "are", "you", "[UNK]", "?", "[SEP]", "[PAD]"]

        NOTE:
        -----
        Passing the argument `padding_length` to set the max length for batch encoding is not
        available yet for Huggingface tokenizers
        """

        if not add_terminals:
            msg = f"The argument 'add_terminals' must be True to encode a batch using " \
                  f"{self.__class__.__name__}."
            logger.error(msg)
            raise ValueError(msg)

        # tokenize each word of each input separately
        # get maximum length of each example, accounting for terminal tokens- cls, sep
        # If padding_length is None, padding has to be done to the max length of the input batch
        tokenized_examples = [
            [self._tokenize(word) for word in example.split(" ")] for example in examples
        ]
        max_curr_len = max([len(sum(t_ex, [])) for t_ex in tokenized_examples]) + 2
        padding_length = min(max_curr_len, padding_length) if padding_length else max_curr_len
        padding_length = padding_length - 2  # -2 to sum to padding_length after adding cls, sep

        sub_grouped_examples = [
            _trim_list_of_subtokens_groups(t_ex, padding_length) for t_ex in
            tokenized_examples
        ]  # List[List[List[str]]], innermost List[str] is a list of sub-words for a given word
        tokenized_examples = [" ".join(sum(ex, [])) for ex in sub_grouped_examples]
        split_lengths = [[len(x) for x in ex] for ex in sub_grouped_examples]

        output = self.tokenizer.encode_batch(
            tokenized_examples, add_special_tokens=True
        )
        seq_ids = [o.ids for o in output]
        attention_masks = [o.attention_mask for o in output]

        return BatchData(**{
            "seq_lengths": torch.as_tensor(  # List[int], number of groups per example
                [len(_split_lengths) + 2 for _split_lengths in split_lengths], dtype=torch.long
            ),
            "split_lengths": split_lengths,  # List[List[int]], len of each subgroup; For each
            # example, sum of its split_lengths will be equal to the sum of attention mask
            # minus terminals.
            "seq_ids": torch.as_tensor(seq_ids, dtype=torch.long),  # List[List[int]]
            "attention_masks": torch.as_tensor(attention_masks, dtype=torch.long),  # List[Lst[int]]
        })

    def get_vocab(self) -> Dict:
        return self.tokenizer.get_vocab()

    def get_pad_token_idx(self) -> int:
        return self.tokenizer.token_to_id("[PAD]")


class BytePairEncodingEncoder(AbstractHuggingfaceTrainableEncoder):
    """
    Encoder that fits a BPE model based on the input examples
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = _get_module_or_attr("tokenizers", "Tokenizer")(BPE())
        self.trainer = BpeTrainer


class WordPieceEncoder(AbstractHuggingfaceTrainableEncoder):
    """
    Encoder that fits a WordPiece model based on the input examples
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = _get_module_or_attr("tokenizers", "Tokenizer")(WordPiece())
        self.trainer = WordPieceTrainer


class HuggingfacePretrainedEncoder(AbstractEncoder):

    def __init__(self, pretrained_model_name_or_path=None, **kwargs):
        super().__init__(**kwargs)
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.config, self.tokenizer = None, None

    def prepare(self, examples: List[str]):
        del examples

        if self.pretrained_model_name_or_path is None:
            msg = f"Need a valid 'pretrained_model_name_or_path' path to fit " \
                  f"{self.__class__.__name__} but found value: {self.pretrained_model_name_or_path}"
            raise ValueError(msg)

        hf_trans = HuggingfaceTransformersContainer(self.pretrained_model_name_or_path)
        self.config = hf_trans.get_transformer_model_config()
        self.tokenizer = hf_trans.get_transformer_model_tokenizer()

    def dump(self, path: str):
        if not os.path.isdir(path):
            msg = f"The dump method of {self.__class__.__name__} only accepts diretory as the " \
                  f"path argument."
            logger.error(msg)
            raise ValueError(msg)
        os.makedirs(path, exist_ok=True)
        self.config.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def load(self, path: str):
        if not os.path.isdir(path):
            msg = f"The dump method of {self.__class__.__name__} only accepts diretory as the " \
                  f"path argument."
            logger.error(msg)
            raise ValueError(msg)
        hf_trans = HuggingfaceTransformersContainer(path)
        self.config = hf_trans.get_transformer_model_config()
        self.tokenizer = hf_trans.get_transformer_model_tokenizer()

    def _tokenize(self, text: str) -> List[str]:
        return self.tokenizer.tokenize(text)

    def batch_encode(
        self, examples: List[str], padding_length: int = None, add_terminals=True, **kwargs
    ) -> BatchData:

        if not add_terminals:
            msg = f"The argument 'add_terminals' must be True to encode a batch using " \
                  f"{self.__class__.__name__}."
            logger.error(msg)
            raise ValueError(msg)

        # tokenize each word of each input seperately
        # get maximum length of each example, accounting for terminal tokens- cls, sep
        # If padding_length is None, padding has to be done to the max length of the input batch
        tokenized_examples = [
            [self._tokenize(word) for word in example.split(" ")]
            for example in examples
        ]
        max_curr_len = max([len(sum(t_ex, [])) for t_ex in tokenized_examples]) + 2
        padding_length = min(max_curr_len, padding_length) if padding_length else max_curr_len
        padding_length = padding_length - 2  # -2 to sum to padding_length after adding cls, sep

        sub_grouped_examples = [
            _trim_list_of_subtokens_groups(t_ex, padding_length) for t_ex in
            tokenized_examples
        ]
        tokenized_examples = [" ".join(sum(ex, [])) for ex in sub_grouped_examples]
        split_lengths = [[len(x) for x in ex] for ex in sub_grouped_examples]

        hgf_encodings = self.tokenizer(
            tokenized_examples, padding=True, truncation=True, max_length=None,
            return_tensors="pt"
        )  # Huggingface returns this as BatchEncodings and needs to be converted to a dictionary

        return BatchData(**{
            "seq_lengths": torch.as_tensor(  # List[int], number of groups per example
                [len(_split_lengths) + 2 for _split_lengths in split_lengths], dtype=torch.long
            ),
            "split_lengths": split_lengths,  # List[List[int]], len of each subgroup; For each
            # example, sum of its split_lengths will be equal to the sum of attention mask
            # minus terminals.
            "hgf_encodings": {**hgf_encodings},
        })

    def get_vocab(self) -> Dict:
        return self.tokenizer.get_vocab()

    def get_pad_token_idx(self) -> int:
        return self.tokenizer.pad_token_id


class InputEncoderFactory:
    TOKENIZER_NAME_TO_CLASS = {
        TokenizerType.WHITESPACE_TOKENIZER: WhitespaceEncoder,
        TokenizerType.CHAR_TOKENIZER: CharEncoder,
        TokenizerType.WHITESPACE_AND_CHAR_DUAL_TOKENIZER: WhitespaceAndCharDualEncoder,
        TokenizerType.BPE_TOKENIZER: BytePairEncodingEncoder,
        TokenizerType.WORDPIECE_TOKENIZER: WordPieceEncoder,
        TokenizerType.HUGGINGFACE_PRETRAINED_TOKENIZER: HuggingfacePretrainedEncoder,
    }

    @classmethod
    def get_encoder_cls(cls, tokenizer_type: str):
        try:
            return InputEncoderFactory.TOKENIZER_NAME_TO_CLASS[TokenizerType(tokenizer_type)]
        except ValueError as e:
            msg = f"Expected tokenizer_type amongst " \
                  f"{[v.value for v in TokenizerType.__members__.values()]} " \
                  f"but found '{tokenizer_type}'. Cannot create an input encoder."
            logger.error(msg)
            raise ValueError(msg) from e
