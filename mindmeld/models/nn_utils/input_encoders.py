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

from .helpers import BatchData, TokenizerType, ClassificationType
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

    def __init__(self, **kwargs):
        if "classification_type" not in kwargs:
            msg = "The key 'classification_type' is required to initialize an Encoder class."
            raise ValueError(msg)
        self.classification_type = ClassificationType(kwargs["classification_type"])

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
            BatchData: A dictionary-like object for the supplied batch of data, consisting of
                various tensor inputs to the neural computation graph as well as any other inputs
                required during the forward computation.

        Special note on `add_terminals` when using for sequence classification:
            This flag can be True or False in general. Setting it to False will lead to errors in
            case of Huggingface tokenizers as they are generally built to include terminals along
            with pad tokens. Hence, the default value for `add_terminals` is False in case of
            encoders built on top of AbstractVocabLookupEncoder and True for Hugginface ones. This
            value can be True or False for encoders based on AbstractVocabLookupEncoder for sequence
            classification.
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

    @property
    def number_of_terminal_tokens(self) -> int:
        """
        Returns the (maximum) number of terminal tokens used by the encoder during
        batch encoding when add_terminals is set to True.
        """
        raise NotImplementedError


def _trim_a_list_of_sub_token_groups(
    x: List[List[Any]],
    max_len: int,
    y: List[Any] = None
) -> Union[Tuple[List[Any], List], List[Any]]:
    """
    Given a list of sub-tokens sequences (aka. groups) upon a tokenization step, this method
    identifies the first N groups that can be consumed within the allowed max length max_len.

    Args:
        x: List of groups of sub-words, obtained upon whitespace pre-tokenization and word-level
            tokenization using a huggingface tokenizer
        max_len: The maximum length of ravelled output expected. If given a value greater than the
            number of all sub-words inputted, it is clipped to number of all sub-words.
        y: Labels accompanying each group in x
    """
    max_len = min(max_len, sum([len(_x) for _x in x]))
    curr_len = 0
    if y:
        new_x, new_y = [], []
        # iter through each sub-tokens group w/ respective
        # group's label (eg. group: ["m", "##ug"])
        for _x, _y in zip(x, y):
            if curr_len + len(_x) > max_len:
                return new_x, new_y
            elif curr_len + len(_x) <= max_len:
                new_x.append(_x)
                new_y.append(_y)
                curr_len += len(_x)
        return new_x, new_y
    else:
        new_x = []
        # iter through each sub-tokens group w/o respective
        # group's label (eg. group: ["m", "##ug"])
        for _x in x:
            if curr_len + len(_x) > max_len:
                return new_x
            elif curr_len + len(_x) <= max_len:
                new_x.append(_x)
                curr_len += len(_x)
        return new_x


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
        self, examples: List[str], padding_length: int = None, add_terminals: bool = False,
        _return_tokenized_examples: bool = False, **kwargs
    ) -> BatchData:

        n_terminals = self.number_of_terminal_tokens if add_terminals else 0

        if self.classification_type == ClassificationType.TEXT:

            tokenized_examples = [self._tokenize(example) for example in examples]

            max_curr_len = max([len(ex) for ex in tokenized_examples]) + n_terminals
            padding_length_including_terminals = min(max_curr_len, padding_length) \
                if padding_length else max_curr_len
            padding_length_excluding_terminals = padding_length_including_terminals - n_terminals

            _trimmed_examples = [
                ex[:padding_length_excluding_terminals] for ex in tokenized_examples
            ]

            split_lengths = [[1] * len(ex) for ex in _trimmed_examples]

            # convert (sub) words into their respective token ids
            seq_ids = [
                self._encode_text(
                    example,
                    padding_length_including_terminals,
                    add_terminals
                )
                for example in _trimmed_examples
            ]

            if _return_tokenized_examples:
                _examples = _trimmed_examples
            else:
                _examples = None

        else:

            # We split input text at whitespace because tagger models always use query_text_type
            # as 'normalized_text' which consist of whitespaces irrespective of the choice of
            # langauge (English, Japanese, etc.)
            split_at = " "

            # tokenize each word of each input separately
            # get maximum length of each example, accounting for terminals (start & end tokens)
            # If padding_length is None, padding has to be done to the max length of the input batch
            tokenized_examples = [
                [self._tokenize(word) for word in example.split(split_at)] for example in examples
            ]

            max_curr_len = max([len(sum(t_ex, [])) for t_ex in tokenized_examples]) + n_terminals
            padding_length_including_terminals = min(max_curr_len, padding_length) \
                if padding_length else max_curr_len
            padding_length_excluding_terminals = padding_length_including_terminals - n_terminals

            _trimmed_examples = [
                _trim_a_list_of_sub_token_groups(
                    tokenized_example, padding_length_excluding_terminals
                )
                for tokenized_example in tokenized_examples
            ]  # List[List[List[str]]], innermost List[str] is a list of sub-words for a given word

            split_lengths = [[len(x) for x in ex] for ex in _trimmed_examples]

            # convert (sub) words into their respective token ids
            seq_ids = [
                self._encode_text(
                    sum(example, []),
                    padding_length_including_terminals,
                    add_terminals
                )
                for example in _trimmed_examples
            ]

            if _return_tokenized_examples:
                _examples = [sum(list_of_t, []) for list_of_t in _trimmed_examples]
            else:
                _examples = None

        return BatchData(**{
            # number of groups per example
            "seq_lengths": torch.as_tensor(  # Tensor1d[int]
                [len(_split_lengths) + n_terminals for _split_lengths in split_lengths],
                dtype=torch.long
            ),
            # len of each subgroup; for each example, sum of its split_lengths will be equal to
            # the sequence length minus terminals.
            "split_lengths": [
                torch.as_tensor(_split_lengths, dtype=torch.long)
                for _split_lengths in split_lengths
            ],  # List[Tensor1d[int]],
            "seq_ids": torch.as_tensor(seq_ids, dtype=torch.long),
            **({"_examples": _examples} if _return_tokenized_examples else {}),
        })

    def _encode_text(self, list_of_tokens: List[str], padding_length: int, add_terminals: bool):
        """
        Encodes a list of tokens in to a list of ids based on vocab and special token ids.

        Args:
            list_of_tokens (List[str]): List of words or sub-words that are to be encoded
            padding_length (int): Maximum length of the encoded sequence; sequences shorter than
                this length are padded with a pad index while longer sequences are trimmed.
                Upon encoding, the length of outputted list of ids will be this length.
            add_terminals (bool): Whether terminal start and end tokens are to be added or not to
                the encoded sequence

        Returns:
            list_of_ids (List[int]): Sequence of ids corresponding to the input tokens
        """
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
        return list_of_ids

    def get_vocab(self) -> Dict:
        return self.token2id

    @property
    def number_of_terminal_tokens(self) -> int:
        """
        Returns the (maximum) number of terminal tokens used by the encoder during
        batch encoding when add_terminals is set to True.
        """
        return 2


class WhitespaceEncoder(AbstractVocabLookupEncoder):
    """
    Encoder that tokenizes at whitespace. Not useful for languages such as Chinese.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if self.classification_type == ClassificationType.TEXT:
            msg = "For languages like Japanese, Chinese, etc. that do not have whitespaces, " \
                  "consider using a pretrained huggingface tokenizer or a character tokenizer " \
                  "when not using 'query_text_type':'normalized_text'."
            logger.warning(msg)

    def _tokenize(self, text: str) -> List[str]:
        return text.strip("\n").split(" ")


class CharEncoder(AbstractVocabLookupEncoder):
    """
    A simple tokenizer that tokenizes at character level
    """

    def _tokenize(self, text: str) -> List[str]:
        return list(text.strip("\n"))


class WhitespaceAndCharDualEncoder(AbstractVocabLookupEncoder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.char_token2id = {}

        if self.classification_type != ClassificationType.TAGGER:
            msg = "The inputted tokenizer type can only be used with a tagger model."
            raise ValueError(msg)

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
    def _char_tokenize(text: str) -> List[str]:
        return list(text.strip("\n"))

    def _tokenize(self, text: str) -> List[str]:
        return text.strip("\n").split(" ")

    def prepare(self, examples: List[str]):
        super().prepare(examples)

        examples = [ex.strip() for ex in examples]
        all_tokens = dict.fromkeys(
            chain.from_iterable([self._char_tokenize(text) for text in examples])
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

    @property
    def number_of_char_terminal_tokens(self) -> int:
        """
        Returns the number of char terminal tokens used by the encoder during batch encoding when
        add_terminals is set to True
        """
        return 2

    @property
    def number_of_terminal_tokens(self) -> int:
        """
        Returns the number of terminal tokens used by the encoder during batch encoding when
        add_terminals is set to True.
        """
        return 0

    def batch_encode(
        self, examples: List[str], char_padding_length: int = None, char_add_terminals: bool = True,
        add_terminals: bool = False, _return_tokenized_examples: bool = False, **kwargs
    ) -> BatchData:

        if add_terminals:
            msg = f"The param 'add_terminals' must not be True to encode a batch using " \
                  f"{self.__class__.__name__}."
            logger.error(msg)
            raise ValueError(msg)

        batch_data = super().batch_encode(
            examples=examples, add_terminals=False, _return_tokenized_examples=True, **kwargs
        )

        # use tokenized examples to obtain tokens for char tokenization
        _examples = batch_data.pop("_examples")
        char_seq_ids, char_seq_lengths = [], []
        for _seq_tokens in _examples:
            # compute padding length for character sequences
            _curr_max = max([len(word) for word in _seq_tokens])
            _curr_max = _curr_max + self.number_of_char_terminal_tokens \
                if char_add_terminals else _curr_max
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
            list_of_tokens[:padding_length - self.number_of_char_terminal_tokens]
            if add_terminals else list_of_tokens[:padding_length]
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

        if self.classification_type == ClassificationType.TEXT:
            msg = f"The pre-tokenizer for {self.__class__.__name__} is set to 'Whitespace'. " \
                  f"For languages like Japanese, Chinese, etc. that do not have whitespaces, " \
                  f"consider using a pretrained huggingface tokenizer or a character tokenizer " \
                  f"when not using 'query_text_type':'normalized_text'."
            logger.warning(msg)

    def prepare(self, examples: List[str]):
        """
        references:
        - Huggingface: tutorials/python/training_from_memory.html @ https://tinyurl.com/6hxrtspa
        - https://huggingface.co/docs/tokenizers/python/latest/index.html
        """

        self._prepare_pipeline()
        trainer = self.trainer(
            # vocab_size=30000,
            vocab_size=100,
            special_tokens=self.__class__.SPECIAL_TOKENS
        )
        self.tokenizer.train_from_iterator(examples, trainer=trainer, length=len(examples))

    def _prepare_pipeline(self):
        self.tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])
        # TODO: The PreTokenizer which is Whitespace currently can be made customizable so as to
        #  use for languages that have no whitespaces such as Japanese, Chinese, etc.
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
        self, examples: List[str], padding_length: int = None, add_terminals: bool = True, **kwargs
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
            msg = f"The param 'add_terminals' must be True to encode a batch using " \
                  f"{self.__class__.__name__}."
            raise ValueError(msg)

        if padding_length is not None:
            msg = f"{self.__class__.__name__} does not support setting padding length during" \
                  f"batch_encode() method."
            logger.warning(msg)

        n_terminals = self.number_of_terminal_tokens if add_terminals else 0

        # We do not distinguish between ClassificationType.TEXT or  ClassificationType.TAGGER here.
        # Also note that the pre_tokenizer is currently set to Whitespace.
        output = self.tokenizer.encode_batch(examples, add_special_tokens=True)
        seq_ids = [o.ids for o in output]
        attention_masks = [o.attention_mask for o in output]
        words = [o.words for o in output]

        split_lengths = []
        for words_nums in words:  # an eg. sequence [None,0,1,2,3,3,3,3,4,4,...,16,None,...,None]
            curr_len = 0
            curr_num = 0
            _split_lengths = []
            for word_num in words_nums[1:]:  # the first corresponds to CLS token
                if word_num == curr_num:
                    curr_len += 1
                elif word_num is None:
                    _split_lengths.append(curr_len)
                    break
                else:
                    assert word_num == curr_num + 1, print(word_num, curr_num)
                    _split_lengths.append(curr_len)
                    curr_len = 1
                    curr_num = word_num
            split_lengths.append(_split_lengths)

        return BatchData(**{
            # num of groups per example
            "seq_lengths": torch.as_tensor(  # Tensor1d[int]
                [len(_split_lengths) + n_terminals for _split_lengths in split_lengths],
                dtype=torch.long
            ),
            # len of each subgroup; for each example, sum of its split_lengths will be equal to
            # the sum of attention mask minus terminals.
            "split_lengths": [
                torch.as_tensor(_split_lengths, dtype=torch.long)
                for _split_lengths in split_lengths
            ],  # List[Tensor1d[int]],
            "seq_ids": torch.as_tensor(seq_ids, dtype=torch.long),  # Tensor2d[int]
            "attention_masks": torch.as_tensor(attention_masks, dtype=torch.long),  # Tensor2d[int]
        })

    def get_vocab(self) -> Dict:
        return self.tokenizer.get_vocab()

    def get_pad_token_idx(self) -> int:
        return self.tokenizer.token_to_id("[PAD]")

    @property
    def number_of_terminal_tokens(self) -> int:
        """
        Returns the (maximum) number of terminal tokens used by the encoder during
        batch encoding when add_terminals is set to True.
        """
        return 2


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

        self._number_of_terminal_tokens = None
        self.__model_max_length = -1

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

    @property
    def number_of_terminal_tokens(self) -> int:
        """Overwrite parent class' definition of number of terminal tokens"""
        if not self._number_of_terminal_tokens:
            self._number_of_terminal_tokens = len(self.tokenizer.encode(""))
        return self._number_of_terminal_tokens

    @property
    def _model_max_length(self) -> int:
        """Returns the maximum length of tokens per example allowed for the pretrained tokenizer"""
        if self.__model_max_length == -1:
            try:
                self.__model_max_length = self.tokenizer.model_max_length
            except AttributeError as e:
                # case in which the huggingface tokenizer doesn't have this attribute
                logger.info(e)
                self.__model_max_length = None
        return self.__model_max_length

    def batch_encode(
        self, examples: List[str], padding_length: int = None, add_terminals: bool = True, **kwargs
    ) -> BatchData:

        if not add_terminals:
            msg = f"The param 'add_terminals' must be True to encode a batch using " \
                  f"{self.__class__.__name__}."
            logger.error(msg)
            raise ValueError(msg)

        n_terminals = self.number_of_terminal_tokens if add_terminals else 0

        if self.classification_type == ClassificationType.TEXT:

            # https://huggingface.co/docs/transformers/v4.16.2/en/preprocessing
            hgf_encodings = self.tokenizer(
                examples, padding=True, truncation=True, max_length=padding_length,
                return_tensors="pt"
            )  # Huggingface returns a BatchEncodings object; needs to be converted to a dictionary
            split_lengths = [
                [1] * (sum(msk) - n_terminals) for msk in hgf_encodings["attention_mask"]
            ]

        else:

            # We split input text at whitespace because tagger models always use query_text_type
            # as 'normalized_text' which consist of whitespaces irrespective of the choice of
            # langauge (English, Japanese, etc.)
            split_at = " "

            if any([
                "GPT2Tokenizer" in str(parent_class) for parent_class in
                self.tokenizer.__class__.__mro__
            ]):  # tokenizers like RobertaTokenizer that use Byte-level BPE (eg. distilroberta-base)
                msg = "The inputted choice of pretrained huggingface tokenizer is based on " \
                      "Byte-level BPE (eg. 'GPT2Tokenizer', 'RobertaTokenizer', etc.) which " \
                      "treats spaces like parts of the tokens. " \
                      "This conflicts with the use of 'query_text_type':'normalized_text' for " \
                      "tagger models. Consider using a different pretrained model for tagging."
                raise NotImplementedError(msg)

            # tokenize each word of each input separately
            # get maximum length of each example, accounting for terminal tokens- cls, sep
            # If padding_length is None, padding has to be done to the max length of the input batch
            tokenized_examples = [
                [self._tokenize(word) for word in example.split(split_at)] for example in examples
            ]

            max_curr_len = max([len(sum(t_ex, [])) for t_ex in tokenized_examples]) + n_terminals
            padding_length_including_terminals = min(max_curr_len, padding_length) \
                if padding_length else max_curr_len
            if self._model_max_length:
                # padding_length cannot exceed the transformer model's maximum length
                padding_length_including_terminals = min(
                    padding_length_including_terminals, self._model_max_length)
            padding_length_excluding_terminals = padding_length_including_terminals - n_terminals

            _trimmed_examples = [
                _trim_a_list_of_sub_token_groups(tokenized_example,
                                                 padding_length_excluding_terminals)
                for tokenized_example in tokenized_examples
            ]  # List[List[List[str]]], innermost List[str] is a list of sub-words for a given word

            split_lengths = [[len(x) for x in ex] for ex in _trimmed_examples]

            # Problem if trimmed examples are not detokenized before __call__ method:
            #  Huggingface does not provide a method where-in the tokenizer's encode method(__call__
            #  method in the latest versions) can be called with a list of already tokenized text.
            #  Calling ```tokenized_examples=[" ".join(sum(ex, [])) for ex in _trimmed_examples]```
            #  & then passing it to ```self.tokenizer.__call__(tokenized_examples, ...)```
            #  inadvertently re-tokenizes already tokenized strings. Eg: "ttyl" tokenized
            #  to ["t", "##ty", "#l"] upon trimming operations and then reverted to "t ##ty ##l" is
            #  incorrectly tokenized later in __call__ as ["t", "#", "#", "t", "y", "#", "#", "l"]
            #  and then encoded into ids.
            #  Note that this is not the case with some subclasses of
            #  AbstractHuggingfaceTrainableEncoder (e.g. BytePairEncodingEncoder)
            #  which might not have prepending tokens such as '##' for sub-words.
            _detokenized_examples = [
                split_at.join(
                    [self.tokenizer.convert_tokens_to_string(group) for group in _trimmed_example]
                )
                for _trimmed_example in _trimmed_examples
            ]
            # https://huggingface.co/docs/transformers/v4.16.2/en/preprocessing
            hgf_encodings = self.tokenizer(
                _detokenized_examples, padding=True, truncation=True, max_length=None,
                return_tensors="pt"
            )  # Huggingface returns a BatchEncodings object; needs to be converted to a dictionary

        return BatchData(**{
            # number of groups per example
            "seq_lengths": torch.as_tensor(  # Tensor1d[int]
                [len(_split_lengths) + n_terminals for _split_lengths in split_lengths],
                dtype=torch.long
            ),
            # len of each subgroup; for each example, sum of its split_lengths will be equal to
            # the sum of attention mask minus terminals.
            "split_lengths": [
                torch.as_tensor(_split_lengths, dtype=torch.long)
                for _split_lengths in split_lengths
            ],  # List[Tensor1d[int]],
            # all the different outputs produced by huggingface's pretrained tokenizer,
            # consisting of inputs_ids, attention_masks, etc.
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
