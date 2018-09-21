from typing import Dict
import logging
import os
import random

from overrides import overrides

from allennlp.data.instance import Instance
from allennlp.data.tokenizers.tokenizer import Tokenizer
from allennlp.data.tokenizers import WordTokenizer
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.token_indexers.token_indexer import TokenIndexer
from allennlp.data.fields import TextField
from allennlp.data.token_indexers import SingleIdTokenIndexer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def filename(folder, i):
    fname = f"news.en-{i:05d}-of-00100"
    return os.path.expanduser(folder + fname)


@DatasetReader.register("billion_words")
class BillionWordsReader(DatasetReader):
    """
    Parameters
    ----------
    tokenizer : ``Tokenizer``, optional (default=``WordTokenizer()``)
        We use this ``Tokenizer`` for the text.  See :class:`Tokenizer`.
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        We use this to define the input representation for the text.  See :class:`TokenIndexer`.
        Note that the `output` representation will always be single token IDs - if you've specified
        a ``SingleIdTokenIndexer`` here, we use the first one you specify.  Otherwise, we create
        one with default parameters.
    """

    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False,
                 shuffle: bool = True) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._shuffle = shuffle
        self._token_indexers = token_indexers or {
            "tokens": SingleIdTokenIndexer()
        }

        # No matter how you want to represent the input, we'll always represent the output as a
        # single token id.  This code lets you learn a language model that concatenates word
        # embeddings with character-level encoders, in order to predict the word token that comes
        # next.
        self._output_indexer: Dict[str, TokenIndexer] = None
        for name, indexer in self._token_indexers.items():
            if isinstance(indexer, SingleIdTokenIndexer):
                self._output_indexer = {name: indexer}
                break
        else:
            self._output_indexer = {"tokens": SingleIdTokenIndexer()}

    @overrides
    def _read(self, dataset_path: str):
        file_ids = list(range(1, 2))
        if self._shuffle:
            random.shuffle(file_ids)

        for file_id in file_ids:
            with open(filename(dataset_path, file_id), "r") as text_file:
                lines = text_file.readlines()
            print("read file")

            lines = [line.strip() for line in lines]
            #lines = lines[:500]
            if self._shuffle:
                random.shuffle(lines)


            for line in lines:
                tokenized_string = self._tokenizer.tokenize(line)
                input_field = TextField(tokenized_string[:-1],
                                        self._token_indexers)
                output_field = TextField(tokenized_string[1:],
                                         self._output_indexer)
                yield Instance({
                    'tokens': input_field,
                    # 'output_tokens': output_field
                })

    @overrides
    def text_to_instance(self, sentence: str) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ
        tokenized_string = self._tokenizer.tokenize(sentence)
        input_field = TextField(tokenized_string[:-1], self._token_indexers)
        output_field = TextField(tokenized_string[1:], self._output_indexer)
        return Instance({
            'input_tokens': input_field,
            'output_tokens': output_field
        })
