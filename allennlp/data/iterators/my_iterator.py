from typing import Iterable
import logging
import random
import itertools

from allennlp.common.util import lazy_groups_of
from allennlp.data.instance import Instance
from allennlp.data.iterators.data_iterator import DataIterator
from allennlp.data.dataset import Batch
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.fields import TextField
from allennlp.data.tokenizers.token import Token

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DataIterator.register("my")
class MyIterator(DataIterator):
    """
    A very basic iterator that takes a dataset, possibly shuffles it, and creates fixed sized batches.

    It takes the same parameters as :class:`allennlp.data.iterators.DataIterator`
    """

    def _create_batches(self, instances: Iterable[Instance], shuffle: bool):
        instances = iter(instances)
        sequence_length = 1000
        bptt_length = 20
        token_indexer = {"tokens": SingleIdTokenIndexer()}
        assert sequence_length % bptt_length == 0

        while True:
            print("starting sequence gen")
            sequences = [build_seq(instances, sequence_length + 1) for _ in range(self._batch_size)]
            for i in range(0, sequence_length, bptt_length):
                batch = []
                for sequence in sequences:
                    x = sequence[i : i + bptt_length]
                    x = TextField(x, token_indexer)
                    y = sequence[i+1 : i + bptt_length + 1]
                    y = TextField(y, token_indexer)
                    batch.append(Instance({"input_tokens": x, "output_tokens": y}))
                yield Batch(batch)



def build_seq(iterator, n):
    seq = []
    while len(seq) < n:
        sent = next(iterator)
        seq += [Token("<s>")] + sent.fields["tokens"].tokens + [Token("</s>")]

    return seq[:n]
