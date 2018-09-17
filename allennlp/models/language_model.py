from typing import Dict, Tuple, List, Optional, NamedTuple, Any
from overrides import overrides

import torch
from torch.nn.modules.linear import Linear
from nltk import Tree

from allennlp.common.checks import check_dimensions_match
from allennlp.data import Vocabulary
from allennlp.modules import Seq2SeqEncoder, TimeDistributed, TextFieldEmbedder, FeedForward
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.span_extractors.span_extractor import SpanExtractor
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.nn.util import masked_softmax, get_lengths_from_binary_sequence_mask
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.training.metrics import EvalbBracketingScorer, DEFAULT_EVALB_DIR
from allennlp.common.checks import ConfigurationError


@Model.register("language_model")
class LanguageModel(Model):
    """

    Parameters
    ----------
    vocab : ``Vocabulary``, required
        A Vocabulary, required in order to compute sizes for input/output projections.
    text_field_embedder : ``TextFieldEmbedder``, required
        Used to embed the ``tokens`` ``TextField`` we get as input to the model.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """

    def __init__(self,
                 vocab: Vocabulary,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None):
        super(LanguageModel, self).__init__(vocab, regularizer)

        self.num_classes = self.vocab.get_vocab_size("labels")

        self.embedding_dim = 256
        self.hidden_dim = 512
        self.embedding = torch.nn.Embedding(vocab.get_vocab_size(),
                                            self.embedding_dim)
        self.decoder = torch.nn.Linear(self.hidden_dim, self.num_classes)

        self.LSTM = torch.nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=.1)

        initializer(self)

    @overrides
    def forward(
            self,  # type: ignore
            input_tokens,
            output_tokens,
    ) -> Dict[str, torch.Tensor]:
        x = self.embedding(input_tokens)
        x, _ = self.LSTM(x)
        class_prob = self.decoder(x)
        output_dict = {}
        output_dict["class_prob"] = class_prob

        # pylint: disable=arguments-differ
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        all_metrics = {}
        all_metrics["tag_accuracy"] = self.tag_accuracy.get_metric(reset=reset)
        if self._evalb_score is not None:
            evalb_metrics = self._evalb_score.get_metric(reset=reset)
            all_metrics.update(evalb_metrics)
        return all_metrics
