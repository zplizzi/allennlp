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
from allennlp.training.metrics import Average
from allennlp.common.checks import ConfigurationError

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


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

        self.num_classes = self.vocab.get_vocab_size()
        print(f"There are {self.num_classes} words in the vocab")

        self.embedding_dim = 256
        self.hidden_dim = 512
        self.embedding = torch.nn.Embedding(vocab.get_vocab_size(),
                                            self.embedding_dim)
        self.decoder = torch.nn.Linear(self.hidden_dim, self.num_classes)
        self.loss = torch.nn.CrossEntropyLoss()

        self.accuracy = CategoricalAccuracy()
        self.perplexity = Average()

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

        try:
            self.lstm_state = repackage_hidden(self.lstm_state)
        except AttributeError:
            pass

        input_tokens = input_tokens["tokens"]
        batch_size, num_sents = input_tokens.shape
        output_tokens = output_tokens["tokens"]
        x = self.embedding(input_tokens)
        try:
            x, self.lstm_state = self.LSTM(x, self.lstm_state)
        except AttributeError:
            x, self.lstm_state = self.LSTM(x)
        class_prob = self.decoder(x)
        class_prob_flat = class_prob.view(batch_size * num_sents, -1)
        targets_flat = output_tokens.view(-1)
        output_dict = {}
        output_dict["class_prob"] = class_prob
        output_dict["loss"] = self.loss(class_prob_flat, targets_flat)

        self.accuracy(class_prob, output_tokens)
        self.perplexity(output_dict["loss"])

        # pylint: disable=arguments-differ
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        all_metrics = {}
        all_metrics["accuracy"] = self.accuracy.get_metric(reset=reset)
        all_metrics["ppl"] = torch.exp(self.perplexity.get_metric(reset=reset))
        return all_metrics
