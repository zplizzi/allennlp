{
    "dataset_reader":{
        "type":"language_modeling",
		"tokenizer": {
			"type": "word",
			"word_splitter": "just_spaces"
		},
		"lazy": "true"
    },
	"vocabulary": {
			"directory_path": "/home/zplizzi/temp/allennlp/vocab/vocabulary/"
		},
    "train_data_path": "/home/zplizzi/data/1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled/news.en-00001-of-00100",
    "validation_data_path": "/home/zplizzi/data/1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled/news.en-00001-of-00100",
    "model": {
      "type": "language_model",
    },
    "iterator": {
      "type": "bucket",
      "sorting_keys": [["input_tokens", "num_tokens"]],
      "batch_size" : 8
    },
    "trainer": {
      "learning_rate_scheduler": {
        "type": "multi_step",
        "milestones": [40, 50, 60, 70, 80],
        "gamma": 0.8
      },
      "num_epochs": 150,
      "grad_norm": 5.0,
      "patience": 20,
      "validation_metric": "+accuracy",
      "cuda_device": 0,
      "optimizer": {
        "type": "adadelta",
        "lr": 1.0,
        "rho": 0.95
      }
    }
  }
