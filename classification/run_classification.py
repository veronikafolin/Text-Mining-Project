import logging
import os
import random
import sys
import json
import math
import torch
from dataclasses import dataclass, field
from typing import Optional
import wandb

import datasets
import evaluate
import numpy as np
from datasets import load_dataset, load_from_disk
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, f1_score
from codecarbon import EmissionsTracker
from lsg_converter import LSGConverter


import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
    MBartTokenizer,
    MBartTokenizerFast,
    MBart50Tokenizer,
    MBart50TokenizerFast,
    get_scheduler,
)
from transformers.utils import check_min_version
from torch.utils.data import DataLoader
from transformers.utils.versions import require_version

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.30.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r requirements.txt")

logger = logging.getLogger(__name__)


# TODO: specify the correct names of input/output columns
task_name_mapping = {
    "ruling_classification": ("full_text", None, "ruling_type"),
    "judgment_classification": ("full_text", None, "judgment_type")
}
lsg_architecture = {"facebook/mbart-large-50": "MBartForSequenceClassification"}


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    lang: Optional[str] = field(default=None, metadata={"help": "Language id for tasks."})
    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on: " + ", ".join(task_name_mapping.keys())},
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_name_local: Optional[str] = field(
        default=None, metadata={"help": "The name of the local dataset to use."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )
    source_prefix: Optional[str] = field(
        default="", metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})
    debug_mode: bool = field(default=False, metadata={"help": "If debug mode."})
    max_seq_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )

    logging : Optional[str] = field(
        default="disabled",
        metadata={
            "help": (
                "Set 'disabled' to disable wandb logging, or else select logging 'online' or 'offline'"
            )
        },
    )    

    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            if self.task_name not in task_name_mapping.keys():
                raise ValueError("Unknown task, you should pick one in " + ",".join(task_name_mapping.keys()))
        elif self.dataset_name is not None:
            pass
        elif self.train_file is None or self.validation_file is None:
            raise ValueError("Need either a GLUE task, a training/validation file or a dataset name.")
        else:
            train_extension = self.train_file.split(".")[-1]
            assert train_extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            validation_extension = self.validation_file.split(".")[-1]
            assert (
                    validation_extension == train_extension
            ), "`validation_file` should have the same extension (csv or json) as `train_file`."


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )
    resize_position_embeddings: Optional[bool] = field(
        default=None,
        metadata={
            "help": (
                "Whether to automatically resize the position embeddings if `max_source_length` exceeds "
                "the model's position embeddings."
            )
        },
    )
    lsg: bool = field(
        default=False,
        metadata={"help": "Adopt LSG Attention"},
    )



def get_carburacy(score, emission_train, emission_test, alpha=10, beta_train=1, beta_test=100):
    score = score + sys.float_info.epsilon
    carburacy_train = None
    if emission_train is not None:
        carburacy_train = math.exp(math.log(score/100, alpha)) / (1 + emission_train * beta_train)
    carburacy_test = None
    if emission_test is not None:
        carburacy_test = math.exp(math.log(score/100, alpha)) / (1 + emission_test * beta_test)
    carburacy = None
    if carburacy_train is not None and carburacy_test is not None:
        carburacy = (2 * carburacy_train * carburacy_test) / (carburacy_train + carburacy_test)
    return carburacy_train, carburacy_test, carburacy


def main():

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if data_args.debug_mode:
        wandb.init(mode="disabled")

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # TODO: add all the models that may need prefix (eg, mT0)
    if data_args.source_prefix is None and model_args.model_name_or_path in [
        "t5-small",
        "t5-base",
        "t5-large",
        "t5-3b",
        "t5-11b",
    ]:
        logger.warning(
            "You're running a t5 model but didn't provide a source prefix, which is the expected, e.g. with "
            "`--source_prefix 'summarize: ' `"
        )

    if training_args.do_train:
        training_args.output_dir += "/" + data_args.task_name + "_" + \
                                model_args.model_name_or_path.partition("/")[-1] + "_"
        training_args.output_dir += "lsg_" if model_args.lsg else ""
        training_args.output_dir += str(data_args.max_train_samples) + "_" + data_args.lang
    else:
        training_args.output_dir = model_args.model_name_or_path

    # assert not os.path.exists(training_args.output_dir), "Output directory already exists"

    wandb.init(mode=data_args.logging,
                name=training_args.output_dir.split("/")[2],
    )

    # Detecting last checkpoint.
    last_checkpoint = None

    # Set seed before initializing model.
    set_seed(training_args.seed)

    if data_args.dataset_name_local is not None:
        # Loading a local dataset.
        raw_datasets = load_from_disk(data_args.dataset_name_local)
    elif data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.lang,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
            extension = data_args.test_file.split(".")[-1]
        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )

    # Get the column names for input/target.
    dataset_columns = task_name_mapping.get(data_args.task_name, None)
    sentence1_key = dataset_columns[0]
    sentence2_key = dataset_columns[1]
    label_name = dataset_columns[-1]
    label_list = list(set(raw_datasets["train"][label_name]))
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        trust_remote_code=True,
    )

    if model_args.lsg:
        converter = LSGConverter(max_sequence_length=data_args.max_seq_length)
        model, tokenizer = converter.convert_from_pretrained(model_args.model_name_or_path, num_labels=num_labels,
                                                             architecture=lsg_architecture[model_args.model_name_or_path])
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_args.model_name_or_path, config=config, trust_remote_code=True)
 
    # TODO: check if also the other models need such settings
    if model.config.decoder_start_token_id is None and isinstance(tokenizer, (MBartTokenizer, MBartTokenizerFast)):
        if isinstance(tokenizer, MBartTokenizer):
            model.config.decoder_start_token_id = tokenizer.lang_code_to_id[data_args.lang]
        else:
            model.config.decoder_start_token_id = tokenizer.convert_tokens_to_ids(data_args.lang)

    if model.config.decoder_start_token_id is None and isinstance(tokenizer, (MBartTokenizer, MBartTokenizerFast)):
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    if (
            hasattr(model.config, "max_position_embeddings")
            and model.config.max_position_embeddings < data_args.max_seq_length
            and isinstance(tokenizer, (MBartTokenizer, MBartTokenizerFast))
    ):
        if model_args.resize_position_embeddings is None:
            logger.warning(
                "Increasing the model's number of position embedding vectors from"
                f" {model.config.max_position_embeddings} to {data_args.max_seq_length}."
            )
            model.resize_position_embeddings(data_args.max_seq_length)
        elif model_args.resize_position_embeddings:
            model.resize_position_embeddings(data_args.max_seq_length)
        else:
            raise ValueError(
                f"`--max_seq_length` is set to {data_args.max_seq_length}, but the model only has"
                f" {model.config.max_position_embeddings} position encodings. Consider either reducing"
                f" `--max_seq_length` to {model.config.max_position_embeddings} or to automatically resize the"
                " model's position encodings by passing `--resize_position_embeddings`."
            )

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    if (
            hasattr(model.config, "max_position_embeddings")
            and model.config.max_position_embeddings < data_args.max_seq_length
    ):
        if model_args.resize_position_embeddings is None:
            logger.warning(
                "Increasing the model's number of position embedding vectors from"
                f" {model.config.max_position_embeddings} to {data_args.max_seq_length}."
            )
            model.resize_position_embeddings(data_args.max_seq_length)
        elif model_args.resize_position_embeddings:
            model.resize_position_embeddings(data_args.max_seq_length)
        else:
            raise ValueError(
                f"`--max_seq_length` is set to {data_args.max_seq_length}, but the model only has"
                f" {model.config.max_position_embeddings} position encodings. Consider either reducing"
                f" `--max_seq_length` to {model.config.max_position_embeddings} or to automatically resize the"
                " model's position encodings by passing `--resize_position_embeddings`."
            )

    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        column_names = raw_datasets["train"].column_names
    elif training_args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        column_names = raw_datasets["validation"].column_names
    elif training_args.do_predict:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        column_names = raw_datasets["test"].column_names
    else:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id and
            data_args.task_name is not None
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if sorted(label_name_to_id.keys()) == sorted(label_list):
            label_to_id = {i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)}
        else:
            label_to_id = {v: i for i, v in enumerate(label_list)}
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {sorted(label_name_to_id.keys())}, dataset labels: {sorted(label_list)}."
                "\nIgnoring the model labels as a result.",
            )
    elif data_args.task_name is None:
        label_to_id = {v: i for i, v in enumerate(label_list)}
    else:  # TODO
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in config.label2id.items()}
    elif data_args.task_name is not None:
        model.config.label2id = {l: i for i, l in enumerate(label_list)}
        model.config.id2label = {id: label for label, id in config.label2id.items()}

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        # remove pairs where at least one record is None
        inputs1, targets = [], []
        inputs2 = None
        for i in range(len(examples[sentence1_key])):
            inputs1.append(examples[sentence1_key][i])
            targets.append(examples[label_name][i])
        inputs1 = [prefix + inp for inp in inputs1]
        # Tokenize the texts
        args = ((inputs1,) if inputs2 is None else (inputs1, inputs2))
        result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)
        # Map labels to IDs
        if label_to_id is not None and label_name in examples:
            result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples[label_name]]
        return result

    # Data collator will default to DataCollatorWithPadding when the tokenizer is passed to Trainer, so we change it if
    # we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None    

    if training_args.do_train:
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        train_dataset = train_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=False,
            #load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on train dataset",)
        if data_args.task_name == "judgment_classification":
            train_dataset = train_dataset.filter(lambda e: e["label"] not in [4, 5, 8, 9])
        print(f"Training set size: {len(train_dataset)}")

        # Optimizer
        # Split weights in two groups, one with weight decay and the other not.
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": training_args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator,
        batch_size=training_args.per_device_train_batch_size)
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=training_args.learning_rate)
        num_update_steps_per_epoch = len(train_dataloader)
        max_train_steps = training_args.num_train_epochs * num_update_steps_per_epoch

        lr_scheduler = get_scheduler(
            name="linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=max_train_steps
        )
        optimizers = (optimizer, lr_scheduler)
    else:
        optimizers = (None, None)

    if training_args.do_eval:
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )

    if training_args.do_predict:
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = predict_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )
        
    # Get the metric function
    metric = evaluate.load("accuracy")

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)
        p_r_f1 = precision_recall_fscore_support(p.label_ids, preds)
        accuracy = accuracy_score(p.label_ids, preds)
        result = {
            "precision": round(100 * p_r_f1[0][0], 2),
            "recall": round(100 * p_r_f1[1][0], 2),
            "F1": round(100 * p_r_f1[2][0], 2),
            "accuracy": round(100 * accuracy, 2),
            "F1_macro": round(100 * f1_score(p.label_ids, preds, average="macro"), 2),
            "F1_micro": round(100 * f1_score(p.label_ids, preds, average="micro"), 2),
        }
        return result
    
    # TODO: check if Seq2SeqTrainer is needed since we use seq-to-seq models like MBart
    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
        optimizers=optimizers,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint

        train_tracker = EmissionsTracker(measure_power_secs=100000, save_to_file=False)
        train_tracker.start()
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        train_emissions = train_tracker.stop()
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))
        metrics["train_emissions"] = train_emissions

        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate(eval_dataset=eval_dataset)

        max_eval_samples = (
            data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        )
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")

        # Removing the `label` columns because it contains -1 and Trainer won't like that.
        labels = predict_dataset["label"]
        predict_dataset = predict_dataset.remove_columns("label")
        test_tracker = EmissionsTracker(measure_power_secs=100000, save_to_file=False)
        test_tracker.start()
        predictions = trainer.predict(predict_dataset, metric_key_prefix="predict")
        test_emissions = test_tracker.stop()

        if "roberta" in model_args.model_name_or_path:
            pred = np.argmax(predictions.predictions, axis=1)
        else:
            pred = np.argmax(predictions.predictions[0], axis=1)

        p_r_f1 = precision_recall_fscore_support(labels, pred)
        accuracy = accuracy_score(labels, pred)
        result = {
            "predict_P": round(100 * p_r_f1[0][0], 2),
            "predict_R": round(100 * p_r_f1[1][0], 2),
            "predict_F1": round(100 * p_r_f1[2][0], 2),
            "predict_accuracy": round(100 * accuracy, 2),
            "predict_F1_macro": round(100 * f1_score(labels, pred, average="macro"), 2),
            "predict_F1_micro": round(100 * f1_score(labels, pred, average="micro"), 2),
        }
        result["predict_F"] = round(np.mean([result["predict_F1_macro"], result["predict_F1_micro"]]) / \
            (1 + (np.var([result["predict_F1_macro"]/100, result["predict_F1_micro"]/100]))), 2)

        result["predict_emissions"] = test_emissions

        if not training_args.do_train:
            # Open the file in read mode
            with open(os.path.join(training_args.output_dir, "all_results.json"), 'r') as file:
                # Load the JSON data into a dictionary
                all_results = json.load(file)
            train_emissions = all_results["train_emissions"]
 
        train_carburacy, predict_carburacy, carburacy = get_carburacy(result["predict_F"], 
                                                                    train_emissions, test_emissions/len(predict_dataset))
        result["train_carburacy"] = train_carburacy
        result["predict_carburacy"] = predict_carburacy
        result["carburacy"] = carburacy

        trainer.log_metrics("predict", result)
        trainer.save_metrics("predict", result)

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": data_args.task_name}
    if data_args.task_name is not None:
        kwargs["language"] = data_args.lang
        kwargs["dataset_tags"] = "comma"
        kwargs["dataset_args"] = data_args.task_name
        kwargs["dataset"] = f"COMMA {data_args.task_name.upper()}"

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
