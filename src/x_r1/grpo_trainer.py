# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import textwrap
import warnings
from collections import defaultdict
from typing import Any, Callable, Optional, Sized, Union
from unittest.mock import patch
from dataset.utils.clips import svg_to_image
import torch
import torch.utils.data
import transformers
from PIL import Image
from dataset.utils.render import render_svg_from_text
import numpy as np
from accelerate.utils import broadcast_object_list, gather, gather_object, is_peft_model, set_seed
from accelerate.utils.other import is_compiled_module
from datasets import Dataset, IterableDataset
from packaging import version
from torch import nn
from torch.utils.data import Sampler
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available

from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.import_utils import is_rich_available, is_vllm_available
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.callbacks import SyncRefModelCallback
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import (
    generate_model_card,
    get_comet_experiment_url,
    pad,
    selective_log_softmax,
)


if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_vllm_available():
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams

if is_wandb_available():
    import wandb

# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


class RepeatRandomSampler(Sampler):
    """
    Sampler that repeats the indices of a dataset in a structured manner.

    Args:
        data_source (`Sized`):
            Dataset to sample from.
        mini_repeat_count (`int`):
            Number of times to repeat each index per batch.
        batch_size (`int`, *optional*, defaults to `1`):
            Number of unique indices per batch.
        repeat_count (`int`, *optional*, defaults to `1`):
            Number of times to repeat the full sampling process.
        seed (`int` or `None`, *optional*, defaults to `None`):
            Random seed for reproducibility (only affects this sampler).

    Example:
    ```python
    >>> sampler = RepeatRandomSampler(["a", "b", "c", "d", "e", "f", "g"], mini_repeat_count=2, batch_size=3, repeat_count=4)
    >>> list(sampler)
    [4, 4, 3, 3, 0, 0,
     4, 4, 3, 3, 0, 0,
     4, 4, 3, 3, 0, 0,
     4, 4, 3, 3, 0, 0,

     1, 1, 2, 2, 6, 6,
     1, 1, 2, 2, 6, 6,
     1, 1, 2, 2, 6, 6,
     1, 1, 2, 2, 6, 6]
    ```

    ```txt
    mini_repeat_count = 3
          -   -   -
         [0,  0,  0,  1,  1,  1,  2,  2,  2,  3,  3,  3,      |
          4,  4,  4,  5,  5,  5,  6,  6,  6,  7,  7,  7,      |
          8,  8,  8,  9,  9,  9, 10, 10, 10, 11, 11, 11,      |
                                                                repeat_count = 2
          0,  0,  0,  1,  1,  1,  2,  2,  2,  3,  3,  3,      |
          4,  4,  4,  5,  5,  5,  6,  6,  6,  7,  7,  7,      |
          8,  8,  8,  9,  9,  9, 10, 10, 10, 11, 11, 11, ...] |
          ---------   ---------   ---------   ---------
           ---------   ---------   ---------   ---------
            ---------   ---------   ---------   ---------
                         batch_size = 12
    ```
    """

    def __init__(
        self,
        data_source: Sized,
        mini_repeat_count: int,
        batch_size: int = 1,
        repeat_count: int = 1,
        seed: Optional[int] = None,
    ):
        self.data_source = data_source
        self.mini_repeat_count = mini_repeat_count
        self.batch_size = batch_size
        self.repeat_count = repeat_count
        self.num_samples = len(data_source)
        self.seed = seed
        self.generator = torch.Generator()  # Create a local random generator
        if seed is not None:
            self.generator.manual_seed(seed)

    def __iter__(self):
        # E.g., [2, 4, 3, 1, 0, 6, 5] (num_samples = 7)
        indexes = torch.randperm(self.num_samples, generator=self.generator).tolist()

        #    [2, 4, 3, 1, 0, 6, 5]
        # -> [[2, 4, 3], [1, 0, 6], [5]]  (batch_size = 3)
        indexes = [indexes[i : i + self.batch_size] for i in range(0, len(indexes), self.batch_size)]

        #    [[2, 4, 3], [1, 0, 6], [5]]
        # -> [[2, 4, 3], [1, 0, 6]]
        indexes = [chunk for chunk in indexes if len(chunk) == self.batch_size]

        for chunk in indexes:
            for _ in range(self.repeat_count):
                for index in chunk:
                    for _ in range(self.mini_repeat_count):
                        yield index

    def __len__(self) -> int:
        return self.num_samples * self.mini_repeat_count * self.repeat_count


class GRPOTrainer(Trainer):
    """
    Trainer for the Group Relative Policy Optimization (GRPO) method. This algorithm was initially proposed in the
    paper [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://huggingface.co/papers/2402.03300).

    Example:

    ```python
    from datasets import load_dataset
    from trl import GRPOTrainer

    dataset = load_dataset("trl-lib/tldr", split="train")

    def reward_func(completions, **kwargs):
        # Dummy reward function that rewards completions with more unique letters.
        return [float(len(set(completion))) for completion in completions]

    trainer = GRPOTrainer(
        model="Qwen/Qwen2-0.5B-Instruct",
        reward_funcs=reward_func,
        train_dataset=dataset,
    )

    trainer.train()
    ```

    Args:
        model (`Union[str, PreTrainedModel]`):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or
              a path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is
              loaded using [`~transformers.AutoModelForCausalLM.from_pretrained`] with the keywork arguments
              in `args.model_init_kwargs`.
            - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
        reward_funcs (`Union[RewardFunc, list[RewardFunc]]`):
            Reward functions to be used for computing the rewards. To compute the rewards, we call all the reward
            functions with the prompts and completions and sum the rewards. Can be either:

            - A single reward function, such as:
                - A string: The *model ID* of a pretrained model hosted inside a model repo on huggingface.co, or a
                path to a *directory* containing model weights saved using
                [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
                using [`~transformers.AutoModelForSequenceClassification.from_pretrained`] with `num_labels=1` and the
                keyword arguments in `args.model_init_kwargs`.
                - A [`~transformers.PreTrainedModel`] object: Only sequence classification models are supported.
                - A custom reward function: The function is provided with the prompts and the generated completions,
                  plus any additional columns in the dataset. It should return a list of rewards. For more details, see
                  [Using a custom reward function](#using-a-custom-reward-function).
            - A list of reward functions, where each item can independently be any of the above types. Mixing different
            types within the list (e.g., a string model ID and a custom reward function) is allowed.
        args ([`GRPOConfig`], *optional*, defaults to `None`):
            Configuration for this trainer. If `None`, a default configuration is used.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. It must include a column `"prompt"`. Any additional columns in the dataset is
            ignored. The format of the samples can be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Union[Dataset, IterableDataset]]`):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], *optional*, defaults to `None`):
            Processing class used to process the data. The padding side must be set to "left". If `None`, the
            processing class is loaded from the model's name with [`~transformers.AutoTokenizer.from_pretrained`].
        reward_processing_classes (`Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]`, *optional*, defaults to `None`):
            Processing classes corresponding to the reward functions specified in `reward_funcs`. Can be either:

            - A single processing class: Used when `reward_funcs` contains only one reward function.
            - A list of processing classes: Must match the order and length of the reward functions in `reward_funcs`.
            If set to `None`, or if an element of the list corresponding to a [`~transformers.PreTrainedModel`] is
            `None`, the tokenizer for the model is automatically loaded using [`~transformers.AutoTokenizer.from_pretrained`].
            For elements in `reward_funcs` that are custom reward functions (not [`~transformers.PreTrainedModel`]),
            the corresponding entries in `reward_processing_classes` are ignored.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*, defaults to `None`):
            List of callbacks to customize the training loop. Will add those to the list of default callbacks
            detailed in [here](https://huggingface.co/docs/transformers/main_classes/callback).

            If you want to remove one of the default callbacks used, use the [`~transformers.Trainer.remove_callback`]
            method.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your
            model and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
        peft_config ([`~peft.PeftConfig`], *optional*, defaults to `None`):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped.
    """

    _tag_names = ["trl", "grpo"]

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: Optional[GRPOConfig] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")
        self.advantage_offset = args.advantage_offset
        self.logp_variance_reg_coef = args.logp_variance_reg_coef
        # Models
        # Trained model
        model_init_kwargs = args.model_init_kwargs or {}
        if isinstance(model, str):
            model_id = model
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
                pass  # torch_dtype is already a torch.dtype or "auto" or None
            elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
                torch_dtype = getattr(torch, torch_dtype)
                model_init_kwargs["torch_dtype"] = torch_dtype
            else:
                raise ValueError(
                    "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                    f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
                )
            # Disable caching if gradient checkpointing is enabled (not supported)
            model_init_kwargs["use_cache"] = (
                False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
            )
            model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated. "
                    "This argument can only be used when the `model` argument is a string."
                )

        if peft_config is not None:
            if not is_peft_available():
                raise ImportError("PEFT is required to use `peft_config`. Run `pip install peft`.")
            model = get_peft_model(model, peft_config)

        # Enable gradient checkpointing if requested
        if args.gradient_checkpointing:
            model = self._enable_gradient_checkpointing(model, args)

        # Reference model
        self.beta = args.beta
        if self.beta == 0.0:
            # If beta is 0.0, the reference model is not needed
            self.ref_model = None
        elif is_deepspeed_zero3_enabled():
            self.ref_model = AutoModelForCausalLM.from_pretrained(model_id, **model_init_kwargs)
        elif is_peft_model(model):
            # If PEFT is used, the reference model is not needed since the adapter can be disabled
            # to revert to the initial model.
            self.ref_model = None
        else:
            # If PEFT configuration is not provided, create a reference model based on the initial model.
            self.ref_model = create_reference_model(model)

        # Processing class
        if processing_class is None:
            processing_class = AutoTokenizer.from_pretrained(model.config._name_or_path, padding_side="left")

        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
        self.reward_funcs = reward_funcs

        # Reward weights
        if args.reward_weights is not None:
            if len(args.reward_weights) != len(reward_funcs):
                raise ValueError(
                    f"Number of reward weights ({len(args.reward_weights)}) must match number of reward "
                    f"functions ({len(reward_funcs)})"
                )
            self.reward_weights = torch.tensor(args.reward_weights, dtype=torch.float32)
        else:
            self.reward_weights = torch.ones(len(reward_funcs), dtype=torch.float32)

        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("The number of reward processing classes must match the number of reward functions.")

        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        # Data collator
        def data_collator(features):  # No data collation is needed in GRPO
            return features

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length  # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations  # = G in the GRPO paper
        self.use_vllm = args.use_vllm

        # Multi-step
        self.num_iterations = args.num_iterations  # = 𝜇 in the GRPO paper
        self.epsilon = args.epsilon
        # Tracks the number of iterations (forward + backward passes), including those within a gradient accumulation cycle.
        self._step = 0
        # Buffer the batch to reuse generated outputs across multiple updates. For more details, see
        # `_get_train_sampler` and `_prepare_inputs`.
        self._buffered_inputs = [None] * args.gradient_accumulation_steps

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in GRPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True

        # Initialize the metrics
        self._metrics = defaultdict(list)
        self.log_completions = args.log_completions

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        # Check if the per_device_train/eval_batch_size * num processes can be divided by the number of generations
        num_processes = self.accelerator.num_processes
        global_batch_size = args.per_device_train_batch_size * num_processes
        possible_values = [n_gen for n_gen in range(2, global_batch_size + 1) if (global_batch_size) % n_gen == 0]
        if self.num_generations not in possible_values:
            raise ValueError(
                f"The global train batch size ({num_processes} x {args.per_device_train_batch_size}) must be evenly "
                f"divisible by the number of generations per prompt ({self.num_generations}). Given the current train "
                f"batch size, the valid values for the number of generations are: {possible_values}."
            )
        if self.args.eval_strategy != "no":
            global_batch_size = args.per_device_eval_batch_size * num_processes
            possible_values = [n_gen for n_gen in range(2, global_batch_size + 1) if (global_batch_size) % n_gen == 0]
            if self.num_generations not in possible_values:
                raise ValueError(
                    f"The global eval batch size ({num_processes} x {args.per_device_eval_batch_size}) must be evenly "
                    f"divisible by the number of generations per prompt ({self.num_generations}). Given the current "
                    f"eval batch size, the valid values for the number of generations are: {possible_values}."
                )

        # Ensure each process receives a unique seed to prevent duplicate completions when generating with
        # transformers if num_generations exceeds per_device_train_batch_size. We could skip it if we use vLLM, but
        # it's safer to set it in all cases.
        set_seed(args.seed, device_specific=True)

        if self.use_vllm:
            if not is_vllm_available():
                raise ImportError(
                    "vLLM is not available and `use_vllm` is set to True. Please install vLLM with "
                    "`pip install vllm` to use it."
                )

            if self.accelerator.is_main_process:
                vllm_device = self.args.vllm_device
                if vllm_device == "auto":
                    if torch.cuda.device_count() == 1:
                        vllm_device = "cuda:0"  # particular case when training with onyl 1 GPU: share it
                    else:
                        vllm_device = f"cuda:{self.accelerator.num_processes}"  # take the next GPU idx
                # Check that the requested device is available
                if vllm_device.split(":")[0] == "cuda" and int(vllm_device.split(":")[1]) >= torch.cuda.device_count():
                    raise ValueError(
                        f"The requested device for vllm ({vllm_device}) is not available. You are likely using vLLM "
                        "without restricting the number of GPUs for training. Set the `--num_processes` argument to a "
                        "value lower than the number of GPUs available on your machine—typically, reducing it by one "
                        f"is sufficient. In your case: `--num_processes {torch.cuda.device_count() - 1}`."
                    )
                # Check that the requested device is not also used for training
                if vllm_device in {f"cuda:{idx}" for idx in range(self.accelerator.num_processes)}:
                    warnings.warn(
                        f"The requested device {vllm_device} is also being used for training. For higher throughput "
                        "and to avoid out-of-memory errors, it is recommended to use a dedicated device for vLLM. "
                        "If this is intentional, you may ignore this warning but should adjust "
                        "`vllm_gpu_memory_utilization` accordingly."
                    )
                # vLLM is not compatible with accelerate. So we need to patch it to make sure we can (1) place the vLLM
                # model on the desired device (world_size_patch) and (2) avoid a test that is not designed for our
                # setting (profiling_patch).
                world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
                profiling_patch = patch(
                    "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling", return_value=None
                )
                with world_size_patch, profiling_patch:
                    self.llm = LLM(
                        model=model.name_or_path,
                        device=vllm_device,
                        gpu_memory_utilization=self.args.vllm_gpu_memory_utilization,
                        dtype=self.args.vllm_dtype,
                        # Automatic Prefix Caching caches the KV cache of existing queries, so that a new query can
                        # directly reuse the KV cache if it shares the same prefix with one of the existing queries.
                        # This is particularly useful here because we generate completions from the same prompts.
                        enable_prefix_caching=self.args.vllm_enable_prefix_caching,
                        max_model_len=self.args.vllm_max_model_len,
                    )

                # Guided decoding, if enabled
                if args.vllm_guided_decoding_regex is not None:
                    guided_decoding = GuidedDecodingParams(backend="outlines", regex=args.vllm_guided_decoding_regex)
                else:
                    guided_decoding = None

                # Sampling parameters
                self.sampling_params = SamplingParams(
                    temperature=args.temperature,
                    max_tokens=self.max_completion_length,
                    guided_decoding=guided_decoding,
                    n=args.num_generations,
                )

            self._last_loaded_step = 0  # tag to avoid useless loading during grad accumulation

            # When using vLLM, the main process is responsible for loading the model weights. This can cause process
            # desynchronization and seems to lead to DeepSpeed hanging during initialization. To prevent this, we
            # synchronize all processes after vLLM has been fully initialized.
            self.accelerator.wait_for_everyone()
        else:
            self.generation_config = GenerationConfig(
                max_new_tokens=self.max_completion_length,
                do_sample=True,
                temperature=args.temperature,
                pad_token_id=processing_class.pad_token_id,
            )

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        # Add tags to the model
        self.model.add_model_tags(self._tag_names)

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        if args.sync_ref_model:
            self.add_callback(SyncRefModelCallback(ref_model=self.ref_model, accelerator=self.accelerator))

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)
        assert self.args.max_grad_norm is not None, "max_grad_norm must be set in GRPO"

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]

    def _get_train_sampler(self) -> Sampler:
        # Returns a sampler that
        # 1. ensures each prompt is repeated across multiple processes. This guarantees that identical prompts are
        #    distributed to different GPUs, allowing rewards to be computed and normalized correctly within each prompt
        #    group. Using the same seed across processes ensures consistent prompt assignment, preventing discrepancies
        #    in group formation.
        # 2. repeats the batch multiple times to allow reusing generaations across multiple updates. Refer to
        #    _prepare_inputs to see how the generations are stored and reused.

        # In the following figure, the values are the prompt indices. The first row shows the first sampled batch, the
        # second row shows the second sampled batch, and so on.
        #
        #                                     |     GPU 0     |     GPU 1     |     GPU 2    |
        #
        #               global_step   step     <───────>  num_generations=3
        #                                      <───────────> per_device_train_batch_size=4
        #                ▲   0          0      0   0   0   1   1   1   2   2   2   3   3   3  │
        #  grad_accum=3  │   0          1      4   4   4   5   5   5   6   6   6   7   7   7  │ Generate completions for each prompt
        #                ▼   0          2      8   8   8   9   9   9  10  10  10  11  11  11  │
        #
        #                    1          3      0   0   0   1   1   1   2   2   2   3   3   3  │ The sampled prompts are the same as in the first iteration
        #                    1          4      4   4   4   5   5   5   6   6   6   7   7   7  │ Reuse the completions (here, once, because num_iterations=2)
        #                    1          5      8   8   8   9   9   9  10  10  10  11  11  11  │
        #
        #                    2          6     12  12  12  13  13  13  14  14  14  15  15  15
        #                    2          7     16  16  16  17  17  17  18  18  18  19  19  19
        #                    2          8     20  20  20  21  21  21  22  22  22  23  23  23
        #                                          ...
        effective_batch_size = (
            self.args.per_device_train_batch_size
            * self.accelerator.num_processes
            * self.args.gradient_accumulation_steps
        )
        return RepeatRandomSampler(
            data_source=self.train_dataset,
            mini_repeat_count=self.num_generations,
            batch_size=effective_batch_size // self.num_generations,
            repeat_count=self.num_iterations,
            seed=self.args.seed,
        )

    def _get_eval_sampler(self, eval_dataset) -> Sampler:
        # See _get_train_sampler for an explanation of the sampler.
        return RepeatRandomSampler(
            data_source=eval_dataset,
            mini_repeat_count=self.num_generations,
            seed=self.args.seed,
        )

    def _enable_gradient_checkpointing(self, model: PreTrainedModel, args: GRPOConfig) -> PreTrainedModel:
        """Enables gradient checkpointing for the model."""
        # Ensure use_cache is disabled
        model.config.use_cache = False

        # Enable gradient checkpointing on the base model for PEFT
        if is_peft_model(model):
            model.base_model.gradient_checkpointing_enable()
        # Enable gradient checkpointing for non-PEFT models
        else:
            model.gradient_checkpointing_enable()

        gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs or {}
        use_reentrant = (
            "use_reentrant" not in gradient_checkpointing_kwargs or gradient_checkpointing_kwargs["use_reentrant"]
        )

        if use_reentrant:
            model.enable_input_require_grads()

        return model

    # Get the per-token log probabilities for the completions for the model and the reference model
    def _get_per_token_logps(self, model, input_ids, attention_mask, logits_to_keep):
    
        # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
        logits = model(input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=logits_to_keep + 1).logits
        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred

        input_ids = input_ids[:, -logits_to_keep:]
        # For transformers<=4.48, logits_to_keep argument isn't supported, so here we drop logits ourselves.
        # See https://github.com/huggingface/trl/issues/2770
        logits = logits[:, -logits_to_keep:]
        return selective_log_softmax(logits, input_ids)  #  compute logprobs for the input tokens


    def _get_completion_hidden_states(self, model, input_ids, attention_mask, completion_length):
        """Get the last layer hidden states for the completion tokens, averaged over sequence length."""
        outputs = model(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            output_hidden_states=True
        )
        # Get the last layer hidden states
        last_hidden_states = outputs.hidden_states[-1]
        del outputs
        # Extract only the completion portion
        completion_hidden_states = last_hidden_states[:, -completion_length:]
        completion_attention_mask = attention_mask[:, -completion_length:]
        
        # Average over sequence length (only considering non-padding tokens)
        # Shape: [batch_size, hidden_dim]
        sequence_lengths = completion_attention_mask.sum(dim=1, keepdim=True)
        sequence_lengths = torch.clamp(sequence_lengths, min=1.0)  # Prevent division by zero
        
        # Multiply by mask before averaging to zero out padding tokens
        masked_states = completion_hidden_states * completion_attention_mask.unsqueeze(-1)
        averaged_states = masked_states.sum(dim=1) / sequence_lengths
        del masked_states, last_hidden_states
        return averaged_states
        
    
    def _move_model_to_vllm(self):
        with unwrap_model_for_generation(
            self.model, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
        ) as unwrapped_model:
            if is_compiled_module(unwrapped_model):
                unwrapped_model = unwrapped_model._orig_mod
            if is_peft_model(unwrapped_model):
                unwrapped_model.merge_adapter()
                state_dict = unwrapped_model.state_dict()
                # Remove base_model and base_layer prefixes
                state_dict = {
                    k.removeprefix("base_model.model.").replace(".base_layer", ""): v for k, v in state_dict.items()
                }
                # Remove values with adapter prefix (example: "_lora")
                state_dict = {k: v for k, v in state_dict.items() if unwrapped_model.prefix not in k}
                # When module to save, remove its prefix and discard the original module
                state_dict = {
                    k.replace("modules_to_save.default.", ""): v
                    for k, v in state_dict.items()
                    if "original_module" not in k
                }
            else:
                state_dict = unwrapped_model.state_dict()
            if self.accelerator.is_main_process:
                llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                llm_model.load_weights(state_dict.items())
            # Unmerge the adapter to restore the model to its original state.
            # This must be done after loading weights to ensure they correspond to the merged state.
            if is_peft_model(unwrapped_model):
                unwrapped_model.unmerge_adapter()
    
    def safe_gather(self, tensor, timeout=30):
        """Safely gather tensors with timeout and fallback."""
        try:
            # First synchronize to ensure all processes are ready
            torch.cuda.synchronize()
            
            # Signal readiness to prevent deadlocks
            ready_tensor = torch.ones(1, device=self.accelerator.device)
            self.accelerator.gather(ready_tensor)
            
            # Now perform the actual gather operation
            return gather(tensor)
        except Exception as e:
            print(f"Process {self.accelerator.process_index}: Gather failed: {str(e)}")
            # Return local tensor as fallback
            return tensor
    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        
       
        inputs = self._generate_and_score_completions(inputs)
        return inputs
    # Add this function near the top of your class
    def debug_print(self, message):
        """Print debug information with rank information."""
        import datetime
        import torch
        
        rank = self.accelerator.process_index
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        global_rank = int(os.environ.get("RANK", "0"))
        device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
        
        # Get GPU memory usage if on CUDA
        mem_info = ""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(device) / (1024**3)  # GB
            reserved = torch.cuda.memory_reserved(device) / (1024**3)    # GB
            mem_info = f", Memory: {allocated:.2f}GB/{reserved:.2f}GB"
        
        timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(f"[{timestamp}] Rank {rank}/{global_rank} (GPU {local_rank}){mem_info}: {message}")
    def _generate_and_score_completions(
        self, inputs: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        # self.debug_print("Generating completions")
        device = self.accelerator.device
        prompts = [x["prompt"] for x in inputs]
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
        prompt_inputs = self.processing_class(
            prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
        )
        prompt_inputs = super()._prepare_inputs(prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        # Generate completions using either vLLM or regular generation
        if self.args.use_vllm:
            # First, have main process load weights if needed
            if self.state.global_step != self._last_loaded_step:
                self._move_model_to_vllm()
                self._last_loaded_step = self.state.global_step

            # Generate completions using vLLM: gather all prompts and use them in a single call in the main process
            all_prompts_text = gather_object(prompts_text)
            if self.accelerator.is_main_process:
                # Since 'prompts' contains 'num_generations' duplicates, we first take unique prompts, and generate
                # num_generations outputs for each one. This is faster than generating outputs for each duplicate
                # prompt individually.
                ordered_set_of_prompts = list(dict.fromkeys(all_prompts_text))
                all_outputs = self.llm.generate(
                    ordered_set_of_prompts, sampling_params=self.sampling_params, use_tqdm=False
                )
                completion_ids = []
                for outputs in all_outputs:
                    for output in outputs.outputs:
                        completion_ids.append(output.token_ids)
            else:
                completion_ids = [None] * len(all_prompts_text)
            # Broadcast the completions from the main process to all processes, ensuring each process receives its
            # corresponding slice.
            completion_ids = broadcast_object_list(completion_ids, from_process=0)
            process_slice = slice(
                self.accelerator.process_index * len(prompts),
                (self.accelerator.process_index + 1) * len(prompts),
            )
            completion_ids = completion_ids[process_slice]

            # Pad the completions, and concatenate them with the prompts
            completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
            completion_ids = pad(completion_ids, padding_value=self.processing_class.pad_token_id)
            prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        else:
            # Regular generation path
            with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped_model:
                prompt_completion_ids = unwrapped_model.generate(
                    prompt_ids, attention_mask=prompt_mask, generation_config=self.generation_config
                )

            # Compute prompt length and extract completion ids
            prompt_length = prompt_ids.size(1)
            prompt_ids = prompt_completion_ids[:, :prompt_length]
            completion_ids = prompt_completion_ids[:, prompt_length:]
        # self.debug_print("Finished completions")
        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # Concatenate prompt_mask with completion_mask for logit computation
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)

        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens
        # self.debug_print("Start logp computation")
        with torch.inference_mode():
            # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip it's
            # computation here, and use per_token_logps.detach() instead.
            if self.num_iterations > 1:
                old_per_token_logps = self._get_per_token_logps(
                    self.model, prompt_completion_ids, attention_mask, logits_to_keep
                )
            else:
                old_per_token_logps = None

            if self.beta == 0.0:
                ref_per_token_logps = None
            elif self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model, prompt_completion_ids, attention_mask, logits_to_keep
                )
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        self.model, prompt_completion_ids, attention_mask, logits_to_keep
                    )
            assert  self.beta != 0.0   
            # completion_hidden_states = self._get_completion_hidden_states(
            #     self.ref_model, 
            #     prompt_completion_ids,
            #     attention_mask,
            #     logits_to_keep
            # )    
        # self.debug_print("finished logp computation")

        # Decode the generated completions
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text
            
        
        # Add after the completions_text = self.processing_class.batch_decode(...) line
        # and before the rewards_per_func calculation

        # Calculate entropy of responses per prompt
        
       
                
        

        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, nn.Module):  # Module instead of PretrainedModel for compat with compiled models
                if is_conversational(inputs[0]):
                    messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                    texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                else:
                    texts = [p + c for p, c in zip(prompts, completions)]
                reward_inputs = reward_processing_class(
                    texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                )
                reward_inputs = super()._prepare_inputs(reward_inputs)
                with torch.inference_mode():
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
            else:
                # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                
                keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
                reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
                # self.debug_print(f"Start reward{i} computation")
                output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
                # self.debug_print(f"Finished reward{i} computation")
                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)
                
                
        # self.debug_print("Finished reward computation")
        # Gather the reward per function: this part is crucial, because the rewards are normalized per group and the
        # completions may be distributed across processes
        # torch.cuda.synchronize()
        rewards_per_func = gather(rewards_per_func)

        # Apply weights to each reward function's output and sum
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).sum(dim=1)

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-2)

        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]
        # Log the metrics
        

        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length)

        reward_per_func = rewards_per_func.mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, nn.Module):  # Module instead of PretrainedModel for compat with compiled models
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())

        self._metrics["reward"].append(rewards.mean().item())
        self._metrics["reward_std"].append(std_grouped_rewards.mean().item())
        solutions = [x["solution"] for x in inputs]
        # print(len(solutions))
        if self.log_completions and (self.state.global_step % self.args.logging_steps == 0 or "eval_" in getattr(self, "current_phase", "")):
            is_eval = "eval_" in getattr(self, "current_phase", "")
            prompts_to_log = gather_object(prompts_text)
            completions_to_log = gather_object(completions_text)
            rewards_to_log = rewards.tolist()
            solutions_to_log = gather_object(solutions)

            if self.accelerator.is_main_process:
                
                if self.args.report_to and "wandb" in self.args.report_to and wandb.run is not None:
                    import pandas as pd

                    prefix = "eval_" if is_eval else ""
                    step_name = f"{prefix}completions/step_{str(self.state.global_step)}"
                    # Track batch number for this step
                    if not hasattr(self, "_log_batch_counter"):
                        self._log_batch_counter = {}

                    key = f"{prefix}step_{str(self.state.global_step)}"
                    if key not in self._log_batch_counter:
                        self._log_batch_counter[key] = 0
                    batch_num = self._log_batch_counter[key]
                    self._log_batch_counter[key] += 1

                    # Use a unique name for each batch
                    unique_step_name = f"{step_name}/batch_{batch_num}"

                    
                    # For logging
                    table = {
                        "step": [str(self.state.global_step)] * len(rewards_to_log),
                        "prompt": prompts_to_log,
                        "completion": completions_to_log,
                        "reward": rewards_to_log,
                        "solution": solutions_to_log,
                    }

            
                
                
                    
                    images = []
                    for svg_code, caption  in zip(completions_to_log, solutions_to_log):
                        try:
                            # Try to render the SVG code to an image
                            img = render_svg_from_text(svg_code)
                            if img is not None:
                                # Convert PIL image to wandb compatible format
                                images.append(wandb.Image(img, caption=caption))
                            else:
                                # If rendering fails, use a placeholder
                                placeholder = np.zeros((100, 100, 3), dtype=np.uint8)
                                images.append(wandb.Image(Image.fromarray(placeholder)))
                        except Exception as e:
                            print(f"Error rendering SVG: {str(e)[:100]}...")
                            # Create a text-based placeholder image with error message
                            placeholder = np.zeros((100, 100, 3), dtype=np.uint8)
                            images.append(wandb.Image(Image.fromarray(placeholder)))
                    
                    # Add images to the table
                    table["rendered_svg"] = images
                            
                        
                        
                    
                    # print(len(solutions_to_log), len(rewards.tolist()))
                    for i, reward_func in enumerate(self.reward_funcs):
                        if isinstance(reward_func, nn.Module):  # Module instead of PretrainedModel for compat with compiled models
                            reward_func_name = reward_func.config._name_or_path.split("/")[-1]
                        else:
                            reward_func_name = reward_func.__name__
                        table[f"rewards/{reward_func_name}"] = rewards_per_func[:,i].tolist()
                        
                    df = pd.DataFrame(table)
                    wandb.log({f"{unique_step_name}": wandb.Table(dataframe=df)})
        rewards = rewards[process_slice]
        
        if self.state.global_step % 5 == 0:
            torch.cuda.empty_cache()

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
            "rewards": rewards,
            # "completion_hidden_states": completion_hidden_states,
        }

    def log_diversity_metrics(self, hidden_states, advantages):
        """
        Calculate embedding diversity metrics based on cosine similarity.
        
        Args:
            hidden_states: Tensor of shape [batch_size, hidden_dim] with averaged embeddings
            advantages: Tensor of shape [batch_size] with advantage values
            
    
        Returns:
            Dictionary of diversity metrics
        """
        import torch.nn.functional as F        
        batch_size = hidden_states.size(0)

        
        # Reshape to group by prompt
        batch_size = hidden_states.size(0)
        
        if batch_size % self.num_generations != 0:
            print(f"Warning: Batch size {batch_size} not divisible by num_generations {self.num_generations}. Skipping diversity metrics.")
            return
        num_prompts = batch_size // self.num_generations
        grouped_states = hidden_states.view(num_prompts, self.num_generations, -1)
        grouped_advantages = advantages.view(num_prompts, self.num_generations)
        
        # Calculate diversity metrics
        diversity_all_values = []
        diversity_right_values = []
        
        for i in range(num_prompts):
            embeds = grouped_states[i]  # [num_generations, hidden_dim]
            advs = grouped_advantages[i]  # [num_generations]
            
            # Normalize embeddings for cosine similarity
            norm_embeds = F.normalize(embeds, p=2, dim=1)
        
            # Calculate pairwise cosine similarities (n×n matrix)
            cos_sim = torch.mm(norm_embeds, norm_embeds.transpose(0, 1))
            
            # Mask out self-similarities (diagonal)
            mask = 1.0 - torch.eye(self.num_generations, device=cos_sim.device)
            masked_cos_sim = cos_sim * mask
            
            # Calculate diversity_all
            n = self.num_generations
            if n > 1:
                sum_sim = masked_cos_sim.sum() / 2
                diversity = 1.0 - (sum_sim / (n * (n - 1)))
                diversity_all_values.append(diversity.item())
            
            # Calculate diversity_right (only for completions with positive advantage)
            right_indices = (advs == 2).nonzero(as_tuple=True)[0]
            n_right = len(right_indices)
            
            if n_right > 1:
                # Extract embeddings for right completions only
                right_embeds = norm_embeds[right_indices]
                
                # Calculate cosine similarities
                right_cos_sim = torch.mm(right_embeds, right_embeds.transpose(0, 1))
                
                # Mask out self-similarities
                right_mask = 1.0 - torch.eye(n_right, device=right_cos_sim.device)
                masked_right_cos_sim = right_cos_sim * right_mask
                
                sum_right_sim = masked_right_cos_sim.sum() / 2
                right_diversity = 1.0 - (sum_right_sim / (n_right * (n_right - 1)))
                diversity_right_values.append(right_diversity.item())
            else:
                pass
        
        # Convert to tensors for proper gathering across processes
        if diversity_all_values:
            
            
            self._metrics["diversity_all"].append(torch.tensor(diversity_all_values).mean().item())
            
        
        if diversity_right_values:
            
            self._metrics["diversity_right"].append(torch.tensor(diversity_right_values).mean().item())
            
            
        

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        # Compute the per-token log probabilities for the model
        # print("start compute_loss")
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        per_token_logps = self._get_per_token_logps(model, input_ids, attention_mask, logits_to_keep)
        # print("_get_per_token_logps")
        # Compute the KL divergence between the model and the reference model
        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )
            per_token_kl = torch.clamp(per_token_kl, min=-10, max = 10)

        seq_logps = (per_token_logps * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)
        rewards = inputs["rewards"]
        reg_term = self._calculate_logp_variance_regularization(seq_logps, rewards)
        # print("_calculate_logp_variance_regularization")

        # Compute the loss
        advantages = inputs["advantages"]
        # Apply advantage offset
        advantages = advantages - self.advantage_offset
        
        
        # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip it's computation (see
        # _generate_and_score_completions) and use per_token_logps.detach() instead.
        old_per_token_logps = inputs["old_per_token_logps"] if self.num_iterations > 1 else per_token_logps.detach()
        # old_per_token_logps = inputs['ref_per_token_logps']
        ratio_diff = torch.clamp(per_token_logps - old_per_token_logps, -10.0, 10.0)
        coef_1 = torch.exp(ratio_diff)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon, 1 + self.epsilon)
        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl
        loss = (per_token_loss * completion_mask).sum() / completion_mask.sum()
        
        #loss += self.logp_variance_reg_coef * reg_term
        

        # Log the metrics
        logps = (per_token_logps * completion_mask).sum(dim = -1)
        logps = self.accelerator.gather_for_metrics(logps)
        entropy = -logps.mean()
        
        self._metrics["entropy"].append(entropy.item())
        
        
        # hidden_states = inputs["completion_hidden_states"]
        # self.log_diversity_metrics(self.accelerator.gather_for_metrics(hidden_states), self.accelerator.gather_for_metrics(inputs["rewards"]))
        

        if self.beta != 0.0:
            mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
            self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        is_clipped = (per_token_loss1 < per_token_loss2).float()
        clip_ratio = (is_clipped * completion_mask).sum() / completion_mask.sum()
        self._metrics["clip_ratio"].append(self.accelerator.gather_for_metrics(clip_ratio).mean().item())
        
        self._metrics["logp_variance_reg"].append(self.accelerator.gather_for_metrics(reg_term).mean().item())
        if self.state.global_step % 3 == 0:
            torch.cuda.empty_cache()
        
        return loss
    def _calculate_logp_variance_regularization(self, seq_logps, rewards):
        """
        Calculate regularization term based on the variance of log probabilities
        for completions with reward value of 2.
        
        Args:
            seq_logps: Tensor of shape [batch_size] with sequence log probabilities
            rewards: Tensor of shape [batch_size] with reward values
            
        Returns:
            Regularization term (scalar tensor)
        """
        seq_logps = gather(seq_logps)
        rewards = gather(rewards)
        device = seq_logps.device
        batch_size = seq_logps.size(0)
        
        # If batch size is not divisible by num_generations, something is wrong
        if batch_size % self.num_generations != 0:
            print(f"Warning: Batch size {batch_size} not divisible by num_generations {self.num_generations}")
            return torch.tensor(0.0, device=device)
        
        num_prompts = batch_size // self.num_generations
        
        # Reshape to group by prompts: [num_prompts, num_generations]
        grouped_logps = seq_logps.view(num_prompts, self.num_generations)
        grouped_rewards = rewards.view(num_prompts, self.num_generations)
        
        # Calculate the regularization term for each prompt group
        reg_values = torch.zeros_like(grouped_logps, device=device)
        
        for i in range(num_prompts):
            # Find completions with reward == 2
            reward_mask = (grouped_rewards[i] == 2)
            reward_2_count = reward_mask.sum()
            
            # Only consider groups with at least 2 completions with reward 2
            if reward_2_count >= 2:
            
                # Get only the reward==2 logps using masked_select which is gradient-friendly
                reward_2_logps = torch.masked_select(grouped_logps[i], reward_mask)
                
                # Calculate mean without indexing
                mean_logp = reward_2_logps.mean()
                
                # Create a tensor same shape as grouped_logps[i] with the mean value
                # where the mask is True, and zeros elsewhere
                mean_tensor = reward_mask.float() * mean_logp
                
                # Calculate the deviation from mean for all values, will be zero where mask is False
                deviation = grouped_logps[i] - mean_tensor
                
                # Only keep deviations where mask is True
                deviation = deviation * reward_mask.float()
                
                # Store in the result tensor
                reg_values[i] = deviation
            
            else:
                reg_values[i] = 0
        # process_slice = slice(
        #         self.accelerator.process_index * num_prompts,
        #         (self.accelerator.process_index + 1) * num_prompts,
        #     )
        squared_reg = (reg_values ** 2)
        local_reg = squared_reg.mean()
        return local_reg

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys: Optional[list[str]] = None):
        # print("start prediction_step")
        inputs = self._prepare_inputs(inputs)
        # print("_prepare_inputs")
        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            loss = loss.mean().detach()
        return loss, None, None

   
    
    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}  # average the metrics

        # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
        # start with "eval_". We need to add the prefix "eval_" to the keys in `metrics` to match the format.
        if next(iter(logs.keys())).startswith("eval_"):
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics.clear()

 
    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the model.
            dataset_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        citation = textwrap.dedent(
            """\
            @article{zhihong2024deepseekmath,
                title        = {{DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}},
                author       = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
                year         = 2024,
                eprint       = {arXiv:2402.03300},
            }
            """
        )

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="GRPO",
            trainer_citation=citation,
            paper_title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
            paper_id="2402.03300",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))
    
    
    
    
    def update_temperature(self, current_step=None, current_epoch=None):
        """
        Update sampling temperature based on training progress.
        
        Args:
            current_step: Current training step (optional)
            current_epoch: Current training epoch (optional)
        """
        import math
        if not hasattr(self.args, "temperature_schedule"):
            # If no schedule, use the current temperature value
            if self.use_vllm and hasattr(self, "sampling_params"):
                new_temp = self.sampling_params.temperature
            elif hasattr(self, "generation_config"):
                new_temp = self.generation_config.temperature
            else:
                new_temp = self.args.temperature
        
        else:
            schedule = self.args.temperature_schedule
            if schedule is None:
                return
            
            # Get current step/epoch if not provided
            if current_step is None:
                current_step = self.state.global_step
            if current_epoch is None:
                current_epoch = self.state.epoch
            
            # Calculate new temperature based on schedule type
            if schedule["type"] == "linear":
                progress = min(1.0, current_step / schedule["total_steps"])
                start_temp = schedule["start"]
                end_temp = schedule["end"]
                new_temp = start_temp - (start_temp - end_temp) * progress
            
            elif schedule["type"] == "cosine":
                progress = min(1.0, current_step / schedule["total_steps"])
                start_temp = schedule["start"]
                end_temp = schedule["end"]
                cos_value = 0.5 * (1.0 + math.cos(math.pi * progress))
                new_temp = end_temp + (start_temp - end_temp) * cos_value
            
            elif schedule["type"] == "exponential":
                decay_rate = schedule["decay_rate"]
                new_temp = schedule["start"] * (decay_rate ** current_step)
                new_temp = max(new_temp, schedule.get("min", 0.1))
            
            elif schedule["type"] == "step":
                new_temp = schedule["start"]
                for step, temp in schedule["steps"]:
                    if current_step >= step:
                        new_temp = temp
            
            else:
                raise ValueError(f"Unknown temperature schedule type: {schedule['type']}")
        
            # Update temperature in both vLLM and generation config
            if self.use_vllm and hasattr(self, "sampling_params"):
                self.sampling_params.temperature = new_temp
                
            if hasattr(self, "generation_config"):
                self.generation_config.temperature = new_temp
            
            # Log the temperature change
            if self.accelerator.is_main_process:
                self._metrics["temperature"] = [new_temp]
                # print(f"Step {current_step}: Updated temperature to {new_temp:.4f}")
        # Log the temperature (append instead of overwrite)
        if self.accelerator.is_main_process:
            self._metrics["temperature"].append(new_temp)
            
            # # Only print changes if using a schedule
            # if hasattr(self.args, "temperature_schedule") and self.args.temperature_schedule is not None:
            #     print(f"Step {current_step}: Temperature is {new_temp:.4f}")
    '''
    def evaluation_loop(self, *args, **kwargs):
        original_temp = None
        original_num_generations = self.num_generations
    
        if self.use_vllm and hasattr(self, "sampling_params"):
            original_temp = self.sampling_params.temperature
            original_n = self.sampling_params.n
            self.sampling_params.temperature = self.args.eval_temperature
            self.sampling_params.n = self.args.eval_num_generations
        elif hasattr(self, "generation_config"):
            original_temp = self.generation_config.temperature
            self.generation_config.temperature = self.args.eval_temperature
    
        self.current_phase = "eval_"
        if self.accelerator.is_main_process:
            print(f"Evaluation: Using temperature {self.args.eval_temperature}")
            
        self.num_generations = self.args.eval_num_generations
    
        result = super().evaluation_loop(*args, **kwargs)
        
        self.num_generations = original_num_generations
        
        # Restore original temperature
        if self.use_vllm and hasattr(self, "sampling_params"):
            self.sampling_params.temperature = original_temp
            self.sampling_params.n = original_n
        elif hasattr(self, "generation_config"):
            self.generation_config.temperature = original_temp
            
    
        
        # Clean up memory after evaluation
        if hasattr(self, "_buffered_inputs"):
            for i in range(len(self._buffered_inputs)):
                self._buffered_inputs[i] = None
        self.current_phase = ""
        torch.cuda.empty_cache()
        return result 
    '''
    def evaluate(
        self,
        eval_dataset=None,
        ignore_keys=None,
        metric_key_prefix="eval",
    ):
        """
        Run evaluation and returns metrics.
        
        Overrides the Trainer.evaluate method to set evaluation-specific parameters
        before running the evaluation loop.
        """
        # Store original settings
        original_temperature = None
        original_num_generations = self.num_generations
        original_n = None
        
        # Set evaluation temperature
        if self.use_vllm and hasattr(self, "sampling_params"):
            original_temperature = self.sampling_params.temperature
            original_n = self.sampling_params.n
            self.sampling_params.temperature = self.args.eval_temperature
            self.sampling_params.n = self.args.eval_num_generations
        elif hasattr(self, "generation_config"):
            original_temperature = self.generation_config.temperature
            self.generation_config.temperature = self.args.eval_temperature
        
        # Set evaluation number of generations
        self.num_generations = self.args.eval_num_generations
        
        # Mark that we're in evaluation mode
        self.current_phase = "eval_"
        
        if self.accelerator.is_main_process:
            print(f"Evaluation: Using temperature {self.args.eval_temperature}, generations {self.args.eval_num_generations}")
        
        try:
            # Run standard evaluation using parent class
            metrics = super().evaluate(
                eval_dataset=eval_dataset,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix
            )
        finally:
            # Restore original settings
            self.num_generations = original_num_generations
            if self.use_vllm and hasattr(self, "sampling_params"):
                self.sampling_params.temperature = original_temperature
                if original_n is not None:
                    self.sampling_params.n = original_n
            elif hasattr(self, "generation_config") and original_temperature is not None:
                self.generation_config.temperature = original_temperature
            
            # Reset phase tracker
            self.current_phase = ""
            
            # Clean up memory after evaluation
            if hasattr(self, "_buffered_inputs"):
                for i in range(len(self._buffered_inputs)):
                    self._buffered_inputs[i] = None
            torch.cuda.empty_cache()
        
        return metrics          
            
    def training_step(self, model, inputs, num_items_in_batch):
        # Update temperature before each training step
        self.update_temperature()
        
        # Rest of training_step implementation
        loss = super().training_step(model, inputs, num_items_in_batch)
        return loss
