# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

import functools
import gc
import importlib
import json
import os
import re
import types
from typing import Callable, Dict, Type, Union

import torch
import transformers

# import torch.distributed as dist
# from torch.distributed import rpc
try:
    from auto_gptq.nn_modules.qlinear.qlinear_cuda import QuantLinear
    from auto_gptq.nn_modules.qlinear.qlinear_cuda_old import (
        QuantLinear as QuantLinearOld,
    )
except ImportError:

    class QuantLinear:
        pass

    class QuantLinearOld:
        pass


from safetensors import safe_open
from tqdm import tqdm
from transformers.modeling_utils import PretrainedConfig, PreTrainedModel

import pregated_moe
from pregated_moe.common import (
    ensure_local_deepseek_vl2_repo,
    parse_expert_type,
    resolve_model_architecture,
)
from pregated_moe.distributed import DistributedExpertExecutor
from pregated_moe.memory import (
    ExpertPredictor,
    ExpertPrefetcher,
    ExpertTracer,
    PregatedRouteController,
)
from pregated_moe.models import (
    DeepseekMoEBlock,
    Qwen3MoEBlock,
    SyncQwen3VLMoeSparseMoeBlock,
    SyncArcticMoeBlock,
    SyncGrokMoeBlock,
    SyncMixtralSparseMoeBlock,
    SyncNllbMoeSparseMLP,
    SyncSwitchTransformersSparseMLP,
)
from pregated_moe.runtime.compile import script_expert
from pregated_moe.runtime.hooks import *
from pregated_moe.utils import (
    ArcherConfig,
    parse_expert_dtype,
    parse_expert_id,
    parse_moe_param,
)
from pregated_moe.utils.arguments import (
    copy_args_to_device,
    copy_kwargs_to_device,
)

_prefetch_lib = None
# Alias for compatibility
prefetch_op = None


def _load_prefetch_lib():
    global _prefetch_lib, prefetch_op
    if _prefetch_lib is None:
        try:
            prefetch_lib = importlib.import_module("pregated_moe._store")
        except ImportError as exc:
            raise ImportError(
                "pregated_moe._store extension is required. Install with CUDA enabled."
            ) from exc

        _prefetch_lib = prefetch_lib
        prefetch_op = prefetch_lib

    return _prefetch_lib


def _get_text_like_config(config: PretrainedConfig) -> PretrainedConfig:
    if hasattr(config, "text_config"):
        return config.text_config
    if hasattr(config, "language_config"):
        return config.language_config
    return config


def _get_first_k_dense_replace(config: PretrainedConfig) -> int:
    return getattr(_get_text_like_config(config), "first_k_dense_replace", 0)


def _is_deepseek_vl2_direct_param(name: str) -> bool:
    return _get_deepseek_vl2_direct_param_name(name) is not None


def _get_deepseek_vl2_direct_param_name(name: str) -> str | None:
    candidates = ("image_newline", "view_seperator", "tile_indicators")
    for candidate in candidates:
        if name == candidate or name.endswith(f".{candidate}"):
            return candidate
    return None


def _get_deepseek_vl2_direct_param_cache_file(checkpoint_dir: str) -> str:
    return os.path.join(checkpoint_dir, "deepseek_vl2_direct_params.pt")


def _get_offload_metadata_file(checkpoint_dir: str) -> str:
    return os.path.join(checkpoint_dir, "offload_metadata.json")


def _is_deepseek_vl2_resident_name(name: str) -> bool:
    if _is_deepseek_vl2_direct_param(name):
        return True

    fixed_prefixes = (
        "vision",
        "projector",
        "aligner",
    )
    if any(name == prefix or name.startswith(f"{prefix}.") for prefix in fixed_prefixes):
        return True

    return False


def _is_deepseek_vl2_resident_module_name(module_name: str) -> bool:
    if module_name in {
        "vision",
        "projector",
        "aligner",
        "language.model.norm",
        "language.lm_head",
    }:
        return True

    if module_name.startswith(("vision.", "projector.", "aligner.", "language.lm_head.")):
        return True

    if module_name.startswith("language.model.layers."):
        if ".mlp.experts" in module_name and ".mlp.shared_experts" not in module_name:
            return False
        return True

    return False


def _is_pregated_gate_module_name(module_name: str) -> bool:
    return module_name.endswith(".gate")


def _get_deepseek_vl2_expected_direct_param_shapes(model):
    config = model.config
    tile_tag = getattr(model, "tile_tag", getattr(config, "tile_tag", None))
    expected_shapes = {}

    if tile_tag == "2D":
        n_embed = int(config.projector_config.n_embed)
        expected_shapes["image_newline"] = (n_embed,)
        expected_shapes["view_seperator"] = (n_embed,)
    elif tile_tag == "1D":
        tile_variants_num = len(config.candidate_resolutions)
        n_embed = int(config.aligner.params.n_embed)
        expected_shapes["tile_indicators"] = (tile_variants_num + 1, n_embed)
    else:
        raise RuntimeError(f"Unsupported DeepSeek-VL2 tile_tag: {tile_tag}")

    return expected_shapes


# class ArcherException(Exception):
#     pass


class OffloadEngine(object):
    param_id = 0
    request_id = 0
    # request_id_flag = False
    config = {}

    def __init__(self, capacity, config: PretrainedConfig):
        self.offload_exemption = set()
        self.expert_modules = []

        # self.model_create_counter = None

        self.ckpt_files = []

        self.expert_tracer = ExpertTracer(capacity, config)
        self.expert_predictor = ExpertPredictor(config)
        self.expert_predictor.add_tracer(self.expert_tracer)

        # self.expert_cache = ExpertCache(config)
        self.config = config

        self.quant_method = None
        self.force_rebuild_offload = False

    # def init_trace(self, trace_path: str):

    def init(
        self,
        cls: Type[PreTrainedModel],
        ar_config: Union[str, Dict, ArcherConfig],
    ):
        self.cls = cls
        self.param_id = 0
        self.request_id = 0
        self.name_id_map = {}
        self.tensor_id_map = {}
        self.registered_tensors = set()
        self.direct_param_tensors = {}
        self.forward_hooks = []
        self.backward_hooks = []
        self.force_rebuild_offload = False

        self.offload_set = set()
        self.active_offload_params = set()
        self.active_offload_buffers = set()

        if isinstance(ar_config, str):
            _archer_config = ArcherConfig.load_from_file(ar_config)
        elif isinstance(ar_config, dict):
            _archer_config = ArcherConfig.load_from_json(ar_config)
        elif isinstance(ar_config, ArcherConfig):
            _archer_config = ar_config
        else:
            raise ValueError(
                "ArcherConfig is not provided. Please provide a path to a config file or a dict."
            )

        # TODO: get trace from trace_path

        self.checkpoint = _archer_config.offload_path

        os.makedirs(self.checkpoint, exist_ok=True)

        # print("Waiting for distributed init ...")

        # local_rank = int(os.getenv('RANK', '0'))
        # world_size = int(os.getenv("WORLD_SIZE", '1'))

        # master_addr = os.getenv('MASTER_ADDR', 'localhost')
        # master_port = os.getenv('MASTER_PORT', '6000')

        # dist.init_process_group(
        #     backend="nccl",
        #     # _transports=["uv"], # https://discuss.pytorch.org/t/rpc-behavior-difference-between-pytorch-1-7-0-vs-1-9-0/124772/5
        #     rank=local_rank,
        #     world_size=world_size,
        #     group_name="pregated-moe",
        #     init_method= f"tcp://{master_addr}:{master_port}",
        # )
        # rpc.init_rpc(name=f"worker_{local_rank}",
        #              rank=local_rank,
        #              world_size=world_size)
        # print("Distributed init done")

        self.prefetch_lib = _load_prefetch_lib()

        # new_alloc = torch.cuda.memory.CUDAPluggableAllocator(
        #     self.prefetch_lib.__file__, "TorchAllocateDevice", "TorchFreeDevice"
        # )
        # # Swap the current allocator
        # torch.cuda.memory.change_current_allocator(new_alloc)

        self.archer_engine = self.prefetch_lib.prefetch_handle(
            self.checkpoint, _archer_config.device_memory_ratio
        )

        self.archer_config = _archer_config
        if _archer_config.trace_path is not None:
            self.expert_tracer.load_trace(_archer_config.trace_path)

        # # truncate self.perfect_cache_file
        # if (
        #     os.path.exists(_archer_config.perfect_cache_file)
        #     and _archer_config.save_cache
        # ):
        #     os.remove(_archer_config.perfect_cache_file)

        self.expert_executor = DistributedExpertExecutor(
            archer_config=_archer_config
        )
        # self.expert_prefetcher = ExpertPrefetcher(self.config)
        # self.device_map_manager = DeviceMapManager(archer_config=_archer_config)

        # self.expert_executor.set_device_map_manager(self.device_map_manager)
        # self.expert_prefetcher.set_device_map_manager(self.device_map_manager)
        # self.expert_prefetcher.set_archer_engine(self.archer_engine)

        return self

    def __enter__(self):
        def torch_index_select_decorator(orig_torch_index_select: Callable):
            @functools.wraps(orig_torch_index_select)
            def archer_torch_index_select(input, dim, index):
                return orig_torch_index_select(
                    input, dim, index.to(input.device)
                ).to("cuda:0")

            return archer_torch_index_select

        def apply_to_model_decorator(orig_apply_to_model: Callable) -> Callable:
            @functools.wraps(orig_apply_to_model)
            def archer_apply_to_model(cls, fn):
                for name, param in cls.named_parameters(recurse=True):
                    if name not in self.name_id_map:
                        continue
                    param.data = torch.zeros(
                        1,
                        dtype=param.dtype,
                        device=param.device,
                        pin_memory=True,
                    )

                for name, buffer in cls.named_buffers(recurse=True):
                    if name not in self.name_id_map:
                        continue
                    buffer.data = torch.zeros(
                        1,
                        dtype=buffer.dtype,
                        device=buffer.device,
                        pin_memory=True,
                    )

            return archer_apply_to_model

        def cast_classifier_decorator(
            orig_cast_classifier: Callable,
        ) -> Callable:
            @functools.wraps(orig_cast_classifier)
            def archer_cast_classifier(cls, *args, **kwargs):
                orig_data_ptr = cls.classifier.weight.data.data_ptr()
                if orig_data_ptr in self.offload_set:
                    self.offload_set.remove(
                        cls.classifier.weight.data.data_ptr()
                    )
                    orig_cast_classifier(cls, *args, **kwargs)
                    new_data_ptr = cls.classifier.weight.data.data_ptr()
                    self.offload_set.add(cls.classifier.weight.data.data_ptr())
                    self.archer_engine.update_tensor_map(
                        orig_data_ptr, new_data_ptr
                    )
                else:
                    orig_cast_classifier(cls, *args, **kwargs)
                    self.offload_set.add(cls.classifier.weight.data.data_ptr())

            return archer_cast_classifier

        # GPTQ Override
        QuantLinear._old_init = QuantLinear.__init__
        QuantLinear.__init__ = empty_param_init_decorator(QuantLinear.__init__)
        QuantLinearOld._old_init = QuantLinearOld.__init__
        QuantLinearOld.__init__ = empty_param_init_decorator(
            QuantLinearOld.__init__
        )

        # GPTQ Override
        QuantLinear._old_init = QuantLinear.__init__
        QuantLinear.__init__ = empty_param_init_decorator(QuantLinear.__init__)
        QuantLinearOld._old_init = QuantLinearOld.__init__
        QuantLinearOld.__init__ = empty_param_init_decorator(
            QuantLinearOld.__init__
        )

        self.cls._old_init = self.cls.__init__
        self.cls.__init__ = do_nothing_decorator(self.cls._old_init)
        # self.cls.config_class._old_from_pretrained = (
        #     self.cls.config_class.from_pretrained)
        # self.cls.config_class.from_pretrained = classmethod(
        #     config_decorator(self.cls.config_class.from_pretrained))
        # self.cls._old_load_pretrained_model = self.cls._load_pretrained_model
        # self.cls._load_pretrained_model = classmethod(
        #     load_pretrained_model_decorator(self.cls._load_pretrained_model))
        # transformers.modeling_utils.old_load_state_dict = (
        #     transformers.modeling_utils.load_state_dict)
        # transformers.modeling_utils.load_state_dict = load_state_dict
        torch.nn.modules.module.Module._old_apply = (
            torch.nn.modules.module.Module.apply
        )
        torch.nn.modules.module.Module.apply = apply_to_model_decorator(
            torch.nn.modules.module.Module._old_apply
        )

        torch._old_index_select = torch.index_select
        torch.index_select = torch_index_select_decorator(
            torch._old_index_select
        )
        torch.Tensor._old_index_select = torch.Tensor.index_select
        torch.Tensor.index_select = torch_index_select_decorator(
            torch.Tensor._old_index_select
        )

        self.cls._old_post_init = self.cls.post_init
        self.cls.post_init = do_nothing_decorator(self.cls._old_post_init)
        PreTrainedModel._old_post_init = PreTrainedModel.post_init
        PreTrainedModel.post_init = do_nothing_decorator(
            PreTrainedModel._old_post_init
        )

        activate_empty_init()

        transformers.models.switch_transformers.modeling_switch_transformers.SwitchTransformersTop1Router._old_cast_classifier = transformers.models.switch_transformers.modeling_switch_transformers.SwitchTransformersTop1Router._cast_classifier
        transformers.models.switch_transformers.modeling_switch_transformers.SwitchTransformersTop1Router._cast_classifier = cast_classifier_decorator(
            transformers.models.switch_transformers.modeling_switch_transformers.SwitchTransformersTop1Router._cast_classifier
        )

        transformers.models.switch_transformers.modeling_switch_transformers._old_sparse_mlp = transformers.models.switch_transformers.modeling_switch_transformers.SwitchTransformersSparseMLP
        transformers.models.switch_transformers.modeling_switch_transformers.SwitchTransformersSparseMLP = SyncSwitchTransformersSparseMLP
        transformers.models.nllb_moe.modeling_nllb_moe._old_sparse_mlp = (
            transformers.models.nllb_moe.modeling_nllb_moe.NllbMoeSparseMLP
        )
        transformers.models.nllb_moe.modeling_nllb_moe.NllbMoeSparseMLP = (
            SyncNllbMoeSparseMLP
        )
        transformers.models.mixtral.modeling_mixtral._old_sparse_mlp = (
            transformers.models.mixtral.modeling_mixtral.MixtralSparseMoeBlock
        )
        transformers.models.mixtral.modeling_mixtral.MixtralSparseMoeBlock = (
            SyncMixtralSparseMoeBlock
        )

        try:
            import transformers.models.qwen3_moe.modeling_qwen3_moe as qwen3_moe_module

            qwen3_moe_module._old_sparse_mlp = (
                qwen3_moe_module.Qwen3MoeSparseMoeBlock
            )
            qwen3_moe_module.Qwen3MoeSparseMoeBlock = Qwen3MoEBlock
            self._has_qwen3_moe = True
        except (ImportError, AttributeError):
            self._has_qwen3_moe = False

        try:
            import transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe as qwen3_vl_moe_module

            qwen3_vl_moe_module._old_sparse_mlp = (
                qwen3_vl_moe_module.Qwen3VLMoeTextSparseMoeBlock
            )
            qwen3_vl_moe_module.Qwen3VLMoeTextSparseMoeBlock = (
                SyncQwen3VLMoeSparseMoeBlock
            )
            self._has_qwen3_vl_moe = True
        except (ImportError, AttributeError):
            self._has_qwen3_vl_moe = False

        pregated_moe.models.modeling_grok.modeling_grok1._old_sparse_mlp = (
            pregated_moe.models.modeling_grok.MoeBlock
        )
        pregated_moe.models.modeling_grok.modeling_grok1.MoeBlock = (
            SyncGrokMoeBlock
        )

        pregated_moe.models.modeling_arctic._old_sparse_mlp = (
            pregated_moe.models.modeling_arctic.ArcticMoE
        )
        pregated_moe.models.modeling_arctic.modeling_arctic.ArcticMoE = (
            SyncArcticMoeBlock
        )

        current_arch = resolve_model_architecture(self.config)
        self._patched_deepseek_v2 = False
        self._patched_deepseek_v3 = False
        self._has_deepseek_vl2 = False

        # Keep text-model monkey patches scoped to the text-model paths.
        if current_arch == "deepseek":
            pregated_moe.models.modeling_deepseek_v2._old_sparse_mlp = (
                pregated_moe.models.modeling_deepseek_v2.DeepseekV2MoE
            )
            pregated_moe.models.modeling_deepseek_v2.modeling_deepseek.DeepseekV2MoE = (
                DeepseekMoEBlock
            )
            self._patched_deepseek_v2 = True
        elif current_arch == "deepseek_v3":
            pregated_moe.models.modeling_deepseek_v3._old_sparse_mlp = (
                pregated_moe.models.modeling_deepseek_v3.DeepseekV3MoE
            )
            pregated_moe.models.modeling_deepseek_v3.modeling_deepseek.DeepseekV3MoE = (
                DeepseekMoEBlock
            )
            self._patched_deepseek_v3 = True
        elif current_arch == "deepseek_vl2":
            try:
                ensure_local_deepseek_vl2_repo()
                import deepseek_vl2.models.modeling_deepseek as deepseek_vl2_modeling

                deepseek_vl2_modeling._old_sparse_mlp = (
                    deepseek_vl2_modeling.DeepseekV2MoE
                )
                deepseek_vl2_modeling.DeepseekV2MoE = DeepseekMoEBlock
                self._has_deepseek_vl2 = True
            except (ImportError, AttributeError) as exc:
                raise RuntimeError(
                    "DeepSeek-VL2 requires the local third_party/DeepSeek-VL2 repo "
                    "and a successful DeepseekV2MoE patch."
                ) from exc

        def from_pretrained_decorator(
            orig_from_pretrained: Callable,
        ) -> Callable:
            @functools.wraps(orig_from_pretrained)
            def archer_from_pretrained(cls, *args, **kwargs):
                # print("Creating model from scratch ...")

                name_id_map_file = os.path.join(
                    self.checkpoint, "name_id_map.json"
                )

                self.model_name = model_name = args[0]

                # if "arctic" in model_name:
                #     self.config = ArcticConfig.from_pretrained(*args, **kwargs)
                # else:
                #     self.config = AutoConfig.from_pretrained(*args, **kwargs)
                self.num_layers, self.num_experts, self.num_encoder_layers = (
                    parse_moe_param(self.config)
                )

                if resolve_model_architecture(self.config) in (
                    "qwen3",
                    "qwen3vlmoe",
                ):
                    text_config = _get_text_like_config(self.config)
                    self.prefetch_lib.init_moe_layer(
                        self.num_experts,
                        text_config.num_experts_per_tok,
                        self.archer_config.max_tokens,
                        text_config.hidden_size,
                        text_config.moe_intermediate_size,
                    )

                self.dtype_cls = self.config.torch_dtype
                if self.dtype_cls is None and (
                    hasattr(self.config, "text_config")
                    or hasattr(self.config, "language_config")
                ):
                    self.dtype_cls = _get_text_like_config(
                        self.config
                    ).torch_dtype
                if self.dtype_cls is None:
                    self.dtype_cls = torch.float16
                if getattr(self.config, "torch_dtype", None) is None:
                    self.config.torch_dtype = self.dtype_cls
                if hasattr(self.config, "text_config") or hasattr(
                    self.config, "language_config"
                ):
                    text_like_config = _get_text_like_config(self.config)
                    if getattr(text_like_config, "torch_dtype", None) is None:
                        text_like_config.torch_dtype = self.dtype_cls
                self.dtype = parse_expert_dtype(self.config)

                if self.config.model_type == "deepseek_v3":
                    self.dtype_cls = torch.float8_e4m3fn
                    self.dtype = 3

                reuse_offload, reuse_reason = self._resolve_offload_reuse(
                    name_id_map_file
                )
                if not reuse_offload:
                    self.force_rebuild_offload = True
                    print(
                        f"Creating model from scratch ... ({reuse_reason})",
                        flush=True,
                    )

                    self.cls.__init__ = self.cls._old_init

                    empty_state_dict = {}
                    self.name_id_map = {}
                    for ckpt in tqdm(
                        self.ckpt_files,
                        desc="Loading checkpoint files",
                        smoothing=0,
                    ):
                        state_dict = {}
                        if "safetensors" in ckpt:
                            with safe_open(
                                ckpt, framework="pt", device="cpu"
                            ) as f:
                                for k in f.keys():
                                    state_dict[k] = f.get_tensor(k)
                        else:
                            state_dict = torch.load(ckpt)

                        # convert all tensors in state_dict to self.dtype_cls
                        for k, v in state_dict.items():
                            try:
                                state_dict[k] = v.to(self.dtype_cls).to("cpu")
                            except Exception as e:
                                print(
                                    f"Error converting {k} (device={v.device}) to {self.dtype_cls} on CPU: {e}",
                                    flush=True,
                                )
                                raise

                        self._offload_state_dict(state_dict, empty_state_dict)

                        # print("Loading ckpt file", ckpt, flush=True)

                        del state_dict
                        gc.collect()
                        torch.cuda.empty_cache()

                    with open(name_id_map_file, "w") as f:
                        json.dump(self.name_id_map, f)
                    self._save_offload_metadata()
                    self._save_deepseek_vl2_direct_params()
                    self.force_rebuild_offload = False
                else:
                    self.force_rebuild_offload = False
                    print("Loading model from offload_path ...", flush=True)
                    self.cls.__init__ = self.cls._old_init
                    # load the name_id_map
                    with open(name_id_map_file, "r") as f:
                        self.name_id_map = json.load(f)
                    self._load_cached_deepseek_vl2_direct_params()

                # print(self.name_id_map, flush=True)

                # get max tensor id from the name_id_map
                # max_tensor_id = max(self.name_id_map.values())
                # self.model_create_counter = tqdm(
                #     total=max_tensor_id, desc="Model create"
                # )

                is_flash_attn_available = kwargs.get(
                    "is_flash_attn_available", False
                )
                attn_impl = kwargs.get("attn_implementation", None)
                if attn_impl is None:
                    attn_impl = (
                        "flash_attention_2"
                        if is_flash_attn_available
                        else "sdpa"
                    )
                # self.archer_prefetch.n_layer, self.archer_prefetch.n_expert, n_encoder_layers = parse_moe_param(self.config)
                model = cls._from_config(
                    self.config,
                    torch_dtype=self.dtype_cls
                    if self.config.model_type != "deepseek_v3"
                    else torch.bfloat16,
                    attn_implementation=attn_impl,
                )

                self._restore_deepseek_vl2_direct_params(model)

                # script_expert(
                #     self.checkpoint,
                #     self.config.model_type,
                #     self.config,
                # )

                if self.config.model_type == "deepseek_v3":
                    model = model.to(torch.float8_e4m3fn)

                # if (
                #     self.dtype_cls is torch.bfloat16
                #     or self.dtype_cls is torch.float16
                # ):
                #     model = cls._from_config(
                #         self.config,
                #         torch_dtype=self.dtype_cls,
                #         attn_implementation=(
                #             "flash_attention_2"
                #             if is_flash_attn_available
                #             else "eager"
                #         ),
                #     )
                # else:
                #     model = cls._from_config(self.config)

                base_model_prefix = model.base_model_prefix
                # model = model.to(self.dtype).to("cpu")

                # print("Model created with dtype", self.dtype, flush=True)
                # for name, param in model.named_parameters(recurse=False):
                #     print(name, param.dtype, flush=True)

                # print(self.config, flush=True)

                if hasattr(self.config, "quantization_config"):
                    self.quant_method = self.config.quantization_config[
                        "quant_method"
                    ]
                    self.config.quantization_config["use_exllama"] = False
                    self.config.quantization_config["disable_exllama"] = True
                    # print("Quantizing model ...", self.quant_method, flush=True)
                    if self.quant_method == "gptq":
                        from optimum.gptq import GPTQQuantizer

                        # print("Quantizing model with GPTQ ...", self.config.quantization_config, flush=True)
                        optimum_quantizer = GPTQQuantizer.from_dict(
                            self.config.quantization_config
                        )

                        model = optimum_quantizer.convert_model(model)

                self.expert_prefetcher = ExpertPrefetcher(self.config)
                self.expert_prefetcher.set_archer_engine(self.archer_engine)
                self.expert_dispatcher = self.prefetch_lib.expert_dispatcher(
                    self.num_experts,
                    self.num_layers,
                    self.dtype,
                    parse_expert_type(self.config),
                    self.archer_config.num_threads,
                )

                for name, param in model.named_parameters(recurse=True):
                    # remove base_model_prefix from self.name_id_map
                    if name.startswith(base_model_prefix):
                        name_without_prefix = name[
                            (len(base_model_prefix) + 1) :
                        ]
                        if name_without_prefix in self.name_id_map:
                            self.name_id_map[name] = self.name_id_map[
                                name_without_prefix
                            ]
                            self.name_id_map.pop(name_without_prefix)
                    param.ar_id = self.name_id_map.get(name, None)

                # the case for NLLB MoE (weight tying)
                if (
                    self.config.model_type == "nllb_moe"
                    and "lm_head.weight" not in self.name_id_map
                ):
                    print(
                        "lm_head.weight not in name_id_map, add it as embed_tokens"
                    )
                    self.name_id_map["lm_head.weight"] = 0
                    self.name_id_map["encoder.embed_tokens.weight"] = 0
                    self.name_id_map["decoder.embed_tokens.weight"] = 0

                    model.lm_head.weight.ar_id = 0
                    model.model.encoder.embed_tokens.weight.ar_id = 0
                    model.model.decoder.embed_tokens.weight.ar_id = 0

                # Rebuild this map from the exact dispatcher registration
                # order later in setup_archer_hooks. That guarantees the
                # layer ids used by pregated routing match the executor.
                self.expert_tensor_map = dict()
                self.expert_prefetcher.expert_tensor_map = {}

                # for deepseek, we need to set the expert_tensor_map for the model
                first_k_dense_replace = 0
                if "deepseek" in resolve_model_architecture(self.config):
                    first_k_dense_replace = _get_first_k_dense_replace(
                        self.config
                    )
                    self.expert_prefetcher.first_k_dense_replace = (
                        first_k_dense_replace
                    )
                # extracted_experts = []
                # for param_name, tensor_id in self.name_id_map.items():
                #     # extract encoder, digits from "encoder.layers.7.ffn.experts.expert_78.fc1.weight"
                #     result = re.findall(
                #         r"(encoder|decoder)\.[a-z]+\.(\d+).*expert_(\d+)",
                #         param_name)
                #     if result:
                #         layer_type, layer_id, expert_id = result[0]
                #         layer_id = int(layer_id)
                #         expert_id = int(expert_id)
                #         extracted_experts.append(
                #             (layer_type, layer_id, expert_id, tensor_id))
                # # remove duplicated experts
                # extracted_experts = list(set(extracted_experts))

                # extracted_experts = [(x[1], x[2],
                #                       x[3]) if x[0] == "encoder" else
                #                      (x[1] + 1000, x[2], x[3])
                #                      for x in extracted_experts]

                # # sort experts by first layer id, then expert id
                # extracted_experts = sorted(extracted_experts,
                #                            key=lambda x: (x[0], x[1]))
                # # transform to np.array
                # # self.archer_prefetch.extracted_experts = np.zeros(
                # #     (self.archer_prefetch.n_layer,
                # #      self.archer_prefetch.n_expert))

                # layer_idx = [x[0] for x in extracted_experts]
                # # make unique and sort
                # layer_idx = sorted(list(set(layer_idx)))

                self.expert_executor.set_expert_dispatcher(
                    self.expert_dispatcher
                )

                module_idx = 0
                self.expert_layer_modules = []
                for module_name, module in model.named_modules():
                    if (
                        isinstance(module, SyncNllbMoeSparseMLP)
                        or isinstance(module, SyncSwitchTransformersSparseMLP)
                        or isinstance(module, SyncNllbMoeSparseMLP)
                        or isinstance(module, SyncMixtralSparseMoeBlock)
                        or isinstance(module, SyncGrokMoeBlock)
                        or isinstance(module, SyncArcticMoeBlock)
                        or isinstance(module, DeepseekMoEBlock)
                        or isinstance(module, SyncQwen3VLMoeSparseMoeBlock)
                        or isinstance(module, Qwen3MoEBlock)
                    ):
                        # module.archer_prefetch = self.archer_prefetch
                        # module.archer_tracer = self.archer_tracer
                        module.archer_engine = self.archer_engine
                        module.archer_config = self.archer_config
                        # module.expert_dispatcher = self.expert_dispatcher
                        self.expert_modules.append(module)
                        module.expert_executor = self.expert_executor
                        module.expert_prefetcher = self.expert_prefetcher
                        module.expert_tracer = self.expert_tracer
                        module.expert_predictor = self.expert_predictor
                        module.expert_tensor_map = self.expert_tensor_map

                        module.lib = self.prefetch_lib
                        module.gate_tensor_ids = self._collect_module_tensor_ids(
                            f"{module_name}.gate",
                            module.gate,
                        )

                        self.expert_layer_modules.append(module)

                        # module_experts = [
                        #     x for x in extracted_experts
                        #     if x[0] == layer_idx[module_idx]
                        # ]

                        # module.expert_tensor_ids = {
                        #     x[1]: x[2]
                        #     for x in module_experts
                        # }
                        # expert_tensor_ids = [
                        #     item for item in module.expert_tensor_ids.items()
                        # ]
                        # #sort by k and v
                        # expert_tensor_ids = sorted(expert_tensor_ids,
                        #                            key=lambda x: (x[0], x[1]))
                        # # self.archer_prefetch.extracted_experts[module_idx] = [
                        # #     x[1] for x in expert_tensor_ids
                        # # ]
                        module.layer_id = module_idx + first_k_dense_replace

                        module_idx += 1

                self.setup_archer_hooks(model)
                self.pregated_route = PregatedRouteController(
                    self.archer_engine,
                    self.expert_tensor_map,
                    self.expert_layer_modules,
                )
                for module in self.expert_layer_modules:
                    module.pregated_route = self.pregated_route
                self._patch_qwen3_vl_vision(model)
                # print("OffloadEngine init done, rank", dist.get_rank(), flush=True)
                return model

            return archer_from_pretrained

        self.cls._old_from_pretrained = self.cls.from_pretrained
        self.cls.from_pretrained = classmethod(
            from_pretrained_decorator(self.cls.from_pretrained)
        )

        return self

    # clean up initialization hooks
    def __exit__(self, exc_type, exc_value, traceback):
        # GPTQ Override
        QuantLinear.__init__ = QuantLinear._old_init
        QuantLinearOld.__init__ = QuantLinearOld._old_init

        self.cls.__init__ = self.cls._old_init
        self.cls.from_pretrained = self.cls._old_from_pretrained
        torch.nn.modules.module.Module.apply = (
            torch.nn.modules.module.Module._old_apply
        )
        torch.index_select = torch._old_index_select
        torch.Tensor.index_select = torch.Tensor._old_index_select

        self.cls.post_init = self.cls._old_post_init
        PreTrainedModel.post_init = PreTrainedModel._old_post_init

        deactivate_empty_init()

    def get_topology(self, model):
        name_lst = []
        ret_dict = {}

        # print("Getting topology ...", self.name_id_map)

        def _resolve_stored_name_with_digits(name):
            layer_matches = list(re.finditer(r"\.(\d+)\.", name))
            if layer_matches:
                last_match = layer_matches[-1]
                return name[: last_match.end(1)]

            digit_matches = list(re.finditer(r"\d", name))
            if digit_matches:
                return name[: digit_matches[-1].start() + 1]

            components = name.rsplit(".", 1)
            return components[0]

        # for name in model.state_dict().keys():
        for name, _ in model.named_parameters(recurse=True):
            if (
                resolve_model_architecture(self.config) == "deepseek_vl2"
                and _is_deepseek_vl2_resident_name(name)
            ):
                continue
            match = re.search(r"\d+", name)
            if name not in self.name_id_map:
                print("param not in self.name_id_map", name)
                continue
            if match:
                if "expert" in name and "shared_experts" not in name:
                    match = re.match(r"(.*experts)", name)
                    assert match, "Not correct expert name!"
                    stored_name = match.group(1)
                    components = name.split(".")
                    # Use negative indexing to get the component between the last third and second dot
                    expert_name = components[-3]
                    if stored_name in name_lst:
                        if expert_name in ret_dict[stored_name]:
                            ret_dict[stored_name][expert_name].append(
                                self.name_id_map[name]
                            )
                        else:
                            ret_dict[stored_name][expert_name] = [
                                self.name_id_map[name]
                            ]
                    else:
                        ret_dict[stored_name] = {
                            expert_name: [self.name_id_map[name]]
                        }
                        name_lst.append(stored_name)

                else:
                    stored_name = _resolve_stored_name_with_digits(name)

                    if stored_name in name_lst:
                        ret_dict[stored_name][0].append(self.name_id_map[name])
                    else:
                        ret_dict[stored_name] = [[self.name_id_map[name]]]
                        name_lst.append(stored_name)

            else:
                components = name.rsplit(".", 1)
                stored_name = components[0]

                if stored_name in name_lst:
                    ret_dict[stored_name][0].append(self.name_id_map[name])
                else:
                    ret_dict[stored_name] = [[self.name_id_map[name]]]
                    name_lst.append(stored_name)

        for name, _ in model.named_buffers(recurse=True):
            if (
                resolve_model_architecture(self.config) == "deepseek_vl2"
                and _is_deepseek_vl2_resident_name(name)
            ):
                continue
            match = re.search(r"\d+", name)
            if name not in self.name_id_map:
                # print("buffer not in self.name_id_map", name)
                continue
            if match:
                if "expert" in name and "shared_experts" not in name:
                    match = re.match(r"(.*experts)", name)
                    assert match, "Not correct expert name!"
                    stored_name = match.group(1)
                    components = name.split(".")
                    # Use negative indexing to get the component between the last third and second dot
                    expert_name = components[-3]
                    if stored_name in name_lst:
                        if expert_name in ret_dict[stored_name]:
                            ret_dict[stored_name][expert_name].append(
                                self.name_id_map[name]
                            )
                        else:
                            ret_dict[stored_name][expert_name] = [
                                self.name_id_map[name]
                            ]
                    else:
                        ret_dict[stored_name] = {
                            expert_name: [self.name_id_map[name]]
                        }
                        name_lst.append(stored_name)

                else:
                    stored_name = _resolve_stored_name_with_digits(name)

                    if stored_name in name_lst:
                        ret_dict[stored_name][0].append(self.name_id_map[name])
                    else:
                        ret_dict[stored_name] = [[self.name_id_map[name]]]
                        name_lst.append(stored_name)
            else:
                components = name.rsplit(".", 1)
                stored_name = components[0]

                if stored_name in name_lst:
                    ret_dict[stored_name][0].append(self.name_id_map[name])
                else:
                    ret_dict[stored_name] = [[self.name_id_map[name]]]
                    name_lst.append(stored_name)

        for i in ret_dict.keys():
            if isinstance(ret_dict[i], dict):
                ret_dict[i] = list(ret_dict[i].values())

        topology = list(ret_dict.items())
        return topology

    def setup_archer_hooks(self, model):
        arch = resolve_model_architecture(self.config)
        for module_name, module in model.named_modules():
            keep_resident = (
                isinstance(module, torch.nn.Embedding)
                or module_name == "lm_head"
                or module_name.endswith("rotary_emb")
                or "RotaryEmbedding" in module.__class__.__name__
            )
            if _is_pregated_gate_module_name(module_name):
                keep_resident = True
            if arch == "deepseek_vl2" and _is_deepseek_vl2_resident_module_name(
                module_name
            ):
                keep_resident = True
            if keep_resident:
                setattr(module, "_archer_keep_resident", True)

        for name, param in model.named_parameters(recurse=True):
            if arch == "deepseek_vl2" and _is_deepseek_vl2_resident_name(name):
                continue
            if name not in self.name_id_map:
                continue
            self.archer_engine.register(param.data, self.name_id_map[name])
            self.offload_set.add(param.data.data_ptr())

            if "shared" in name:
                self.offload_exemption.add(param.data.data_ptr())

        for name, buffer in model.named_buffers(recurse=True):
            if arch == "deepseek_vl2" and _is_deepseek_vl2_resident_name(name):
                continue
            if name not in self.name_id_map:
                continue
            self.archer_engine.register(buffer.data, self.name_id_map[name])
            self.offload_set.add(buffer.data.data_ptr())

        topo = self.get_topology(model)
        self.archer_engine.set_topology(topo)

        @torch.no_grad()
        def _pre_forward_input_hook(module, input, kwargs, device, tensors):
            # print("pre_forward_input_hook", device, input, tensors)
            self.archer_engine.fetch_tensors(self.request_id, tensors)
            new_args = copy_args_to_device(device, input)
            new_kwargs = copy_kwargs_to_device(device, kwargs)
            return new_args, new_kwargs

        @torch.no_grad()
        def _post_forward_output_hook(module, input, output, device, tensors):
            if isinstance(output, tuple):
                new_args = copy_args_to_device(device, output)
            elif isinstance(output, dict):
                new_args = copy_kwargs_to_device(device, output)
            else:
                new_args = output.to(device)
            return new_args

        def gen_args_hook(
            key, input_device_index, output_device_index, tensors
        ):
            keys = key.split(".")
            # print(keys)
            m = model
            for k in keys:
                if k.isdigit():
                    m = m[int(k)]
                else:
                    m = getattr(m, k)

            m.register_forward_pre_hook(
                functools.partial(
                    _pre_forward_input_hook,
                    device=input_device_index,
                    tensors=tensors,
                ),
                prepend=True,
                with_kwargs=True,
            )
            if "lm_head" in key:
                m.register_forward_hook(
                    functools.partial(
                        _post_forward_output_hook, device=0, tensors=tensors
                    ),
                    prepend=False,
                )

        expert_layer_id = 0
        if "deepseek" in self.model_name:
            expert_layer_id = _get_first_k_dense_replace(self.config)

        self.expert_tensor_map = {}
        output_device_index = None
        for key, tensors in topo:
            # print(key, tensors)
            if "shared" in key or "lm_head" in key:
                key = key.split(".")[0]
                output_device_index = 0

            if "expert" in key:
                for expert_idx, expert_tensors in enumerate(tensors):
                    expert_key = (
                        f"{key}.expert_{expert_idx}"
                        if self.config.model_type != "mixtral"
                        and self.config.model_type != "grok-1"
                        and self.config.model_type != "arctic"
                        and self.config.model_type != "deepseek_v2"
                        and self.config.model_type != "deepseek_v3"
                        else f"{key}.{expert_idx}"
                    )
                    input_device_index = (
                        self.archer_engine.get_node_default_device(
                            expert_tensors
                        )
                    )
                    # gen_args_hook(
                    #     expert_key,
                    #     input_device_index,
                    #     output_device_index,
                    #     expert_tensors,
                    # )

                    self.expert_dispatcher.register_expert(
                        expert_layer_id,
                        expert_idx,
                        expert_tensors,
                        os.path.join(self.checkpoint, f"expert.pt"),
                    )
                    if len(expert_tensors) == 0:
                        raise RuntimeError(
                            f"Empty expert tensor list for layer={expert_layer_id}, expert={expert_idx}"
                        )
                    self.expert_tensor_map[(expert_layer_id, expert_idx)] = int(
                        expert_tensors[0]
                    )
                expert_layer_id += 1
            else:
                input_device_index = self.archer_engine.get_node_default_device(
                    tensors[0]
                )
                gen_args_hook(
                    key, input_device_index, output_device_index, tensors[0]
                )
                output_device_index = input_device_index

        self.expert_prefetcher.expert_tensor_map = self.expert_tensor_map

        # @torch.no_grad()
        # def request_id_hook(module, *args):
        #     self.request_id_flag = False
        #     # self.archer_tracer.clear_request_id()
        #     # self.archer_prefetch.clear_request()

        # model.register_forward_hook(request_id_hook)

        enable_recursive_hooks_env = os.getenv(
            "MOE_INFINITY_ENABLE_RECURSIVE_HOOKS"
        )
        if enable_recursive_hooks_env is None:
            enable_recursive_hooks = True
        else:
            enable_recursive_hooks = enable_recursive_hooks_env.lower() in {
                "1",
                "true",
                "yes",
                "on",
            }

        if enable_recursive_hooks:
            self._register_hooks_recursively(model)
        else:
            print(
                f"[Pregated-MoE] Skip recursive module hooks for architecture `{arch}`"
            )

    def _generate_param_id(self):
        param_id = self.param_id
        self.param_id += 1
        return param_id

    def _generate_request_id(self):
        request_id = self.request_id
        self.request_id += 1
        return request_id

    def _collect_module_tensor_ids(self, module_name, module):
        if module is None:
            return []

        tensor_ids = []
        for name, _ in module.named_parameters(recurse=True):
            full_name = f"{module_name}.{name}" if name else module_name
            tensor_id = self.name_id_map.get(full_name)
            if tensor_id is not None:
                tensor_ids.append(int(tensor_id))

        for name, _ in module.named_buffers(recurse=True):
            full_name = f"{module_name}.{name}" if name else module_name
            tensor_id = self.name_id_map.get(full_name)
            if tensor_id is not None:
                tensor_ids.append(int(tensor_id))

        return tensor_ids

    def _build_offload_metadata(self):
        return {
            "model_name_or_path": str(self.model_name),
            "architecture": resolve_model_architecture(self.config),
            "model_type": getattr(self.config, "model_type", None),
            "dtype": str(self.dtype_cls),
            "num_layers": int(self.num_layers),
            "num_experts": int(self.num_experts),
            "num_encoder_layers": int(self.num_encoder_layers),
            "first_k_dense_replace": int(
                _get_first_k_dense_replace(self.config)
            ),
            "checkpoint_files": list(self.ckpt_files),
        }

    def _load_offload_metadata(self):
        metadata_file = _get_offload_metadata_file(self.checkpoint)
        if not os.path.exists(metadata_file):
            return None
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
        return metadata if isinstance(metadata, dict) else None

    def _save_offload_metadata(self):
        metadata_file = _get_offload_metadata_file(self.checkpoint)
        with open(metadata_file, "w") as f:
            json.dump(self._build_offload_metadata(), f, indent=2)

    def _resolve_offload_reuse(self, name_id_map_file):
        if not self.archer_engine.is_tensor_index_initialized():
            return False, "tensor index is not initialized"
        if not os.path.exists(name_id_map_file):
            return False, "name_id_map.json is missing"

        cached_metadata = self._load_offload_metadata()
        expected_metadata = self._build_offload_metadata()
        if cached_metadata is None:
            return False, "offload metadata is missing"
        if cached_metadata != expected_metadata:
            return False, "offload metadata does not match current model"
        return True, None

    def _load_cached_deepseek_vl2_direct_params(self):
        cache_file = _get_deepseek_vl2_direct_param_cache_file(
            self.checkpoint
        )
        if not os.path.exists(cache_file):
            return

        cached_tensors = torch.load(cache_file, map_location="cpu")
        if not isinstance(cached_tensors, dict):
            return

        for name, tensor in cached_tensors.items():
            direct_param_name = _get_deepseek_vl2_direct_param_name(name)
            if direct_param_name is None or not isinstance(
                tensor, torch.Tensor
            ):
                continue
            self.direct_param_tensors[direct_param_name] = tensor.cpu()

    def _save_deepseek_vl2_direct_params(self):
        if not self.direct_param_tensors:
            return

        cache_file = _get_deepseek_vl2_direct_param_cache_file(
            self.checkpoint
        )
        torch.save(self.direct_param_tensors, cache_file)

    def _load_deepseek_vl2_direct_params_from_checkpoints(self):
        missing_names = {
            "image_newline",
            "view_seperator",
            "tile_indicators",
        } - set(self.direct_param_tensors)
        if not missing_names:
            return

        for ckpt in self.ckpt_files:
            if not missing_names:
                break

            if "safetensors" in ckpt:
                with safe_open(ckpt, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        direct_param_name = (
                            _get_deepseek_vl2_direct_param_name(key)
                        )
                        if direct_param_name not in missing_names:
                            continue
                        tensor = f.get_tensor(key)
                        self.direct_param_tensors[direct_param_name] = (
                            tensor.to(self.dtype_cls).cpu()
                        )
                        missing_names.remove(direct_param_name)
            else:
                state_dict = torch.load(ckpt, map_location="cpu")
                try:
                    for key, tensor in state_dict.items():
                        direct_param_name = (
                            _get_deepseek_vl2_direct_param_name(key)
                        )
                        if direct_param_name not in missing_names:
                            continue
                        self.direct_param_tensors[direct_param_name] = (
                            tensor.to(self.dtype_cls).cpu()
                        )
                        missing_names.remove(direct_param_name)
                finally:
                    del state_dict

        self._save_deepseek_vl2_direct_params()

    def _ensure_deepseek_vl2_direct_params_loaded(self):
        if resolve_model_architecture(self.config) != "deepseek_vl2":
            return
        if self.direct_param_tensors:
            return

        self._load_cached_deepseek_vl2_direct_params()
        if self.direct_param_tensors:
            return

        self._load_deepseek_vl2_direct_params_from_checkpoints()

    def _restore_deepseek_vl2_direct_params(self, model):
        if resolve_model_architecture(self.config) != "deepseek_vl2":
            return

        self._ensure_deepseek_vl2_direct_params_loaded()
        expected_shapes = _get_deepseek_vl2_expected_direct_param_shapes(model)
        loaded_tensors = dict(self.direct_param_tensors)

        for name, expected_shape in expected_shapes.items():
            if name not in loaded_tensors:
                raise RuntimeError(
                    f"Missing DeepSeek-VL2 direct parameter `{name}` in checkpoints."
                )

            param = getattr(model, name, None)
            if not isinstance(param, torch.nn.Parameter):
                raise RuntimeError(
                    f"DeepSeek-VL2 model is missing direct parameter `{name}`."
                )

            tensor = loaded_tensors[name]
            if tuple(tensor.shape) != tuple(expected_shape):
                raise RuntimeError(
                    f"DeepSeek-VL2 direct parameter `{name}` has checkpoint shape "
                    f"{tuple(tensor.shape)}, expected {tuple(expected_shape)}."
                )

            param.data = tensor.to(device=param.device, dtype=param.dtype)

    def _offload_state_dict(
        self,
        state_dict: Dict[str, torch.Tensor],
        empty_state_dict: Dict[str, torch.Tensor],
    ) -> None:
        param_names = list(state_dict.keys())

        for param_name in param_names:
            tensor = state_dict[param_name]
            direct_param_name = None

            if (
                resolve_model_architecture(self.config) == "deepseek_vl2"
                and (
                    direct_param_name := _get_deepseek_vl2_direct_param_name(
                        param_name
                    )
                )
                is not None
            ):
                self.direct_param_tensors[direct_param_name] = tensor.cpu()
                continue

            if (
                "mlp.experts.gate_up_proj" in param_name
                and tensor.dim() == 3
            ):
                num_experts = tensor.shape[0]
                moe_inter = tensor.shape[2] // 2
                for expert_id in range(num_experts):
                    expert_tensor = tensor[expert_id]
                    gate = expert_tensor[:, :moe_inter].t()
                    up = expert_tensor[:, moe_inter:].t()
                    gate_name = param_name.replace(
                        "experts.gate_up_proj",
                        f"experts.{expert_id}.gate_proj.weight",
                    )
                    up_name = param_name.replace(
                        "experts.gate_up_proj",
                        f"experts.{expert_id}.up_proj.weight",
                    )
                    self.name_id_map[gate_name] = self._generate_param_id()
                    if self.force_rebuild_offload or not self.archer_engine.is_tensor_offloaded(
                        self.name_id_map[gate_name]
                    ):
                        self.archer_engine.offload(
                            gate.contiguous(),
                            self.name_id_map[gate_name],
                        )
                    self.name_id_map[up_name] = self._generate_param_id()
                    if self.force_rebuild_offload or not self.archer_engine.is_tensor_offloaded(
                        self.name_id_map[up_name]
                    ):
                        self.archer_engine.offload(
                            up.contiguous(),
                            self.name_id_map[up_name],
                        )
                continue

            if (
                "mlp.experts.down_proj" in param_name
                and tensor.dim() == 3
            ):
                num_experts = tensor.shape[0]
                for expert_id in range(num_experts):
                    down_weight = tensor[expert_id].t()
                    down_name = param_name.replace(
                        "experts.down_proj",
                        f"experts.{expert_id}.down_proj.weight",
                    )
                    self.name_id_map[down_name] = self._generate_param_id()
                    if self.force_rebuild_offload or not self.archer_engine.is_tensor_offloaded(
                        self.name_id_map[down_name]
                    ):
                        self.archer_engine.offload(
                            down_weight.contiguous(),
                            self.name_id_map[down_name],
                        )
                continue

            self.name_id_map[param_name] = self._generate_param_id()
            if self.force_rebuild_offload or not self.archer_engine.is_tensor_offloaded(
                self.name_id_map[param_name]
            ):
                self.archer_engine.offload(tensor, self.name_id_map[param_name])

        gc.collect()
        torch.cuda.empty_cache()

    def _register_hooks_recursively(self, module, count=[0]):
        my_count = count[0]
        module.id = my_count

        for child in module.children():
            count[0] = count[0] + 1
            self._register_hooks_recursively(child, count=count)

        @torch.no_grad()
        def _pre_forward_module_hook(module, args, kwargs):
            # if self.request_id_flag == False:
            #     self.request_id_flag = True
            #     # print(kwargs, args, type(module))

            #     request_id = self._generate_request_id()
            #     # self.archer_tracer.set_request_id(request_id)
            #     # self.archer_prefetch.set_request(request_id)

            device_list = []

            for name, param in module.named_parameters(recurse=False):
                param_key = id(param)
                if param.data.data_ptr() not in self.offload_set:
                    num_devices = torch.cuda.device_count()
                    param.data = param.data.to(f"cuda:{num_devices-1}")
                    device_list.append(param.data.device)
                    continue

                self.offload_set.remove(param.data.data_ptr())
                self.archer_engine.begin(self.request_id, param)
                self.active_offload_params.add(param_key)
                self.offload_set.add(param.data.data_ptr())

                device_list.append(param.data.device)

            for name, buf in module.named_buffers(recurse=False):
                buffer_key = id(buf)
                if buf.data.data_ptr() not in self.offload_set:
                    buf.data = buf.data.to("cuda:0")
                    device_list.append(buf.data.device)
                    continue

                # print("offload buffer", name, buf.data.data_ptr())

                self.offload_set.remove(buf.data_ptr())
                self.archer_engine.begin(self.request_id, buf)
                self.active_offload_buffers.add(buffer_key)
                # buf = buf.to(self.dtype)
                self.offload_set.add(buf.data_ptr())

                device_list.append(buf.data.device)

            if device_list:
                device = device_list[0]
                new_args = copy_args_to_device(device, args)
                new_kwargs = copy_kwargs_to_device(device, kwargs)
                return new_args, new_kwargs

        @torch.no_grad()
        def _post_forward_module_hook(module, input, output):
            keep_module_resident = bool(
                getattr(module, "_archer_keep_resident", False)
            )
            device_list = []
            param_not_offload = set()
            for param in module.parameters(recurse=False):
                param_key = id(param)
                if param.data.data_ptr() not in self.offload_set:
                    param_not_offload.add(param.data.data_ptr())
                    continue
                if param_key not in self.active_offload_params:
                    continue

                if keep_module_resident:
                    self.active_offload_params.discard(param_key)
                    self.offload_set.discard(param.data.data_ptr())
                    self.offload_exemption.add(param.data.data_ptr())
                    continue

                self.offload_set.remove(param.data.data_ptr())
                self.archer_engine.end(self.request_id, param)
                self.active_offload_params.discard(param_key)
                self.offload_set.add(param.data.data_ptr())

                device_list.append(param.data.device)

            for buf in module.buffers(recurse=False):
                buffer_key = id(buf)
                if buf.data_ptr() not in self.offload_set:
                    continue
                if buffer_key not in self.active_offload_buffers:
                    continue

                if keep_module_resident:
                    self.active_offload_buffers.discard(buffer_key)
                    self.offload_set.discard(buf.data_ptr())
                    self.offload_exemption.add(buf.data_ptr())
                    continue

                self.offload_set.remove(buf.data_ptr())
                self.archer_engine.end(self.request_id, buf)
                self.active_offload_buffers.discard(buffer_key)
                self.offload_set.add(buf.data_ptr())

                device_list.append(buf.device)

            if param_not_offload:
                if isinstance(output, torch.Tensor):
                    return output.to(torch.device("cuda:0"))

                if isinstance(output, dict):
                    return copy_kwargs_to_device(torch.device("cuda:0"), output)

                return copy_args_to_device(torch.device("cuda:0"), output)

        # Pre forward hook
        self.forward_hooks.append(
            module.register_forward_pre_hook(
                _pre_forward_module_hook, with_kwargs=True
            )
        )

        # Post forward hook
        self.forward_hooks.append(
            module.register_forward_hook(_post_forward_module_hook)
        )

    def _patch_qwen3_vl_vision(self, model):
        visual = getattr(getattr(model, "model", None), "visual", None)
        if visual is None:
            visual = getattr(model, "visual", None)
        if visual is None or not hasattr(
            visual, "fast_pos_embed_interpolate"
        ):
            return

        def _fixed_fast_pos_embed_interpolate(self, grid_thw):
            grid_ts = grid_thw[:, 0]
            grid_hs = grid_thw[:, 1]
            grid_ws = grid_thw[:, 2]

            idx_list = [[] for _ in range(4)]
            weight_list = [[] for _ in range(4)]

            for t, h, w in zip(grid_ts, grid_hs, grid_ws):
                h_idxs = torch.linspace(
                    0, self.num_grid_per_side - 1, h
                )
                w_idxs = torch.linspace(
                    0, self.num_grid_per_side - 1, w
                )

                h_floor = h_idxs.int()
                w_floor = w_idxs.int()
                h_ceil = (h_idxs.int() + 1).clip(
                    max=self.num_grid_per_side - 1
                )
                w_ceil = (w_idxs.int() + 1).clip(
                    max=self.num_grid_per_side - 1
                )

                dh = h_idxs - h_floor
                dw = w_idxs - w_floor

                base_h = h_floor * self.num_grid_per_side
                base_h_ceil = h_ceil * self.num_grid_per_side

                indices = [
                    (base_h[None].T + w_floor[None]).flatten(),
                    (base_h[None].T + w_ceil[None]).flatten(),
                    (base_h_ceil[None].T + w_floor[None]).flatten(),
                    (base_h_ceil[None].T + w_ceil[None]).flatten(),
                ]
                weights = [
                    ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
                    ((1 - dh)[None].T * dw[None]).flatten(),
                    (dh[None].T * (1 - dw)[None]).flatten(),
                    (dh[None].T * dw[None]).flatten(),
                ]
                for idx in range(4):
                    idx_list[idx].extend(indices[idx].tolist())
                    weight_list[idx].extend(weights[idx].tolist())

            idx_tensor = torch.tensor(idx_list, dtype=torch.long)
            weight_tensor = torch.tensor(
                weight_list,
                dtype=self.pos_embed.weight.dtype,
            )
            pos_embeds = self.pos_embed(
                idx_tensor.to(self.pos_embed.weight.device)
            )
            device = pos_embeds.device
            pos_embeds = pos_embeds * weight_tensor.to(device)[:, :, None]
            patch_pos_embeds = (
                pos_embeds[0]
                + pos_embeds[1]
                + pos_embeds[2]
                + pos_embeds[3]
            )

            patch_pos_embeds = patch_pos_embeds.split(
                [h * w for h, w in zip(grid_hs, grid_ws)]
            )

            patch_pos_embeds_permute = []
            merge_size = self.config.spatial_merge_size
            for pos_embed, t, h, w in zip(
                patch_pos_embeds, grid_ts, grid_hs, grid_ws
            ):
                pos_embed = pos_embed.repeat(t, 1)
                pos_embed = (
                    pos_embed.view(
                        t,
                        h // merge_size,
                        merge_size,
                        w // merge_size,
                        merge_size,
                        -1,
                    )
                    .permute(0, 1, 3, 2, 4, 5)
                    .flatten(0, 4)
                )
                patch_pos_embeds_permute.append(pos_embed)

            return torch.cat(patch_pos_embeds_permute)

        visual.fast_pos_embed_interpolate = types.MethodType(
            _fixed_fast_pos_embed_interpolate,
            visual,
        )

    # clean runtime hooks
    def clean_up(self):
        transformers.models.switch_transformers.modeling_switch_transformers.SwitchTransformersTop1Router._cast_classifier = transformers.models.switch_transformers.modeling_switch_transformers.SwitchTransformersTop1Router._old_cast_classifier
        transformers.models.switch_transformers.modeling_switch_transformers.SwitchTransformersSparseMLP = transformers.models.switch_transformers.modeling_switch_transformers._old_sparse_mlp

        transformers.models.nllb_moe.modeling_nllb_moe.NllbMoeSparseMLP = (
            transformers.models.nllb_moe.modeling_nllb_moe._old_sparse_mlp
        )

        transformers.models.mixtral.modeling_mixtral.MixtralSparseMoeBlock = (
            transformers.models.mixtral.modeling_mixtral._old_sparse_mlp
        )

        pregated_moe.models.modeling_grok.modeling_grok1.MoeBlock = (
            pregated_moe.models.modeling_grok.modeling_grok1._old_sparse_mlp
        )

        pregated_moe.models.modeling_arctic.modeling_arctic.ArcticMoE = (
            pregated_moe.models.modeling_arctic._old_sparse_mlp
        )

        if getattr(self, "_patched_deepseek_v2", False):
            pregated_moe.models.modeling_deepseek_v2.modeling_deepseek.DeepseekV2MoE = (
                pregated_moe.models.modeling_deepseek_v2._old_sparse_mlp
            )
        if getattr(self, "_patched_deepseek_v3", False):
            pregated_moe.models.modeling_deepseek_v3.modeling_deepseek.DeepseekV3MoE = (
                pregated_moe.models.modeling_deepseek_v3._old_sparse_mlp
            )

        if getattr(self, "_has_deepseek_vl2", False):
            import deepseek_vl2.models.modeling_deepseek as deepseek_vl2_modeling

            deepseek_vl2_modeling.DeepseekV2MoE = (
                deepseek_vl2_modeling._old_sparse_mlp
            )

        if getattr(self, "_has_qwen3_vl_moe", False):
            import transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe as qwen3_vl_moe_module

            qwen3_vl_moe_module.Qwen3VLMoeTextSparseMoeBlock = (
                qwen3_vl_moe_module._old_sparse_mlp
            )

        if getattr(self, "_has_qwen3_moe", False):
            import transformers.models.qwen3_moe.modeling_qwen3_moe as qwen3_moe_module

            qwen3_moe_module.Qwen3MoeSparseMoeBlock = (
                qwen3_moe_module._old_sparse_mlp
            )
