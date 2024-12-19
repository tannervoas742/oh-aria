# ==============================================================================
# Copyright (c) Intel [2024]
#
# Modifications:
# - adapt_transformers_to_gaudi
# - Loosen attention settings
#
# Original Copyright:
# Copyright (c) 2024 Rhymes AI. All rights reserved.
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# ==============================================================================

import logging

from transformers.configuration_utils import PretrainedConfig

from .moe_lm import AriaMoELMConfig
from .vision_encoder import AriaVisionConfig
from .utils import is_torch_hpu_available

if is_torch_hpu_available():
    from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi
    adapt_transformers_to_gaudi()
    IS_HPU = True
else:
    IS_HPU = False

logger = logging.getLogger(__name__)


# adapted from transformers.models.llava.configuration_llava.LlavaConfig
class AriaConfig(PretrainedConfig):
    """
    Configuration class for Aria model.

    This class handles the configuration for both vision and text components of the Aria model,
    as well as additional parameters for image token handling and projector mapping.

    Args:
        vision_config (AriaVisionConfig or dict): Configuration for the vision component.
        text_config (AriaMoELMConfig or dict): Configuration for the text component.
        projector_patch_to_query_dict (dict): Mapping of patch sizes to query dimensions.
        ignore_index (int): Index to ignore in loss calculation.
        image_token_index (int): Index used to represent image tokens.
        **kwargs: Additional keyword arguments passed to the parent class.

    Attributes:
        model_type (str): Type of the model, set to "aria".
        is_composition (bool): Whether the model is a composition of multiple components.
        ignore_index (int): Index to ignore in loss calculation.
        image_token_index (int): Index used to represent image tokens.
        projector_patch_to_query_dict (dict): Mapping of patch sizes to query dimensions.
        vision_config (AriaVisionConfig): Configuration for the vision component.
        text_config (AriaMoELMConfig): Configuration for the text component.
    """

    model_type = "aria"
    is_composition = False

    def __init__(
        self,
        vision_config=AriaVisionConfig(),
        text_config=AriaMoELMConfig(),
        projector_patch_to_query_dict={
            1225: 128,
            4900: 256,
        },
        ignore_index=-100,
        image_token_index=32000,
        tie_word_embeddings=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.ignore_index = ignore_index
        self.image_token_index = image_token_index
        self.tie_word_embeddings = tie_word_embeddings
        attn_implementation = kwargs.pop("attn_implementation", None)

        # Set the default attention implementation to flash_attention_2 if not specified
        if attn_implementation is not None:
            self._attn_implementation = attn_implementation

        # Convert the keys and values of projector_patch_to_query_dict to integers
        # This ensures consistency even if they were provided as strings
        self.projector_patch_to_query_dict = {
            int(k): int(v) for k, v in projector_patch_to_query_dict.items()
        }

        if isinstance(vision_config, dict) and "model_type" in vision_config:
            vision_config = AriaVisionConfig(**vision_config)
            if attn_implementation is not None:
                vision_config._attn_implementation = attn_implementation

        self.vision_config = vision_config

        if isinstance(text_config, dict) and "model_type" in text_config:
            text_config = AriaMoELMConfig(**text_config)
            if attn_implementation is not None:
                text_config._attn_implementation = attn_implementation

        self.text_config = text_config

        # This is needed for the static kv cache
        self.num_hidden_layers = self.text_config.num_hidden_layers
