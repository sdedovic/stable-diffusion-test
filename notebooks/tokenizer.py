# Forked from:
#   https://github.com/huggingface/transformers/blob/main/src/transformers/tokenization_utils.py
#   https://github.com/huggingface/transformers/blob/main/src/transformers/tokenization_utils_base.py
#   https://github.com/huggingface/transformers/blob/main/src/transformers/models/clip/tokenization_clip.py
#   https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py


# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team.
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
"""
 Tokenization classes for python tokenizers. For fast tokenizers (provided by HuggingFace's tokenizers library) see
 tokenization_utils_fast.py
"""
from keras_cv.models.generative.stable_diffusion.clip_tokenizer import SimpleTokenizer
from trie import Trie


class ExtendableTokenizer:
    def __init__(self):
        self.tokenizer = SimpleTokenizer()
        self.last_idx = 49407

        self.added_tokens_encoder = {}
        self.added_tokens_decoder = {}

        self.trie = Trie()

    def add_token(self, token) -> int:
        if self.added_tokens_encoder.get(token) is None:
            token_id = self.last_idx + 1
            self.added_tokens_encoder[token] = token_id
            self.added_tokens_decoder[token_id] = token

            self.trie.add(token)
            self.last_idx += 1
            return token_id
        else:
            return self.added_tokens_encoder.get(token)

    def get_added_vocab(self):
        """
        Returns the added tokens in the vocabulary as a dictionary of token to index.
        Returns:
            `Dict[str, int]`: The added tokens.
        """
        return self.added_tokens_encoder.keys()

    def encode(self, text):
        start = None
        end = None
        out = []
        for split in self.trie.split(text):
            if self.added_tokens_encoder.get(split) is None:
                encoding = self.tokenizer.encode(split)
                if start is None or end is None:
                    start = encoding[0]
                    end = encoding[-1]
                out.extend(encoding[1:-1])
            else:
                out.append(self.added_tokens_encoder.get(split))
        return [start] + out + [end]
