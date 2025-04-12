from torch import Tensor, nn
from transformers import (CLIPTextModel, CLIPTokenizer, T5EncoderModel,
                          T5Tokenizer)


class HFEmbedder(nn.Module):
    def __init__(self,version: str, max_length: int, local_path, **hf_kwargs ):
        super().__init__()
        self.is_clip = version.startswith("text_encoder")
        self.max_length = max_length
        self.output_key = "pooler_output" if self.is_clip else "last_hidden_state"

        if self.is_clip:
            self.tokenizer: T5Tokenizer = CLIPTokenizer.from_pretrained(local_path, max_length=max_length, subfolder="tokenizer")
            self.hf_module: T5EncoderModel = CLIPTextModel.from_pretrained(local_path,subfolder='text_encoder' , **hf_kwargs)
        else:
            self.tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained(local_path, max_length=max_length, ignore_mismatched_sizes=True, subfolder="tokenizer_2")
            self.hf_module: T5EncoderModel = T5EncoderModel.from_pretrained(local_path,ignore_mismatched_sizes=True,subfolder='text_encoder_2' , **hf_kwargs)

        self.hf_module = self.hf_module.eval().requires_grad_(False)

    def forward(self, text: list[str]) -> Tensor:
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=False,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )

        outputs = self.hf_module(
            input_ids=batch_encoding["input_ids"].to(self.hf_module.device),
            attention_mask=None,
            output_hidden_states=False,
        )
        return outputs[self.output_key]
