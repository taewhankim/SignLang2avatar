import torch, os
from transformers.models.gpt2.tokenization_gpt2_fast import GPT2TokenizerFast
from transformers.models.gpt2.modeling_gpt2 import GPT2PreTrainedModel, GPT2Model
from transformers.tokenization_utils_base import BatchEncoding
from transformers import PreTrainedTokenizerFast
from seq2seq_translator.core.models import generate_square_subsequent_mask

tokenizer = PreTrainedTokenizerFast.from_pretrained(
    "skt/kogpt2-base-v2",
    bos_token="</s>",
    eos_token="</s>",
    unk_token="<unk>",
    pad_token="<pad>",
    mask_token="<mask>",
    cache_dir="/mnt/workdir/sign/seq2seq_translator/core/gpt/pretrained",
)

# model: GPT2PreTrainedModel = AutoModelForPreTraining.from_pretrained(
#     "skt/kogpt2-base-v2",
#     cache_dir="/mnt/workdir/sign/seq2seq_translator/core/gpt/pretrained",
# )
# model: GPT2Model = GPT2Model.from_pretrained(
#     "skt/kogpt2-base-v2",
#     cache_dir="/mnt/workdir/sign/seq2seq_translator/core/gpt/pretrained",
# )

tokenizer: GPT2TokenizerFast
print(tokenizer.bos_token_id)
print(tokenizer.eos_token_id)
exit()
tokenizer.bos_token
tokenizer.bos_token_id
input_ids = tokenizer(
    [
        "</s>감자 고구마 피자 양파 토끼고기</s>",
        "</s>니가 가서 과자좀 가져와라</s>",
        "</s>김치 프라이 자판기 토스트 쀍쮋</s>",
        "</s>감자 고구마 피자 양파 토끼고기</s>",
        "</s>니가 가서 과자좀 가져와라</s>",
        "</s>김치 프라이 자판기 토스트 쀍쮋fdsafdsafdsafdsafdsaㄹㅇㄴㅁㄹㅇㄴㅁㄹㅇㅁㄴㄹㅁㄴ</s>",
        "</s>감자 고구마 피자 양파 토끼고기</s>",
        "</s>니가 가서 과자좀 가져와라</s>",
        "</s>김치 프라이 자판기 토스트 쀍쮋</s>",
        "</s>감자 고구마 피자 양파 토끼고기</s>",
        "</s>니가 가서 과자좀 가져와라</s>",
        "</s>김치 프라이 자판기 토스트 쀍쮋</s>",
        "</s>감자 고구마 피자 양파 토끼고기</s>",
        "</s>니가 가서 과자좀 가져와라</s>",
        "</s>김치 프라이 자판기 토스트 쀍쮋</s>",
        "</s>감자 고구마 피자 양파 토끼고기</s>",
        "</s>니가 가서 과자좀 가져와라</s>",
        "</s>김치 프라이 자판기 토스트 쀍쮋</s>",
        "</s>감자 고구마 피자 양파 토끼고기</s>",
        "</s>니가 가서 과자좀 가져와라</s>",
        "</s>김치 프라이 자판기 토스트 쀍쮋</s>",
        "</s>감자 고구마 피자 양파 토끼고기</s>",
        "</s>니가 가서 과자좀 가져와라</s>",
        "</s>김치 프라이 자판기 토스트 쀍쮋</s>",
        "</s>감자 고구마 피자 양파 토끼고기</s>",
        "</s>니가 가서 과자좀 가져와라</s>",
        "</s>김치 프라이 자판기 토스트 쀍쮋</s>",
        "</s>감자 고구마 피자 양파 토끼고기</s>",
        "</s>니가 가서 과자좀 가져와라</s>",
        "</s>김치 프라이 자판기 토스트 쀍쮋</s>",
        "</s>감자 고구마 피자 양파 토끼고기</s>",
        "</s>니가 가서 과자좀 가져와라</s>",
        "</s>김치 프라이 자판기 토스트 쀍쮋</s>",
        "</s>감자 고구마 피자 양파 토끼고기</s>",
        "</s>니가 가서 과자좀 가져와라</s>",
        "</s>김치 프라이 자판기 토스트 쀍쮋</s>",
    ],
    padding=True,
    return_special_tokens_mask=True,
    return_tensors="pt",
)  # type:BatchEncoding

attn_mask = input_ids["attention_mask"]
token_type_ids = input_ids["token_type_ids"]
input_id = input_ids["input_ids"]


# inp = torch.randint(0, 51200, (32, 64))

mask = generate_square_subsequent_mask(
    sz=input_id.shape[1], device="cpu", fill_masked=0, fill_non_masked=1
)

# emb = torch.randn((3, len(input_ids[0]), 768))

print(input_id.shape)
print(mask.shape)
print(attn_mask.shape)
print(token_type_ids.shape)
r = model.forward(
    input_ids=input_id,
    attention_mask=mask,
    token_type_ids=token_type_ids,
)


# print(r)