import torch
from transformers import PreTrainedTokenizerFast
from transformers import BartForConditionalGeneration

tokenizer = PreTrainedTokenizerFast.from_pretrained('/home/user/.cache/huggingface/hub/models--gogamza--kobart-summarization/snapshots/31f181b155a0ad74bd93bd90ee04310ff72691f4')
model = BartForConditionalGeneration.from_pretrained('/home/user/.cache/huggingface/hub/models--gogamza--kobart-summarization/snapshots/31f181b155a0ad74bd93bd90ee04310ff72691f4')

def sum(text, sum_rate):
    len_text = len(text)
    len_sum = int(sum_rate * len_text)

    raw_input_ids = tokenizer.encode(text)
    input_ids = [tokenizer.bos_token_id] + raw_input_ids + [tokenizer.eos_token_id]

    summary_ids = model.generate(torch.tensor([input_ids]), \
                                max_length=len_sum, \
                                do_sample=True, \
                                top_k=30, top_p=0.3, \
                                num_beams=4, temperature=0.4)
    return tokenizer.decode(summary_ids.squeeze().tolist(), \
                            skip_special_tokens=True).strip()