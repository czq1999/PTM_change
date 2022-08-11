import torch
# from src.transformers import BertTokenizer, BertModel, AutoModel, AutoTokenizer, T5ForConditionalGeneration
from torch import nn

# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model1 = BertModel.from_pretrained("bert-base-uncased")
from PTM.transformers.src.transformers import AutoTokenizer, T5ForConditionalGeneration, T5Config

text = "[CLS] Replace me by any text you'd like.[SEP] I love China"
# encoded_input = tokenizer(text, return_tensors='pt')
# output = model1(**encoded_input)

tokenizer = AutoTokenizer.from_pretrained('Salesforce/codet5-base')
# special_token_dict = {'additional_special_tokens': ['<LINES_START>', '<LINES_END>']}
# num_added_toks = tokenizer.add_special_tokens(special_token_dict)
source_ids = tokenizer.encode(text, max_length=256, padding='max_length', truncation=True)
source_ids = torch.LongTensor([source_ids for _ in range(10)])
target_ids = source_ids
source_mask = source_ids.ne(tokenizer.pad_token_id)
target_mask = target_ids.ne(tokenizer.pad_token_id)
model_config = T5Config.from_pretrained('Salesforce/codet5-base')
model = T5ForConditionalGeneration(config=model_config)
model_dict = model.state_dict()
model_dict.update(torch.load('./pytorch_model.bin'))
model.load_state_dict(model_dict)
# model2 = T5ForConditionalGeneration.from_pretrained('Salesforce/codet5-base')
ast_encoding = torch.FloatTensor(10, 256, 768)
# 训练阶段
outputs = model(input_ids=source_ids, labels=target_ids, attention_mask=source_mask, decoder_attention_mask=target_mask,
                ast_encoding=ast_encoding)
# 测试阶段
# outputs = model.generate(input_ids=source_ids, attention_mask=source_mask,
#                          use_cache=True, num_beams=5, max_length=128,
#                          ast_encoding=ast_encoding)

print()
