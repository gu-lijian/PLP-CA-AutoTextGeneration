from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import BertTokenizer
import torch
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import numpy as np

docs_new = "anything testing"
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
output_model_file = "./topicModel/models/model_file.bin"
output_config_file = "./topicModel/models/config_file.bin"
output_vocab_file = "./topicModel/models/vocab_file.bin"
tokenizer = BertTokenizer(output_vocab_file, do_lower_case=True)

def getMaxSentenceLength(sentences):
  max_len = 0
  for sent in sentences:
    input_ids = tokenizer.encode(sent, add_special_tokens=True)
    max_len = max(max_len, len(input_ids))
  return max_len

def bertEmbedding(sentences):
  maxLen = getMaxSentenceLength(sentences)
  input_ids = []
  attention_masks = []
  for sent in sentences:
    encoded_dict = tokenizer.encode_plus(
      sent, 
      add_special_tokens = True, 
      max_length = 64, 
      pad_to_max_length = True, 
      truncation=True, 
      return_attention_mask = True, 
      return_tensors = 'pt',
      )
    input_ids.append(encoded_dict['input_ids'])
    attention_masks.append(encoded_dict['attention_mask'])
  input_ids = torch.cat(input_ids, dim=0)
  attention_masks = torch.cat(attention_masks, dim=0)
  return input_ids, attention_masks

def predicTopicLabel(input_ids, attention_masks):
  config = BertConfig.from_json_file(output_config_file)
  model = BertForSequenceClassification(config)
  state_dict = torch.load(output_model_file)
  model.load_state_dict(state_dict)
  model.eval()
  print('input_ids : ', input_ids)
  # input_ids = torch.cat(input_ids, dim=0)
  # attention_masks = torch.cat(attention_masks, dim=0)
  batch_size = 32
  prediction_data = TensorDataset(input_ids, attention_masks)
  prediction_sampler = SequentialSampler(prediction_data)
  prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)
  for batch in prediction_dataloader:
    b_input_ids, b_input_mask = batch
    outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
  predict = []
  for prediction in outputs:  # predict is one by one, so the length of probabilities=1
    pred_label = np.argmax(prediction.detach().numpy())
    predict.append(pred_label)
  if predict[0] == 0:
    result_label = 'economy'
  elif predict[0] == 1:
    result_label = 'vaccine'
  else:
    result_label = 'quarantine'
  return result_label

def MainFunction(sentence):
  sentences = [sentence]
  input_ids, attention_masks = bertEmbedding(sentences)
  result_label = predicTopicLabel(input_ids, attention_masks)
  return result_label



