import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from transformers import GPT2Config, GPT2LMHeadModel
import numpy as np 
import math


class GPT2Mol(nn.Module):

    def __init__(self, vocab, config):
        super(GPT2Mol, self).__init__()

        self.vocabulary = vocab
        # Special symbols
        for ss in ('bos', 'eos', 'unk', 'pad'):
            setattr(self, ss, getattr(vocab, ss))

        self.hidden_size = config.hidden
        self.n_layers = config.n_layers # num of enc/dec layers
        self.n_heads = config.num_attention_heads # 4 default
        self.dropout = config.dropout
        
        # Word embeddings layer
        n_vocab, d_emb = len(vocab), vocab.vectors.size(1)
        self.input_size= self.output_size = n_vocab
        #print("GPT2 vocab size, d_emb pad idx ", n_vocab, " ", d_emb, " ", self.pad)
        config = GPT2Config(len(vocab),n_positions=100 ,n_ctx=100,
                         #n_embd=len(vocab),
                         summary_use_proj=False,
                         bos_token_id=vocab.bos, eos_token_id=vocab.eos)
        print("CONFIG ", config)
        self.gpt2 = GPT2LMHeadModel(config)

    def forward(self,src):
        return self.gpt2(src)

    @property
    def device(self):
        return next(self.parameters()).device

    def string2tensor(self, string, device='model'):
        ids = self.vocabulary.string2ids(string, add_bos=True, add_eos=True)
        tensor = torch.tensor(ids, dtype=torch.long,
                              device=self.device if device == 'model' else device)

        return tensor

    def tensor2string(self, tensor):
        ids = tensor.tolist()
        string = self.vocabulary.ids2string(ids, rem_bos=True, rem_eos=True)
        return string

    def sample(self,batch_size, max_length=100):

      with torch.no_grad():

        new_smiles_list = [
             torch.tensor(self.vocabulary.pad, dtype=torch.long, device=self.device).repeat(max_length + 2)
                  for _ in range(batch_size)]

        len_smiles_list = [1 for _ in range(batch_size)]

        for i in range(batch_size):
          new_smiles_list[i][0] = self.vocabulary.bos
          #len_smiles_list = [1 for _ in range(1)]
          len_smiles_list = [1]*batch_size
          starts = [torch.tensor([self.vocabulary.bos], dtype=torch.long, device=self.device)
                      for _ in range(1)]
          starts = torch.tensor(starts, dtype=torch.long, device=self.device).unsqueeze(0)
          input, past = starts, None
          for j in range(max_length):
            logits, past = self.gpt2(input, past=past)
            input = torch.multinomial(F.softmax(logits[:, -1]),1)
            new_char = input.item()
            if(new_char == self.vocabulary.eos or new_char == self.vocabulary.pad):
              break
            else:
              new_smiles_list[i][j] = new_char
              len_smiles_list[i] = len_smiles_list[i] + 1

        new_smiles_list = [new_smiles_list[i][:l] for i, l in enumerate(len_smiles_list)]
        return [self.tensor2string(t) for t in new_smiles_list]

