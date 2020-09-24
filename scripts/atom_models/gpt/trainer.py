import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
import numpy
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence

from moses.interfaces import MosesTrainer
#from moses.utils import CharVocab, Logger
from moses.utils import OneHotVocab, Logger


class GPT2MolTrainer(MosesTrainer):

    def __init__(self, config):
        self.config = config
    
    def get_vocabulary(self, data):
        return OneHotVocab.from_data(data)

    #def get_vocabulary(self, data):
    #    return CharVocab.from_data(data)

    def get_collate_fn(self, model):
        device = self.get_collate_device(model)

        def collate(data):
            data.sort(key=len, reverse=True)
            #print("Collate data size ", len(data), " ", data)
            tensors = [model.string2tensor(string, device=device)
                       for string in data]
            
            #padding = [model.pad]*(self.seq_len - len(X))
            #X.extend(padding)
            #This hack uses longest sequence in a batch by default, rewrite to use fixed max sequence length
            tensors = pad_sequence([t for t in tensors], batch_first=True, padding_value=model.pad)

            return tensors

        return collate

    def _train_epoch(self, model, tqdm_data, optimizer=None):
        if optimizer is None:
            model.eval()
        else:
            model.train()

        postfix = { 'loss' : 0,
                    'running_loss' : 0, }
        #print("In _train_epoch LEN tqdm_data ", len(tqdm_data))
        
        #print("Trainer model vocab len, pad ", len(model.vocabulary), " ", model.pad)
        for i, input_batch in enumerate(tqdm_data):
            #input_batch = tuple(data.to(model.device) for data in input_batch)
            #input_batch = torch.t(input_batch.cuda()) # (T,B)
            input_batch = input_batch.to(model.device) # (T,B)
            #[print("Data shape ", data.shape) for data in input_batch]
            # Forward
            if optimizer is not None:
              optimizer.zero_grad()
            lm_logits = model(input_batch)[0] 

            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = input_batch[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), 
                            shift_labels.view(-1),
                            ignore_index=model.pad)
            #loss = F.nll_loss(loss.view(-1, len(vocab)),
            '''
            print("Ground truth string ")
            #[print(model.tensor2string(t))
            #           for t in input_batch]
            print("Ground truth string contigous ")
            [print(t.shape, " ", t)
                       for t in input_batch.contiguous()]
            
            print("Ground truth string contigous view ")
            [print(t.shape, " ", t)
                       for t in input_batch.contiguous().view(-1)]

            #print("Reconstructed string ")
            #[print(model.tensor2string(t))
            #Reshape to from k*V to V for each batch  
            #[print(t.shape, " ", t)
            #           for t in output.view(-1,len(model.vocabulary))]
            '''
        #for i, (prevs, nexts, lens) in enumerate(tqdm_data):
        #    prevs = prevs.to(model.device)
        #    nexts = nexts.to(model.device)
        #    lens = lens.to(model.device)

        #    outputs, _, _ = model(prevs, lens)

        #    loss = criterion(outputs.view(-1, outputs.shape[-1]), nexts.view(-1))

            if optimizer is not None:
                #optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            postfix['loss'] = loss.item()
            postfix['running_loss'] += (loss.item() - postfix['running_loss']) / (i + 1)
            tqdm_data.set_postfix(postfix)

        postfix['mode'] = 'Eval' if optimizer is None else 'Train'
        return postfix

    def _train(self, model, train_loader, val_loader=None, logger=None):
        def get_params():
            return (p for p in model.parameters() if p.requires_grad)

        device = model.device
        #criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(get_params(), lr=self.config.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, self.config.step_size, self.config.gamma)

        model.zero_grad()

        for epoch in range(self.config.train_epochs):
            scheduler.step()
            tqdm_data = tqdm(train_loader, desc='Train (epoch #{})'.format(epoch))
            postfix = self._train_epoch(model, tqdm_data, optimizer)
            if logger is not None:
                logger.append(postfix)
                logger.save(self.config.log_file)

            if val_loader is not None:
                tqdm_data = tqdm(val_loader, desc='Validation (epoch #{})'.format(epoch))
                postfix = self._train_epoch(model, tqdm_data)
                if logger is not None:
                    logger.append(postfix)
                    logger.save(self.config.log_file)

            if (self.config.model_save is not None) and (epoch % self.config.save_frequency == 0):
                model = model.to('cpu')
                torch.save(model.state_dict(), self.config.model_save[:-3]+'_{0:03d}.pt'.format(epoch))
                model = model.to(device)


    def fit(self, model, train_data, val_data=None):
        logger = Logger() if self.config.log_file is not None else None

        train_loader = self.get_dataloader(model, train_data, shuffle=True)
        val_loader = None if val_data is None else self.get_dataloader(model, val_data, shuffle=False)

        self._train(model, train_loader, val_loader, logger)
        return model
