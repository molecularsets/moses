import torch.optim as optim
import tqdm


class JTreeTrainer:
    def __init__(self, config):
        self.config = config

    def fit(self, model, data):
        def get_params():
            return (p for p in model.parameters() if p.requires_grad)

        model.train()

        n_epoch = self.config.num_epochs

        optimizer = optim.Adam(get_params(), lr=self.config.lr)
        for epoch in range(n_epoch):
            if epoch < self.config.kl_start:
                kl_w = 0
            else:
                kl_w = self.config.kl_w

            word_acc, topo_acc, assm_acc, steo_acc, all_kl = 0, 0, 0, 0, 0
            with tqdm.tqdm(data) as train_dataloader:
                train_dataloader.set_description('Train (epoch #{})'.format(epoch))

                for it, batch in enumerate(train_dataloader):
                    model.zero_grad()
                    loss, kl_div, wacc, tacc, sacc, dacc = model(batch, kl_w)
                    loss.backward()
                    optimizer.step()

                    word_acc += wacc
                    topo_acc += tacc
                    assm_acc += sacc
                    steo_acc += dacc
                    all_kl += kl_div

                    postfix = {'kl': all_kl / (it + 1),
                               'word': word_acc / (it + 1) * 100,
                               'topo': topo_acc / (it + 1) * 100,
                               'assm': assm_acc / (it + 1) * 100,
                               'steo': steo_acc / (it + 1) * 100}

                    train_dataloader.set_postfix(postfix)
