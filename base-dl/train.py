import torch

from rbdl.train import TrainerBase
from rbdl.metrics import psnr

class Trainer(TrainerBase):
    def loss_step(self, inputs, targets):
        self.model.train()
        self.inputs = inputs.to(self.cfg.device)
        self.targets = targets.to(self.cfg.device)

        self.outputs = self.model(self.inputs)

        loss = self.loss_func(self.targets, self.outputs)
        metric = psnr(self.targets, self.outputs)

        loss.backward()
        self.optimizer.step() 
        self.optimizer.zero_grad()

        return loss, metric

    def validate(self):
        self.model.eval()  
        with torch.no_grad():
            total_loss = 0
            total_metric = 0
            for self.batch_index, [self.inputs, self.targets] in enumerate(self.valid_loader):
                self.inputs = self.inputs.to(self.cfg.device)
                self.targets = self.targets.to(self.cfg.device)
                self.outputs = self.model(self.inputs)
                loss = self.loss_func(self.targets, self.outputs)
                total_loss += loss
                total_metric += psnr(self.targets, self.outputs)

            mean_loss = total_loss / len(self.valid_loader)
            mean_metric = total_metric / len(self.valid_loader)

            return mean_loss, mean_metric 
