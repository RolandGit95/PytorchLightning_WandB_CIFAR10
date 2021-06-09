import torch
import torch.nn as nn
import pytorch_lightning as pl

class ImageModule(pl.LightningModule): 
    def __init__(self, model, cfg): 
        super(ImageModule, self).__init__() 
        self.cfg = cfg
        self.lr = cfg.lr
        
        self.model = model
        
        self.loss = nn.NLLLoss()
        self.val_loss = nn.NLLLoss()

    def forward(self, X):
        return self.model(X)
  
    def configure_optimizers(self): 
        optimizer = torch.optim.AdamW(self.parameters(), 
                                      lr=self.lr, 
                                      weight_decay=self.cfg.weight_decay)
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               mode='min',
                                                               factor=0.3,
                                                               patience=4,
                                                               min_lr=1e-6,
                                                               verbose=True)
        
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}
        
    def training_step(self, train_batch, batch_idx): 
        X, y = train_batch 
        y_pred = self.forward(X)
        loss = self.loss(y_pred, y)
        #print(loss)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss 
  
    def validation_step(self, valid_batch, batch_idx): 
        X, y = valid_batch 
        y_pred = self.forward(X)
        
        acc = torch.sum(torch.argmax(y_pred, dim=1)==y)/len(y)
        val_loss = self.val_loss(y_pred, y)
        
        self.log('acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log('val_loss', val_loss, on_step=True, on_epoch=True, prog_bar=True)
