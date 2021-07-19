'''
Arquivo com as classe do LRScheduler e EarlyStopping.
File with LRScheduler and EarlyStopping classes.  
'''

import torch

class LRScheduler():
    '''
    Learning rate scheduler. If the validation loss does not decrease for the 
    given number of `patience` epochs, then the learning rate will decrease by
    by given `factor`.
    '''
    def __init__(self, optimizer, patience=5, min_lr=1e-6, factor=0.5):
        '''
        new_lr = old_lr * factor
        :param optimizer: the optimizer we are using
        :param patience: how many epochs to wait before updating the lr
        :param min_lr: least lr value to reduce to while updating
        :param factor: factor by which the lr should be updated
        '''
        self.optimizer = optimizer
        self.patience = patience
        self.min_lr = min_lr
        self.factor = factor
        
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                patience = self.patience,
                factor = self.factor,
                min_lr = self.min_lr,
                verbose=True)
                
    def __call__(self, metrics):
        self.lr_scheduler.step(metrics)
        
    def state_dict(self):
        return self.lr_scheduler.state_dict()  
        
    def load_state_dict(self, state_dict):
        self.lr_scheduler.load_state_dict(state_dict)  
        
class EarlyStopping():
    
    '''
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    '''
    def __init__(self, patience, min_delta=0):
        '''
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        '''
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, metrics):
        if self.best_loss == None:
            self.best_loss = metrics
        elif self.best_loss - metrics > self.min_delta:
            self.best_loss = metrics
        elif self.best_loss - metrics < self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print("INFO: Early stopping")
                self.early_stop = True
                
    def state_dict(self):
        ''' Returns early stopping state '''
        return {key: value for key, value in self.__dict__.items()}
        
    def load_state_dict(self, state_dict):
        ''' Loads early stopping state '''
        self.__dict__.update(state_dict)
