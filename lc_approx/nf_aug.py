import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable


# TODO: consider to remove
class InvertibleLayer(nn.Module):
    def __init__(self, var_size):
        super().__init__()

        self.var_size = var_size

    
class NormalizingFlow(nn.Module):
    def __init__(self, layers, prior):
        super().__init__()

        self.layers = nn.ModuleList(layers)
        self.prior = prior

    def log_prob(self, x, y):
        """
        x: torch.Tensor of shape [batch_size, var_size]
            Data
        y: torch.Tensor of shape [batch_size, cond_size]
            Condition
        """
        log_likelihood = None

        for layer in self.layers:
            x, change = layer.f(x, y)
            if log_likelihood is not None:
                log_likelihood = log_likelihood + change
            else:
                log_likelihood = change
        log_likelihood = log_likelihood + self.prior.log_prob(x)

        return log_likelihood.mean()

    def sample(self, y):
        """
        y: torch.Tensor of shape [batch_size, cond_size]
            Condition
        """
        
        x = self.prior.sample((len(y), ))
        for layer in self.layers[::-1]:
            x = layer.g(x, y)

        return x
    
    
class RealNVP(InvertibleLayer):
    
    def __init__(self, var_size, cond_size, mask, hidden=10):
        super().__init__(var_size=var_size)

        self.mask = mask

        self.nn_t = nn.Sequential(
            nn.Linear(var_size+cond_size, hidden),
            nn.Tanh(),
            nn.Linear(hidden, var_size)
            )
        self.nn_s = nn.Sequential(
            nn.Linear(var_size+cond_size, hidden),
            nn.Tanh(),
            nn.Linear(hidden, var_size),
            )

    def f(self, x, y):
        """
        x: torch.Tensor of shape [batch_size, var_size]
            Data
        y: torch.Tensor of shape [batch_size, cond_size]
            Condition
        """
        xy = torch.cat((x * self.mask[None, :], y), dim=1)
        t = self.nn_t(xy)
        s = self.nn_s(xy)

        new_x = (x * torch.exp(s) + t) * (1 - self.mask[None, :]) + x * self.mask[None, :]
        log_det = (s * (1 - self.mask[None, :])).sum(dim=-1)
        return new_x, log_det

    def g(self, x, y):
        """
        x: torch.Tensor of shape [batch_size, var_size]
            Data
        y: torch.Tensor of shape [batch_size, cond_size]
            Condition
        """
        xy = torch.cat((x * self.mask[None, :], y), dim=1)
        t = self.nn_t(xy)
        s = self.nn_s(xy)

        new_x = ((x - t) * torch.exp(-s)) * (1 - self.mask[None, :]) + x * self.mask[None, :]
        return new_x
    
    
class NFFitter(object):
    
    def __init__(self, var_size=2, cond_size=2, normalize_y=True, batch_size=32, n_epochs=10, lr=0.0001,
                 randomize_x=True, device='cpu'):
        
        self.normalize_y = normalize_y
        self.randomize_x = randomize_x
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.lr = lr
        
        prior = torch.distributions.MultivariateNormal(torch.zeros(var_size), torch.eye(var_size))

        layers = []
        for i in range(8):
            layers.append(RealNVP(var_size=var_size, cond_size=cond_size+randomize_x, mask=((torch.arange(var_size) + i) % 2)))

        self.nf = NormalizingFlow(layers=layers, prior=prior)
        self.opt = torch.optim.Adam(self.nf.parameters(), lr=self.lr)

        self.device = torch.device(device)
        
    def reshape(self, y):
        if y.ndim < 2:
            return y.reshape(-1, 1)
        return y

    def fit(self, X, y, y_std=None):
        
        # reshape
        y = self.reshape(y)
        
        # normalize
        if self.normalize_y:
            self.ss_y = StandardScaler()
            y = self.ss_y.fit_transform(y)
            
        if y_std is not None:
            y_std = self.reshape(y_std)
            if self.normalize_y:
                y_std /= self.ss_y.scale_
        else:
            y_std = np.zeros_like(y)
            
        #noise = np.random.normal(0, 1, (y.shape[0], 1))
        #y = np.concatenate((y, noise), axis=1)
        
        # numpy to tensor
        y_real = torch.tensor(y, dtype=torch.float32, device=self.device)
        y_real_std = torch.tensor(y_std, dtype=torch.float32, device=self.device)
        X_cond = torch.tensor(X, dtype=torch.float32, device=self.device)
        
        # tensor to dataset
        dataset_real = TensorDataset(y_real, y_real_std, X_cond)
        
        criterion = nn.MSELoss()
        self.loss_history = []

        # Fit GAN
        for epoch in range(self.n_epochs):
            for i, (y_batch, std_batch, x_batch) in enumerate(
                DataLoader(dataset_real, batch_size=self.batch_size, shuffle=True)
            ):   
                noise = np.random.normal(0, 1, (len(y_batch), 1))
                noise = torch.tensor(noise, dtype=torch.float32, device=self.device)
                
                y_batch = torch.normal(y_batch, std_batch)
                y_batch = torch.cat((y_batch, noise), dim=1)
                
                if self.randomize_x:
                    noise = np.random.normal(0, 1, (len(x_batch), 1))
                    noise = torch.tensor(noise, dtype=torch.float32, device=self.device)
                    x_batch = torch.cat((x_batch, noise), dim=1)
                
                #y_pred = self.nf.sample(x_batch)
                
                # caiculate loss
                loss = -self.nf.log_prob(y_batch, x_batch)
                #loss = criterion(y_batch, y_pred)
                
                # optimization step
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                    
                # caiculate and store loss
                self.loss_history.append(loss.detach().cpu())            
        
    def predict(self, X):
        #noise = np.random.normal(0, 1, (X.shape[0], 1))
        #X = np.concatenate((X, noise), axis=1)
        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        if self.randomize_x:
            noise = np.random.normal(0, 1, (len(X), 1))
            noise = torch.tensor(noise, dtype=torch.float32, device=self.device)
            X = torch.cat((X, noise), dim=1)
        y_pred = self.nf.sample(X).cpu().detach().numpy()[:, 0]
        # normalize
        if self.normalize_y:
            y_pred = self.ss_y.inverse_transform(y_pred)
        return y_pred
    
    def predict_n_times(self, X, n_times=100):
        predictions = []
        for i in range(n_times):
            y_pred = self.predict(X)
            predictions.append(y_pred)
        predictions = np.array(predictions)
        mean = predictions.mean(axis=0)
        std = predictions.std(axis=0)
        return mean, std
