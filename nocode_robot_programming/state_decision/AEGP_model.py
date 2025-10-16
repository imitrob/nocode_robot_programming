

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import gpytorch
from tqdm import tqdm


class AEGP():
    """ Autoencoder + Gaussian Process """
    def __init__(self):
        self.videoembedder = VideoEmbedder()
        self.riskestimator = GPEstimator()
        # ORIGINAL ILESIA
        # self.riskestimator = GPEstimatorIlesia()
        self.y_cls = None

        # (1/3) TMP: PLOT PROBS
        self.mean_probs = []

    def train(self, X: torch.Tensor, y: torch.Tensor, y_cls):
        self.y_cls = y_cls 
        """ X.shape = (samples, width, height), y.shape = (samples, ) """
        Xpp = torch.concatenate([X[1:].clone(), X[-1:].clone()])
        X_ = torch.stack([X, Xpp]).swapaxes(0,1) # X_.shape = (samples, 2, width, height)

        self.videoembedder.training_loop(DataLoader(X_))
        self.videoembedder.model.eval()
        latent = torch.tensor([]).cuda()
        for i in range(len(X_)):
            latent_ = self.videoembedder.model.encoder(X_[i:i+1,0:1,:,:]) # extract image (discards next_image). 4D
            latent = torch.concatenate([latent, latent_])
        
        self.riskestimator.training_loop(latent, y)

    def predict(self, image: torch.Tensor, timestep: float | None = None) -> str:
        """ See state_decider.py:StateDeciderBase model
        """
        self.videoembedder.model.eval()
        with torch.no_grad():
            latent = self.videoembedder.model.encoder(image.unsqueeze(0).unsqueeze(0)) # (1, 1, width, height), 4D

        # self.riskestimator.model.eval()
        # self.riskestimator.likelihood.eval()
        # with torch.no_grad(), gpytorch.settings.fast_pred_var():
        #     observed_pred = self.riskestimator.likelihood(self.riskestimator.model(latent))
        #     mean = observed_pred.mean.cpu().numpy()
        #     std = observed_pred.stddev.cpu().numpy()
            
        #     r = mean + std
        #     pred = r > 0.5
        # return bool(pred), self.y_cls[int(mean)]

        device = next(self.riskestimator.model.parameters()).device
        x = latent.to(device)

        self.riskestimator.model.eval()
        self.riskestimator.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            # Draw MC samples from the likelihood, shape: (S, N, C)
            with gpytorch.settings.num_likelihood_samples(128):
                pred = self.riskestimator.likelihood(self.riskestimator.model(x))
                probs = pred.probs  # (S, N, C)
                # ORIGINAL ILESIA
                # mean = float(pred.mean.cpu().numpy())
                # std = pred.stddev.cpu().numpy()

        mean_probs = probs.mean(0)          # (N, C)
        labels = mean_probs.argmax(dim=-1)  # (N,)
        # ORIGINAL ILESIA
        # mean_probs = [(mean, 1-mean)]
        # labels = round(mean)

        # (2/3) TMP: PLOT PROBS
        self.mean_probs.append([float(mean_probs[0][0]), float(mean_probs[0][1])])
        
        return self.y_cls[int(labels)]

    def predict_many(self, X):
        r = [self.predict(x) for x in X]
        
        # (3/3) TMP: PLOT PROBS
        import matplotlib.pyplot as plt
        import numpy as np
        plt.hist(np.array(self.mean_probs), bins=10)
        plt.legend(self.y_cls)

        return r

class Autoencoder3(nn.Module):
    def forward(self, x):
        z = self.encoder(x)
        x_reconstructed = self.decoder(z)
        return x_reconstructed
    
    def __init__(self, latent_dim: int = 12):
        super(Autoencoder3, self).__init__()

        # Encoder with dropout
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.1),  # Added
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(0.05),  # Added
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(0.05),  # Added
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, latent_dim),
            nn.Dropout(0.02)  # Added after linear layer
        )
        
        # Decoder with dropout
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256 * 8 * 8),
            nn.Dropout(0.02),  # Added
            nn.Unflatten(1, (256, 8, 8)),
            
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(0.05),  # Added
            
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.1),  # Added
            
            nn.ConvTranspose2d(64, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

class VideoEmbedder():
    def __init__(self):
        self.model = Autoencoder3(12)
        self.model.to("cuda")
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=0.01, weight_decay=0.0000001
        )

    def nuclear_norm_loss(self, x):
        # Compute the nuclear norm of the input tensor
        # x = x.view(x.size(0), -1)  # Flatten the tensor
        u, s, v = torch.svd(x)
        return torch.sum(s)

    def training_loop(self, dataloader: DataLoader, num_epochs: int = 50, patience = 100):    
        self.model.train()
        best_loss: float = float("inf")
        counter = 0
        
        dataset = dataloader.dataset
        dataset_size = len(dataset)
        train_size = int(0.8 * dataset_size)
        test_size = dataset_size - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
        train_loader = DataLoader(train_dataset, batch_size=dataloader.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=dataloader.batch_size, shuffle=False)

        pviz = tqdm(range(num_epochs))
        for epoch in pviz:
            if epoch % 10 == 0 and epoch > 0:
                self.optimizer.param_groups[0]["lr"] *= 0.5
            for data_tensor in train_loader:
                next_image = data_tensor[:, 1, :, :].unsqueeze(1)
                input_batch = data_tensor[:, 0, :, :].unsqueeze(1)
                self.optimizer.zero_grad()
                # output = self.model(input_batch)
                latent_vec = self.model.encoder(input_batch)
                output = self.model.decoder(latent_vec)

                latent_vec_next_image = self.model.encoder(next_image)

                alpha = 100.0
                beta = 0.0001
                l1_lambda = 0.00001
                continuity_loss_lambda = 1.0

                # continuity loss
                continuity_loss = torch.nn.functional.mse_loss(latent_vec, latent_vec_next_image)

                # l1 loss on all weights
                l1_loss = 0
                for name, param in self.model.named_parameters():
                    l1_loss += torch.sum(torch.abs(param))

                # reconstruction loss
                recon_loss = self.criterion(output, input_batch)

                # nuclear norm loss
                nuclear_loss = self.nuclear_norm_loss(latent_vec)

                loss = (
                    alpha * recon_loss
                    + beta * nuclear_loss
                    + l1_lambda * l1_loss
                    + continuity_loss_lambda * continuity_loss
                )
                loss.backward()
                self.optimizer.step()

                train_loss = loss.item()

            # compute the test loss
            for data_tensor in test_loader:
                next_image = data_tensor[:, 1, :, :].unsqueeze(1)
                input_batch = data_tensor[:, 0, :, :].unsqueeze(1)
                # input_batch = data[0]
                with torch.no_grad():
                    latent_vec = self.model.encoder(input_batch)
                    output = self.model.decoder(latent_vec)
                    loss = self.criterion(output, input_batch)

            val_loss = loss.item()

            if val_loss < best_loss:
                best_loss = val_loss
                counter = 0  # Reset patience counter
            else:
                counter += 1

            if counter >= patience:  # Stop if no improvement for `patience` epochs
                print(f"No improvement for {patience} epochs. Stopping training.")
                break

            pviz.set_description(
                desc=f"Epoch [{epoch}/{num_epochs}], Trainloss: {round(train_loss,3)}, ValLoss: {round(val_loss,6)}"
            )

        return epoch, loss




from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution, MultitaskVariationalStrategy

class MulticlassGP(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points: torch.Tensor, num_classes: int, ard_dims: int):
        # Base variational strategy (single latent)
        q_dist = CholeskyVariationalDistribution(inducing_points.size(-2))
        base_vs = VariationalStrategy(self, inducing_points, q_dist, learn_inducing_locations=True)
        # Promote to multi-task: one latent per class
        mt_vs = MultitaskVariationalStrategy(base_vs, num_tasks=num_classes)
        super().__init__(mt_vs)

        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=ard_dims,
                                       lengthscale_constraint=gpytorch.constraints.Interval(0.1, 1.0))
        )

    def forward(self, x):
        mean_x  = self.mean_module(x)
        covar_x = self.covar_module(x)
        # With MultitaskVariationalStrategy, return a standard MVN; the strategy handles tasks
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GPEstimator:
    def __init__(self):
        super(GPEstimator, self).__init__()
    
    def training_loop(self, X, Y, train_epoch=400, lr=0.01, M=128):
        """
        X: (N, D) float tensor
        Y: (N,) long tensor with class indices 0..num_classes-1
        M: number of inducing points (choose <= N)
        """
        num_classes = len(set(Y.cpu().numpy()))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        X = X.to(device)
        Y = Y.long().to(device)

        # Choose inducing points (simple random subset; use k-means for better coverage if you like)
        idx = torch.randperm(X.size(0))[:min(M, X.size(0))]
        Z = X[idx].contiguous()

        self.model = MulticlassGP(Z, num_classes=num_classes, ard_dims=X.size(-1)).to(device)
        self.likelihood = gpytorch.likelihoods.SoftmaxLikelihood(num_classes=num_classes, num_features=num_classes).to(device)

        self.model.train()
        self.likelihood.train()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        mll = gpytorch.mlls.VariationalELBO(self.likelihood, self.model, num_data=Y.size(0))

        for _ in tqdm(range(train_epoch)):
            optimizer.zero_grad()
            output = self.model(X)           # latent functions (one per class)
            loss = -mll(output, Y)           # Y must be LongTensor
            loss.backward(retain_graph=True)
            optimizer.step()

        self.trained_epoch = train_epoch


class GPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()  
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                ard_num_dims=train_x.size(-1),
                lengthscale_constraint=gpytorch.constraints.Interval(0.1, 1.0),
                ),
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class GPEstimatorIlesia():
    def __init__(self):
        pass
    def training_loop(self, X, Y, train_epoch = 400):
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.model = GPModel(X, Y, self.likelihood)
        self.model=self.model.cuda()
        self.likelihood=self.likelihood.cuda()

        self.model.train()
        self.likelihood.train()
        optimizer = torch.optim.Adam([
            {'params': self.model.parameters()},
        ], lr=0.01)

        # Our loss object. We're using the VariationalELBO
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        try:
            for _ in tqdm(range(train_epoch)):
                optimizer.zero_grad()
                output = self.model(X)
                loss = -mll(output, Y)
                loss.backward(retain_graph=True)
                self.loss = loss.item()
                optimizer.step()
        except KeyboardInterrupt:
            print("Stopping on interrupt")

if __name__ == "__main__":
    AEGP()