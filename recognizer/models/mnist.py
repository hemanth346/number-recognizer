import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
# from torch.optim.lr_scheduler import StepLR

from tqdm.auto import tqdm, trange # auto select tqdm for notebook
import matplotlib.pyplot as plt

from recognizer import device 


class Learner():
    def __init__(self, network, train_loader, test_loader, device=device):
        self.model = network().to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device=device
        self.train_losses = []
        self.test_losses = []
        self.train_acc = []
        self.test_acc = []

    def train_epoch(self, optimizer, scheduler, device=None):
        self.model.train()
        pbar = tqdm(self.train_loader, ncols="80%")
        correct = 0
        processed = 0
        if not device:
            device = self.device
        for batch_idx, (data, target) in enumerate(pbar):
        # for data, target in self.train_loader:
            data, target = data.to(device), target.to(device) # get batch data
            optimizer.zero_grad() # Init
            y_pred = self.model(data) # Predict
            loss = F.nll_loss(y_pred, target) # Calculate loss
            self.train_losses.append(loss) 

            loss.backward() # Backpropagation
            optimizer.step()
            # https://discuss.pytorch.org/t/what-does-scheduler-step-do/47764/6
            scheduler.step() # Decay the Learning Rate
            
            pred = y_pred.argmax(dim=1, keepdim=True) # get the index of the max log-probability, NLL loss
            correct += pred.eq(target.view_as(pred)).sum().item()
            processed += len(data)

            # # Update pbar-tqdm
            pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
            self.train_acc.append(100*correct/processed)

    def test_epoch(self, device=None):
        self.model.eval()
        test_loss = 0
        correct = 0
        if not device:
            device = self.device
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(device), target.to(device)
                output = self.model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self.test_loader.dataset)
        self.test_losses.append(test_loss)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(self.test_loader.dataset),
            100. * correct / len(self.test_loader.dataset)))
        
        self.test_acc.append(100. * correct / len(self.test_loader.dataset))


    def start_train(self, epochs=10, device=device):
        optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        # scheduler = StepLR(optimizer, step_size=6, gamma=0.1)
        scheduler = OneCycleLR(optimizer, max_lr=0.1, steps_per_epoch=len(self.train_loader), epochs=epochs)

        for epoch in range(epochs):
            # Print Learning Rate
            print("EPOCH:", epoch+1, 'LR:', scheduler.get_lr())
            self.train_epoch(optimizer, scheduler)
            self.test_epoch()
    
    def save_model(self, path):
        torch.save(self.model, path)
        return path

    def load_model(self, path):
        pass
    
    def training_plot(self):
        fig, axs = plt.subplots(2,2,figsize=(15,10))
        axs[0, 0].plot(self.train_losses)
        axs[0, 0].set_title("Training Loss")
        axs[1, 0].plot(self.train_acc[4000:])
        axs[1, 0].set_title("Training Accuracy")
        axs[0, 1].plot(self.test_losses)
        axs[0, 1].set_title("Test Loss")
        axs[1, 1].plot(self.test_acc)
        axs[1, 1].set_title("Test Accuracy")
        plt.show()

