import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
print(f"Using {device} device")
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        #Μετατροπή μιας 2D εικόνας σε 1D (π.χ 28x28 -> 784 pixels)
        self.flatten = nn.Flatten() 
        #Sequential: περιέχει σειριακά τα modules του μοντέλου
        #τα δεδομένα επίσης περνούν σειριακά μέσα από το δίκτυο
        self.linear_relu_stack = nn.Sequential(
            #Εφαρμόζει ένα γραμμικό μετασχηματισμό στα 
            #δεδομένα εισόδου χρησιμοποιώντας τα weights & biases
            nn.Linear(28*28, 512),
            #Μη-γραμμικά activations που βοηθούν το μοντέλο 
            #να "μάθει" τη συσχέτιση μεταξύ δεδομένων
            #εισόδου και εξόδου
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
model = NeuralNetwork().to(device)
print(model)

X = torch.rand(1, 28, 28, device=device)
logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")


class NeuralNetwork2(nn.Module):
    def __init__(self):
       
        super().__init__()
        self.simple_NN = nn.Sequential(
            nn.Linear(5, 3)
        )
    def forward(self, x):
        
        logits = self.simple_NN(x)
        return logits
    
model = NeuralNetwork2().to(device)
print(model)