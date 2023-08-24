import torch
import torch.nn as nn

class PyTorchMlp(nn.Module):
  #the observation space is a 1D array of lenght 18 : [goal_angle, goal_dist, x, y, vx, vy, 12 sectors]
  #the policy is 2 layers of 64 neurons fully connected
  #the output is a multibinary vector for each sector (lenght 12) ex: [0 1 0 0 1 1 1 1 0 0 0 1]

  def __init__(self, n_inputs=18, n_actions=12):
      nn.Module.__init__(self)

      self.fc1 = nn.Linear(n_inputs, 64)
      self.fc2 = nn.Linear(64, 64)
      self.fc3 = nn.Linear(64, n_actions)
      self.activ_fn = nn.Tanh()
      self.out_activ = nn.Softmax(dim=0) #is softmax used for the output layer in SB? don't know

  def forward(self, x):
      x = self.activ_fn(self.fc1(x))
      x = self.activ_fn(self.fc2(x))
      x = self.activ_fn(self.fc3(x)) #do i need an activation function here?
      return x