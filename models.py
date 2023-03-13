"""
This file contains three models needed to build DANN:
    - Encoder: to extract features.
    - Classifier: to perform classification.
    - Discriminator model: to perform domain adaptation.

Domain-Adversarial Neural Networks (DANN):
    Unsupervised Domain Adaptation by Backpropagation, Ganin & Lemptsky (2014)
    Domain-adversarial training of neural networks, Ganin et al. (2016)
"""

from torch import nn
from utils import GradientReversal


class Encoder(nn.Module):
    """
    encoder for DANN.
    """
    def __init__(self):
        super(Encoder, self).__init__()

        self.restored = False

        self.layer = nn.Sequential(
            nn.Linear(2, 20),
            nn.ReLU(),
            nn.Linear(20, 50),
            nn.ReLU(),
            nn.Linear(50, 2)
        )

        # self.layer = nn.Sequential(
        #     nn.Linear(2, 20),
        #     nn.ReLU(),
        #     nn.Linear(20, 50),
        #     nn.ReLU(),
        #     nn.Linear(50, 2)
        # )

    def forward(self, input):
        out = self.layer(input)
        return out


class Classifier(nn.Module):
    """
    classifier for DANN.
    """
    def __init__(self):
        super(Classifier, self).__init__()

        self.restored = False

        self.layer = nn.Sequential(
            nn.Linear(2, 2),
            nn.LogSoftmax()
        )

    def forward(self, input):
        out = self.layer(input)
        return out


class Discriminator(nn.Module):
    """
    Discriminator model for source domain.
    """

    def __init__(self):
        """Init discriminator."""
        super(Discriminator, self).__init__()

        self.restored = False

        self.layer = nn.Sequential(
            GradientReversal(),
            nn.Linear(2, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 1)#,
            #nn.LogSoftmax()
        )

    def forward(self, input):
        """Forward the discriminator."""
        out = self.layer(input)
        return out
