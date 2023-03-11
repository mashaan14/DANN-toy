import os

import torch
import torch.optim as optim
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import adjusted_rand_score

import params
from utils import make_variable


def train_src(feature_extractor_src, classifier, src_data_loader):

    # set train state for Dropout and BN layers
    feature_extractor_src.train()

    # setup criterion and optimizer
    optimizer_feature_extractor_src = optim.Adam(feature_extractor_src.parameters(),
                               lr=params.c_learning_rate,
                               betas=(params.beta1, params.beta2))

    len_data_loader = len(src_data_loader)

    for epoch in range(params.num_epochs_pre):

        if epoch + 1 <= params.num_epochs:
            encoded_feat = np.zeros((len(src_data_loader), 2))
            label_pred = np.zeros((len(src_data_loader), 1), dtype=int)
            label_true = np.zeros((len(src_data_loader), 1), dtype=int)

        # zip source and target data pair
        for step, (samples_src, labels_src) in enumerate(src_data_loader):

            # prepare samples
            samples_src = make_variable(samples_src)

            # prepare source class label
            labels_src = make_variable(labels_src)

            # extract features
            feat_src = feature_extractor_src(samples_src)

            # predict source samples on classifier
            labels_src_pred = classifier(feat_src)

            # compute losses
            loss = F.cross_entropy(labels_src_pred, labels_src)

            # Backpropagation
            optimizer_feature_extractor_src.zero_grad()
            loss.backward()
            optimizer_feature_extractor_src.step()

            encoded_feat[step, :] = feature_extractor_src(samples_src).detach().numpy()
            label_pred[step, :] = classifier(feature_extractor_src(samples_src)).data.max(1)[1].detach().numpy()
            label_true[step, :] = labels_src.numpy()

            if ((step + 1) % params.log_step == 0):
                print("Epoch [{}/{}] Step [{}/{}]:"
                      "loss={:.5f}"
                      .format(epoch + 1,
                              params.num_epochs,
                              step + 1,
                              len_data_loader,
                              loss.item()))

        # plot visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        fig.suptitle('Encoding features for source data')
        for g in np.unique(label_true):
            ix = np.where(label_true == g)
            ax1.scatter(encoded_feat[ix, 0], encoded_feat[ix, 1])

        for g in np.unique(label_pred):
            ix = np.where(label_pred == g)
            ax2.scatter(encoded_feat[ix, 0], encoded_feat[ix, 1])

        ax1.set_title('true labels')
        ax2.set_title('predicted labels')
        ax1.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,
                        labelbottom=False, labeltop=False, labelleft=False, labelright=False)
        ax2.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,
                        labelbottom=False, labeltop=False, labelleft=False, labelright=False)
        plt.savefig('plot-XS-train'+str(epoch)+'.png', bbox_inches='tight', dpi=600)

    return feature_extractor_src


def eval_src(encoder, classifier, data_loader, fig_title):
    """Evaluate classifier for source domain."""
    # set eval state for Dropout and BN layers
    encoder.eval()
    classifier.eval()

    # init loss and accuracy
    loss = 0
    acc = 0

    # set loss function
    criterion = nn.CrossEntropyLoss()

    feat = np.zeros((len(data_loader), 2))
    label_pred = np.zeros((len(data_loader), 1), dtype=int)
    label_true = np.zeros((len(data_loader), 1), dtype=int)
    step = 0

    # evaluate network
    for (samples, labels) in data_loader:

        # make smaples and labels variable
        samples = make_variable(samples)
        labels = make_variable(labels)

        preds = classifier(encoder(samples))
        loss += criterion(preds, labels).item()

        pred_cls = preds.data.max(1)[1]
        acc += pred_cls.eq(labels.data).cpu().sum()

        feat[step, :] = samples.detach().numpy()
        label_pred[step, :] = preds.data.max(1)[1].detach().numpy()
        label_true[step, :] = labels.numpy()
        step += 1

    loss /= len(data_loader)
    acc = acc.item()/len(data_loader.dataset)
    ari = adjusted_rand_score(label_true.flatten(), label_pred.flatten())

    print("Avg Loss = {:.5f}, Avg Accuracy = {:2%}, ARI = {:.5f}".format(loss, acc, ari))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(fig_title)
    for g in np.unique(label_true):
        ix = np.where(label_true == g)
        ax1.scatter(feat[ix, 0], feat[ix, 1])

    for g in np.unique(label_pred):
        ix = np.where(label_pred == g)
        ax2.scatter(feat[ix, 0], feat[ix, 1])

    ax1.set_title('true labels')
    ax2.set_title('predicted labels')
    ax1.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,
                    labelbottom=False, labeltop=False, labelleft=False, labelright=False)
    ax2.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,
                    labelbottom=False, labeltop=False, labelleft=False, labelright=False)
    plt.savefig(fig_title+'.png', bbox_inches='tight', dpi=600)


def train_tgt(feature_extractor, discriminator, classifier, src_data_loader, tgt_data_loader):
    """
    Domain-Adversarial Neural Networks (DANN):
        Unsupervised Domain Adaptation by Backpropagation, Ganin & Lemptsky (2014)
        Domain-adversarial training of neural networks, Ganin et al. (2016)
    """

    # set train state for Dropout and BN layers
    feature_extractor.train()
    discriminator.train()

    # setup criterion and optimizer
    optimizer_feature_extractor = optim.Adam(feature_extractor.parameters(),
                               lr=params.c_learning_rate,
                               betas=(params.beta1, params.beta2))
    optimizer_discriminator = optim.Adam(discriminator.parameters(),
                                  lr=params.d_learning_rate,
                                  betas=(params.beta1, params.beta2))
    len_data_loader = min(len(src_data_loader), len(tgt_data_loader))


    for epoch in range(params.num_epochs):

        if epoch + 1 <= params.num_epochs:
            encoded_feat = np.zeros((len(tgt_data_loader), 2))
            label_pred = np.zeros((len(tgt_data_loader), 1), dtype=int)
            label_true = np.zeros((len(tgt_data_loader), 1), dtype=int)

        # zip source and target data pair
        for step, ((samples_src, labels_src), (samples_tgt, labels_tgt)) in enumerate(zip(src_data_loader, tgt_data_loader)):

            # prepare samples
            samples_src = make_variable(samples_src)
            samples_tgt = make_variable(samples_tgt)

            # prepare domain labels
            domain_label_src = make_variable(torch.ones(samples_src.size(0)))
            domain_label_tgt = make_variable(torch.zeros(samples_tgt.size(0)))

            # prepare source class label
            labels_src = make_variable(labels_src)

            # extract features
            feat_src = feature_extractor(samples_src)
            feat_tgt = feature_extractor(samples_tgt)

            # predict on discriminator
            domain_label_src_pred = discriminator(feat_src)
            domain_label_tgt_pred = discriminator(feat_tgt)

            # predict source samples on classifier
            labels_src_pred = classifier(feat_src)

            # compute losses
            domain_loss_src = F.binary_cross_entropy_with_logits(domain_label_src_pred.flatten(), domain_label_src)
            domain_loss_tgt = F.binary_cross_entropy_with_logits(domain_label_tgt_pred.flatten(), domain_label_tgt)
            label_loss = F.cross_entropy(labels_src_pred, labels_src)
            loss = domain_loss_src + domain_loss_tgt + label_loss

            # Backpropagation
            optimizer_feature_extractor.zero_grad()
            optimizer_discriminator.zero_grad()
            loss.backward()
            optimizer_feature_extractor.step()
            optimizer_discriminator.step()


            encoded_feat[step, :] = feature_extractor(samples_tgt).detach().numpy()
            label_pred[step, :] = classifier(feature_extractor(samples_tgt)).data.max(1)[1].detach().numpy()
            label_true[step, :] = labels_tgt.numpy()

            if ((step + 1) % params.log_step == 0):
                print("Epoch [{}/{}] Step [{}/{}]:"
                      "label_loss={:.5f} domain_loss_src={:.5f} domain_loss_tgt={:.5f}"
                      .format(epoch + 1,
                              params.num_epochs,
                              step + 1,
                              len_data_loader,
                              label_loss.item(),
                              domain_loss_src.item(),
                              domain_loss_tgt.item()))

        # plot visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        fig.suptitle('Encoding features for target data')
        for g in np.unique(label_true):
            ix = np.where(label_true == g)
            ax1.scatter(encoded_feat[ix, 0], encoded_feat[ix, 1])

        for g in np.unique(label_pred):
            ix = np.where(label_pred == g)
            ax2.scatter(encoded_feat[ix, 0], encoded_feat[ix, 1])

        ax1.set_title('true labels')
        ax2.set_title('predicted labels')
        ax1.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,
                        labelbottom=False, labeltop=False, labelleft=False, labelright=False)
        ax2.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,
                        labelbottom=False, labeltop=False, labelleft=False, labelright=False)
        plt.savefig('plot-XT-train'+str(epoch)+'.png', bbox_inches='tight', dpi=600)

    return feature_extractor
