import pickle
import argparse
import os
from tqdm import tqdm
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from model import RNModel
from utils import load_data, splice_data, tensorize

def train(model, rel, norel, args):
    
    for epoch in range(args.epochs):
        model.train()

        random.shuffle(rel)
        random.shuffle(norel)
        
        rel_train = splice_data(rel)
        norel_train = splice_data(norel)
        
        acc_rels = []
        acc_norels = []
        loss_rels = []
        loss_norels = []

        # Create progress bar
        num_steps = len(rel_train[0]) // args.batch_size
        pbar = tqdm(total=num_steps * 2, desc='Epoch {}'.format(epoch+1))
        
        # Training
        for batch_idx in range(num_steps):
            
            # Train on Relational questions
            images, questions, answers = tensorize(rel_train, batch_idx, args)
            rel_acc, rel_loss = model.train_(images, questions, answers)
            
            # Train on Non-Relational questions
            images, questions, answers = tensorize(norel_train, batch_idx, args)
            norel_acc, norel_loss = model.train_(images, questions, answers)
            
            # Progress Bar Logging
            pbar.update(2)
            pbar.set_postfix({'Relation Accuracy': rel_acc, 
                              'Non-relations Accuracy': norel_acc})
            
            acc_rels.append(rel_acc.item())
            acc_norels.append(norel_acc.item())
            loss_rels.append(rel_loss.item())
            loss_norels.append(norel_loss.item())
        
        # Save checkpoint
        model.save_model(epoch+1)
        
        mean_rel_acc = np.array(acc_rels).mean()
        mean_norel_acc = np.array(acc_norels).mean()        
        pbar.set_postfix({'Mean Relation Accuracy': mean_rel_acc, 
                          'Mean Non-relations Accuracy': mean_norel_acc})
        
        pbar.close()
        
def test(model, rel, norel, args):
    
    model.eval()

    random.shuffle(rel)
    random.shuffle(norel)

    rel_test = splice_data(rel)
    norel_test = splice_data(norel)

    acc_rels = []
    acc_norels = []
    loss_rels = []
    loss_norels = []

    # Create progress bar
    num_steps = len(rel_test[0]) // args.batch_size
    pbar = tqdm(total=num_steps * 2, desc='Evaluating...')

    with torch.no_grad():
        for batch_idx in range(num_steps):

            # Train on Relational questions
            images, questions, answers = tensorize(rel_test, batch_idx, args)
            rel_acc, rel_loss = model.evaluate(images, questions, answers)

            # Train on Non-Relational questions
            images, questions, answers = tensorize(norel_test, batch_idx, args)
            norel_acc, norel_loss = model.evaluate(images, questions, answers)

            # Progress Bar Logging
            pbar.update(2)
            pbar.set_postfix({'Relation Accuracy': rel_acc, 
                              'Non-relations Accuracy': norel_acc})

            acc_rels.append(rel_acc.item())
            acc_norels.append(norel_acc.item())
            loss_rels.append(rel_loss.item())
            loss_norels.append(norel_loss.item())

        mean_rel_acc = np.array(acc_rels).mean()
        mean_norel_acc = np.array(acc_norels).mean()        
        pbar.set_postfix({'Mean Test Relation Accuracy': mean_rel_acc, 
                          'Mean Test Non-relations Accuracy': mean_norel_acc})

        pbar.close()
        
        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Relation Network Sort-of-CLEVR Training Script')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--data-dir', type=str, default='data', help='Pickle data path')
    args = parser.parse_args()
    
    # Check cuda
    args.device = 'cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu'
    
    SEED = 1
    torch.manual_seed(SEED)
    if args.device == 'cuda':
        torch.cuda.manual_seed(SEED)
    
    # Loading Data from generated pickle files
    rel_train, rel_test, norel_train, norel_test = load_data(args.data_dir)
    
    # Create model
    model = RNModel(args)
    model.to(args.device)
    
    # Create './models' directory for saving weights
    try:
        os.mkdir('models')
    except:
        pass
    
    # Train
    train(model, rel_train, norel_train, args)
    
    # Evaluate on test set
    test(model, rel_test, norel_test, args)