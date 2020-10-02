import numpy as np
import torch
import os
import pickle

def load_data(data_dir='./data'):
    
    print('Loading Data...')
    filename = os.path.join(data_dir, 'sort-of-clevr.pickle')
    with open(filename, 'rb') as f:
        train_data, test_data = pickle.load(f)
        
    rel_train = []
    norel_train = []
    rel_test = []
    norel_test = []
    
    print('Processing Data...')

    for img, relations, norelations in train_data:
        img = np.swapaxes(img, 0, 2)
        
        for ques, ans in zip(relations[0], relations[1]):
            rel_train.append((img, ques, ans))
        for ques, ans in zip(norelations[0], norelations[1]):
            norel_train.append((img, ques, ans))

    for img, relations, norelations in test_data:
        img = np.swapaxes(img, 0, 2)
        
        for ques, ans in zip(relations[0], relations[1]):
            rel_test.append((img, ques, ans))
        for ques, ans in zip(norelations[0], norelations[1]):
            norel_test.append((img, ques, ans))
    
    return rel_train, rel_test, norel_train, norel_test

def splice_data(data):
    imgs = [e[0] for e in data]
    ques = [e[1] for e in data]
    ans = [e[2] for e in data]
    return (imgs, ques, ans)

def tensorize(data, i, args):
    bs = args.batch_size
    imgs = torch.from_numpy(np.asarray(data[0][bs*i:bs*(i+1)])).float()
    ques = torch.from_numpy(np.asarray(data[1][bs*i:bs*(i+1)])).float()
    ans = torch.from_numpy(np.asarray(data[2][bs*i:bs*(i+1)])).long()

    return imgs.to(args.device), ques.to(args.device), ans.to(args.device)