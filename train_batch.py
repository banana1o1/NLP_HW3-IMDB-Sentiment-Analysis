import torch
import torch.nn as nn
from torch import optim
import time, random
import os
from tqdm import tqdm
from lstm import LSTMSentiment
from bilstm import BiLSTMSentiment
from torchtext import data
import numpy as np
import argparse
from torchtext.data import Field
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from att_lstm import ATT_LSTMSentiment



torch.set_num_threads(8)
torch.manual_seed(1)
random.seed(1)


def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in range(vocab_size):
            word = []
            while True:
                ch = f.read(1).decode('latin-1')
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            if word in vocab:
               word_vecs[word] = np.frombuffer(f.read(binary_len), dtype='float32')
            else:
                f.read(binary_len)
    return word_vecs


def get_accuracy(truth, pred):
    assert len(truth) == len(pred)
    right = 0
    # for i in range(len(truth)):
    #     if truth[i] == pred[i]:
    #         right += 1.0
    # acc=right / len(truth)
    acc=accuracy_score(truth, pred)
    precision=precision_score(truth, pred)
    recall=recall_score(truth, pred)
    f1score=f1_score(truth, pred)

    return acc,precision,recall,f1score


def train_epoch_progress(model, train_iter, loss_function, optimizer, text_field, label_field, epoch):
    model.train()
    avg_loss = 0.0
    truth_res = []
    pred_res = []
    count = 0
    for batch in tqdm(train_iter, desc='Train epoch '+str(epoch+1)):
        sent, label = batch.text, batch.label
        label.data.sub_(1)
        truth_res += list(label.data)
        model.batch_size = len(label.data)
        model.hidden = model.init_hidden()
        # import pdb;pdb.set_trace()
        pred = model(sent.cuda())
        pred_label = pred.data.max(1)[1].cpu().numpy()
        pred_res += [x for x in pred_label]
        model.zero_grad()
        loss = loss_function(pred, label.cuda())
        avg_loss += loss.item()
        count += 1
        loss.backward()
        optimizer.step()
    avg_loss /= len(train_iter)
    acc = get_accuracy(truth_res, pred_res)[0]
    return avg_loss, acc


def train_epoch(model, train_iter, loss_function, optimizer):
    model.train()
    avg_loss = 0.0
    truth_res = []
    pred_res = []
    count = 0
    for batch in train_iter:
        sent, label = batch.text, batch.label
        label.data.sub_(1)
        truth_res += list(label.data)
        model.batch_size = len(label.data)
        model.hidden = model.init_hidden()
        pred = model(sent.cuda())
        pred_label = pred.data.max(1)[1].cpu().numpy()
        pred_res += [x for x in pred_label]
        model.zero_grad()
        loss = loss_function(pred, label.cuda())
        avg_loss += loss.data[0]
        count += 1
        loss.backward()
        optimizer.step()
    avg_loss /= len(train_iter)
    acc = get_accuracy(truth_res, pred_res)[0]
    return avg_loss, acc


def evaluate(model, data, loss_function, name):
    model.eval()
    avg_loss = 0.0
    truth_res = []
    pred_res = []
    for batch in data:
        # import pdb;pdb.set_trace()

        sent, label = batch.text, batch.label
        label.data.sub_(1)
        truth_res += list(label.data.numpy())
        model.batch_size = len(label.data)
        model.hidden = model.init_hidden()
        # import pdb;pdb.set_trace()
        pred = model(sent.cuda())
        pred_label = pred.data.max(1)[1].cpu().numpy()
        pred_res += [x for x in pred_label]
        loss = loss_function(pred, label.cuda())
        avg_loss += loss.item()
    avg_loss /= len(data)

    acc,precision,recall,f1score = get_accuracy(truth_res, pred_res)
    print(name + ': loss %.2f acc %.1f precision %.1f recall %.1f f1score %.1f ' % (avg_loss, 
        acc*100,precision*100,recall*100,f1score*100))
    return acc,f1score


def load_sst(text_field, label_field, batch_size):
    train, dev, test = data.TabularDataset.splits(path='./FOR_USE/', train='train.tsv',
                                                  validation='dev.tsv', test='test.tsv', format='tsv',
                                                  fields=[('text', text_field), ('label', label_field)])
    text_field.build_vocab(train, dev, test)
    label_field.build_vocab(train, dev, test)
    train_iter, dev_iter, test_iter = data.BucketIterator.splits((train, dev, test),
                batch_sizes=(batch_size, batch_size, batch_size), sort_key=lambda x: len(x.text), repeat=False)
    # for data_train, data_dev, data_test in zip(train_iter, dev_iter, test_iter):
        # sent_train, _= data_train.text, data_train.label
        # sent_dev, _ = data_dev.text, data_dev.label
        #import pdb;pdb.set_trace()
    ## for GPU run
#     train_iter, dev_iter, test_iter = data.BucketIterator.splits((train, dev, test),
#                 batch_sizes=(batch_size, len(dev), len(test)), sort_key=lambda x: len(x.text), repeat=False, device=None)
    return train_iter, dev_iter, test_iter


# def adjust_learning_rate(learning_rate, optimizer, epoch):
#     lr = learning_rate * (0.1 ** (epoch // 10))
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr
#     return optimizer


args = argparse.ArgumentParser()
args.add_argument('--m', dest='model', default='lstm', help='specify the mode to use (default: lstm)')
args = args.parse_args()

EPOCHS = 16
USE_GPU = torch.cuda.is_available()
EMBEDDING_DIM = 300
HIDDEN_DIM = 150

BATCH_SIZE = 32
timestamp = str(int(time.time()))
best_dev_evalue = 0.0



text_field = data.Field(lower=True)#,fix_length=128
label_field = data.Field(sequential=False)
train_iter, dev_iter, test_iter = load_sst(text_field, label_field, BATCH_SIZE)

#torch.backends.cudnn.benchmark = True

if args.model == 'lstm':
    model = LSTMSentiment(embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, vocab_size=len(text_field.vocab), label_size=len(label_field.vocab)-1,\
                          use_gpu=USE_GPU, batch_size=BATCH_SIZE)

if args.model == 'bilstm':
    model = BiLSTMSentiment(embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, vocab_size=len(text_field.vocab), label_size=len(label_field.vocab)-1,\
                          use_gpu=USE_GPU, batch_size=BATCH_SIZE)

if args.model == 'attbilstm':
    model = ATT_BiLSTMSentiment(embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, vocab_size=len(text_field.vocab), label_size=len(label_field.vocab)-1,\
                          use_gpu=USE_GPU, batch_size=BATCH_SIZE)
                        
if args.model == 'attlstm':
    model = ATT_LSTMSentiment(embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, vocab_size=len(text_field.vocab), label_size=len(label_field.vocab)-1,\
                          use_gpu=USE_GPU, batch_size=BATCH_SIZE)


if USE_GPU:
    model = model.cuda()



#print('---',args,'---',args.model,'---',USE_GPU)
print('Load word embeddings...')
# # glove
# text_field.vocab.load_vectors('glove.6B.100d')

# word2vector
word_to_idx = text_field.vocab.stoi
pretrained_embeddings = np.random.uniform(-0.25, 0.25, (len(text_field.vocab), 300))
pretrained_embeddings[0] = 0
word2vec = load_bin_vec('./FOR_USE/GoogleNews-vectors-negative300.bin', word_to_idx)
for word, vector in word2vec.items():
    pretrained_embeddings[word_to_idx[word]-1] = vector

# text_field.vocab.load_vectors(wv_type='', wv_dim=300)

model.embeddings.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
# model.embeddings.weight.data = text_field.vocab.vectors
# model.embeddings.embed.weight.requires_grad = False


best_model = model
optimizer = optim.Adam(model.parameters(), lr=0.5*1e-3)
loss_function = nn.CrossEntropyLoss()

print('Training...')
out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
print("Writing to {}\n".format(out_dir))
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
for epoch in range(EPOCHS):
    avg_loss, acc= train_epoch_progress(model, train_iter, loss_function, optimizer, text_field, label_field, epoch)
    tqdm.write('Train: loss %.2f acc %.1f' % (avg_loss, acc*100))
    dev_evalue = evaluate(model, dev_iter, loss_function, 'Dev')[1]
    if dev_evalue > best_dev_evalue:
        if best_dev_evalue > 0:
            os.system('rm '+ out_dir + '/best_model' + '.pth')
        best_dev_evalue = dev_evalue
        best_model = model
        torch.save(best_model.state_dict(), out_dir + '/best_model' + '.pth')
        # evaluate on test with the best dev performance model
        test_acc = evaluate(best_model, test_iter, loss_function, 'Test')

best_model.load_state_dict(torch.load(out_dir + '/best_model' + '.pth'))
evaluate(best_model, test_iter, loss_function, 'Final Test')
