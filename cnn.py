import torch
import numpy as np
import torch.nn as nn
import time
from tqdm import tqdm
import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import defaultdict
import mmap


emb_dim = 300
glove_file = os.path.join(".", "glove.6B", "glove.6B.{}d.txt".format(emb_dim))
glove_switch = True
train_file = os.path.join(".", "train.csv")
test_file = os.path.join(".", "test.csv")
sample_file = os.path.join(".", "sample_submission.csv")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')


batch_size = 500
epoch_total = 500
hidden_dim = 400
hidden_layer_num = 2
sentence_max_length = 512
title_len_maximum = 50
feat_num = 8
lr = 0.02
momen = 0.95

model_save_file_name = "model.torch"

# target_name_lst = ["question_body_critical", "question_opinion_seeking", "question_type_instructions", "question_type_reason_explanation", "answer_type_instructions", "answer_type_reason_explanation"]
target_name_lst = ["answer_type_reason_explanation"]


def get_num_lines(file_path):
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines


def text_cleaning(df, cols):
    puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#',
              '*', '+', '\\', '•', '~', '@', '£',
              '·', '_', '{', '}', '©', '^', '®', '`', '<', '→', '°', '€', '™', '›', '♥', '←', '×', '§', '″', '′', 'Â',
              '█', '½', 'à', '…', '\n', '\xa0', '\t',
              '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―',
              '¥', '▓', '—', '‹', '─', '\u3000', '\u202f',
              '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸',
              '¾', 'Ã', '⋅', '‘', '∞', '«',
              '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø',
              '¹', '≤', '‡', '√', ]

    mispell_dict = {"aren't": "are not",
                    "can't": "cannot",
                    "couldn't": "could not",
                    "couldnt": "could not",
                    "didn't": "did not",
                    "doesn't": "does not",
                    "doesnt": "does not",
                    "don't": "do not",
                    "hadn't": "had not",
                    "hasn't": "has not",
                    "haven't": "have not",
                    "havent": "have not",
                    "he'd": "he would",
                    "he'll": "he will",
                    "he's": "he is",
                    "i'd": "I would",
                    "i'd": "I had",
                    "i'll": "I will",
                    "i'm": "I am",
                    "isn't": "is not",
                    "it's": "it is",
                    "it'll": "it will",
                    "i've": "I have",
                    "let's": "let us",
                    "mightn't": "might not",
                    "mustn't": "must not",
                    "shan't": "shall not",
                    "she'd": "she would",
                    "she'll": "she will",
                    "she's": "she is",
                    "shouldn't": "should not",
                    "shouldnt": "should not",
                    "that's": "that is",
                    "thats": "that is",
                    "there's": "there is",
                    "theres": "there is",
                    "they'd": "they would",
                    "they'll": "they will",
                    "they're": "they are",
                    "theyre": "they are",
                    "they've": "they have",
                    "we'd": "we would",
                    "we're": "we are",
                    "weren't": "were not",
                    "we've": "we have",
                    "what'll": "what will",
                    "what're": "what are",
                    "what's": "what is",
                    "what've": "what have",
                    "where's": "where is",
                    "who'd": "who would",
                    "who'll": "who will",
                    "who're": "who are",
                    "who's": "who is",
                    "who've": "who have",
                    "won't": "will not",
                    "wouldn't": "would not",
                    "you'd": "you would",
                    "you'll": "you will",
                    "you're": "you are",
                    "you've": "you have",
                    "'re": " are",
                    "wasn't": "was not",
                    "we'll": " will",
                    "didn't": "did not",
                    "tryin'": "trying"}

    def clean_text(x):
        x = str(x)
        for punct in puncts:
            x = x.replace(punct, f' {punct} ')
        return x

    def clean_numbers(x):
        x = re.sub('[0-9]{5,}', '#####', x)
        x = re.sub('[0-9]{4}', '####', x)
        x = re.sub('[0-9]{3}', '###', x)
        x = re.sub('[0-9]{2}', '##', x)
        return x

    def _get_mispell(mispell_dict):
        mispell_re = re.compile('(%s)' % '|'.join(mispell_dict.keys()))
        return mispell_dict, mispell_re

    def replace_typical_misspell(text):
        mispellings, mispellings_re = _get_mispell(mispell_dict)

        def replace(match):
            return mispellings[match.group(0)]

        return mispellings_re.sub(replace, text)

    def clean_data(df, columns: list):
        for col in columns:
            col_n = col + "_clean"
            df[col] = df[col].apply(lambda x: clean_numbers(x))
            df[col] = df[col].apply(lambda x: replace_typical_misspell(x.lower()))
            df[col] = df[col].apply(lambda x: clean_text(x.lower()))
            # df[col_n] = df[col].apply(lambda x: clean_numbers(x))
            # df[col_n] = df[col_n].apply(lambda x: replace_typical_misspell(x.lower()))
            # df[col_n] = df[col_n].apply(lambda x: clean_text(x.lower()))

        return df

    return clean_data(df, cols)


def pre_process_reclass(df, cols):

    total_len = len(df)
    ttls = []
    for col in cols:
        count = defaultdict(int)
        replace = {}
        for index, row in df.iterrows():
            count[row[col]] += 1
        for key in count.keys():
            if count[key] < total_len * 0.05:
                replace[key] = 0
                for key2 in count.keys():
                    if key2>replace[key] and key2<key:
                        replace[key] = key2
            else:
                replace[key] = key
        final_set = set(replace.values())
        f_ind = {}
        iint = 0
        for val in final_set:
            f_ind[val] = iint
            iint += 1
        ls = []
        for index, row in df.iterrows():
            ls.append(f_ind[replace[row[col]]])
        df[col] = ls
        ttls.append(iint)

    return ttls


def form_tok_seq(t, q, a):
    t_ls = t.split()
    q_ls = q.split()
    a_ls = a.split()
    seq = []
    if len(t_ls) > title_len_maximum:
        seq.extend(t_ls[:title_len_maximum])
    else:
        seq.extend(t_ls)

    rest_max = sentence_max_length - len(seq)
    q_max = rest_max // 2
    a_max = rest_max - q_max
    if len(q_ls) <= q_max and len(a_ls) <= a_max:
        seq.extend(q_ls)
        seq.extend(a_ls)
    elif len(q_ls) > q_max and len(a_ls) > a_max:
        seq.extend(q_ls[:q_max])
        seq.extend(a_ls[:a_max])
    elif len(q_ls) <= q_max and len(a_ls) > a_max:
        seq.extend(q_ls)
        seq.extend(a_ls[:rest_max-len(q_ls)])
    else:
        seq.extend(q_ls[:rest_max-len(a_ls)])
        seq.extend(a_ls)

    if len(seq) == sentence_max_length:
        return seq
    tmp = [" "] * (sentence_max_length-len(seq))
    tmp.extend(seq)
    return tmp


def organize_embeding(t, q, a, ddict, index):
    seq = form_tok_seq(t, q, a)
    for tok in seq:
        if tok not in ddict:
            ddict[tok] = index
            index += 1
    return index


def form_embeding(t, q, a, ddict):
    seq = form_tok_seq(t, q, a)
    ls = []
    for tok in seq:
        ls.append(ddict[tok])
    return ls




class cnn_net(nn.Module):
    def __init__(self, sent_len, output_dim,
                 init_weight, embed_index, embed_dim, feat_num=feat_num, out_channels=16):
        super().__init__()

        self.sent_len = sent_len
        self.output_dim = output_dim
        self.init_weight = init_weight
        self.embed_index = embed_index
        self.embed_dim = embed_dim
        self.feat_num = feat_num
        self.out_channels = out_channels

        # embedding layer
        self.embed = torch.nn.Embedding(embed_index, embed_dim)
        with torch.no_grad():
            # print(self.embed.weight.shape)
            self.embed.weight.data = torch.Tensor(init_weight)
            self.embed.weight.requires_grad = True

        # batch_size * sent_length * embed_dim
        # reshape --> batch_size * 1 * sent_length * embed_dim

        self.cov_ls = {}
        self.pool_ls = {}
        for i in range(2, 10):
            # cnn-i layer
            self.cov_ls[i] = torch.nn.Conv2d(in_channels=1, out_channels=out_channels, kernel_size=(i, embed_dim), stride=1).to(device)
            # batch_size * out_channels * (sent_length - i + 1) * 1
            # reshape --> batch_size * out_channels * (sent_length - i + 1)
            self.pool_ls[i] = torch.nn.MaxPool1d(sent_len - i + 1).to(device)
            # reshape --> batch_size * out_channels


        # concatenate
        # --> batch_size * (8*out_channels)

        # nn layer
        self.fc1 = torch.nn.Linear(8 * out_channels, hidden_dim)
        self.act = torch.nn.Sigmoid()
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        z = self.embed(x.long())
        sh = z.shape
        # print("orig:", sh)
        for i in range(2, 10):
            zz = z.reshape((sh[0], 1, self.sent_len, self.embed_dim))
            # print("1: ", zz.shape)
            zz = self.cov_ls[i](zz)
            # print("2: ", zz.shape)
            zz = zz.reshape((sh[0], self.out_channels, self.sent_len - i + 1))
            # print("3: ", zz.shape)
            zz = self.pool_ls[i](zz)
            # print("4: ", zz.shape)
            zz = zz.reshape((sh[0], self.out_channels))
            # print("5: ", zz.shape)

            try:
                rec = torch.cat((rec, zz), dim=1)
            except:
                rec = zz

        # print(rec.shape)
        return self.fc2(self.act(self.fc1(rec)))


def find_max(ls):
    pos = 0
    for i in range(1, len(ls)):
        if ls[i] > ls[pos]:
            pos = i
    return pos

def test_data(test_dataloader, net, criterion):

    total_loss = 0.0
    total_count = 0
    correct_count = 0
    for i, (x, y) in enumerate(test_dataloader):
        x = x.to(device)
        y = y.to(device).reshape((y.shape[0], ))

        y_pred = net(x)
        loss = criterion(y_pred, y)
        total_loss += loss.item() * y_pred.shape[0]

        for j in range(y_pred.shape[0]):
            if find_max(y_pred[j]) == y[j]:
                correct_count += 1

        total_count += y_pred.shape[0]

    acc = []
    return total_loss / total_count, correct_count / total_count


def train_data(train_dataloader, test_dataloader, net, criterion, optimizer, epoch_total):

    print("Training ... ")
    val_acc_best = 0
    for epoch in range(epoch_total):
        print("@ Starting epoch {}".format(epoch))
        start = time.time()
        running_loss = 0.0

        for i, (x, y) in enumerate(train_dataloader):
            x = x.to(device)
            y = y.to(device).reshape((y.shape[0], ))

            optimizer.zero_grad()
            y_pred = net(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            print("    @ Epoch {}, iter {}, loss: {}".format(
                epoch, i, loss.item()
            ))

        test_loss, test_acc = test_data(test_dataloader, net, criterion)
        print(" Finish epoch. val loss: {}, val accuracy: {}%.".format(
            test_loss, test_acc*100
        ))
        if test_acc > val_acc_best:
            val_acc_best = test_acc
            torch.save(net, model_save_file_name)

    print("Best accuracy:", val_acc_best)
    print("Training finished.")


if __name__ == "__main__":

    print("Loading files.")

    # Loading files
    total_data = pd.read_csv(train_file)
    test = pd.read_csv(test_file)
    sample = pd.read_csv(sample_file)

    score_name_lst = list(sample.columns)[1:]
    content_name_lst = ["question_title", "question_body", "answer"]
    len_total = len(total_data[score_name_lst[0]])
    total_data = text_cleaning(total_data, content_name_lst)

    ttls = pre_process_reclass(total_data, target_name_lst)

    print("Total data size: ", len_total)

    # print(total_data[target_name_lst[0]])

    train, test = train_test_split(total_data, test_size=0.1, random_state=0)

    print("Train data size: ", len(train))
    print("Test data size: ", len(test))


    # organize embedding
    print("Organize embedding.")
    word_dict = {}
    word_dict[" "] = 0
    index = 1
    for i, row in total_data.iterrows():
        index = organize_embeding(row[content_name_lst[0]], row[content_name_lst[1]], row[content_name_lst[2]], word_dict, index)

    # glove
    print("Loading glove.")
    word2emb = {}
    if glove_switch:
        print("Start loading glove.")
        with open(glove_file, 'rb') as f:
            for line in tqdm(f, total=get_num_lines(glove_file)):
                vec = line.strip().split()
                word2emb[vec[0]] = np.array(vec[1:]).astype(np.float)
        print("Finish loading glove.")

    # init_weight
    print("Build init weight")
    init_weight = np.zeros((index, emb_dim))
    for key in word_dict:
        if key in word2emb.keys():
            init_weight[word_dict[key]] = word2emb[key].copy()
        else:
            init_weight[word_dict[key]] = np.random.randn(emb_dim, )

    train_x = np.zeros((len(train), sentence_max_length))
    train_y = np.zeros((len(train), len(target_name_lst)))
    test_x = np.zeros((len(test), sentence_max_length))
    test_y = np.zeros((len(test), len(target_name_lst)))

    j = 0
    for i, row in tqdm(train.iterrows()):
        ls = form_embeding(row[content_name_lst[0]], row[content_name_lst[1]], row[content_name_lst[2]], word_dict)
        train_x[j] = np.array(ls)
        ls = []
        for name in target_name_lst:
            ls.append(row[name])
        train_y[j] = np.array(ls)
        j += 1

    j = 0
    for i, row in tqdm(test.iterrows()):
        ls = form_embeding(row[content_name_lst[0]], row[content_name_lst[1]], row[content_name_lst[2]], word_dict)
        test_x[j] = np.array(ls)
        ls = []
        for name in target_name_lst:
            ls.append(row[name])
        test_y[j] = np.array(ls)
        j += 1


    # print(train_x.shape)
    # print(train_y.shape)
    # print(test_x.shape)
    # print(test_y.shape)
    # print(train_x[:5])
    # print(train_y[:5])

    # data loader
    train_x = torch.LongTensor(train_x)
    train_y = torch.LongTensor(train_y)
    test_x = torch.LongTensor(test_x)
    test_y = torch.LongTensor(test_y)

    train_dataset = torch.utils.data.TensorDataset(train_x, train_y)
    test_dataset = torch.utils.data.TensorDataset(test_x, test_y)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    print(ttls)

    # for x, y in train_dataloader:
    #     print(x, y)
    #     print(x.shape, y.shape)

    net = cnn_net(sentence_max_length, ttls[0],
                  init_weight, index, emb_dim).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizor = torch.optim.SGD(net.parameters(), lr=lr, momentum=momen)

    train_data(train_dataloader, test_dataloader, net, criterion, optimizor, 100)
    test_data(test_dataloader, net, criterion)













