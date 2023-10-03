from __future__ import print_function, division
import argparse
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader
from utils import GeneDataset
cudnn.deterministic = True
cudnn.benchmark = True
import random
seed = 124
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
from torch_geometric.nn import GCNConv
warnings.filterwarnings('ignore')

class AAE_GCN(nn.Module):
    def __init__(self, n_enc_1, n_enc_2, n_dec_1, n_dec_2, n_input, n_z, hidden_dim=256):
        super(AAE_GCN, self).__init__()
        self.n_z = n_z
        self.gcn = GCNConv(1, 1, node_dim=0,improved=True)
        self.enc_1 = nn.Linear(n_input, n_enc_1)
        self.enc_2 = nn.Linear(n_enc_1 , n_enc_2)
        self.enc_3 = nn.Linear(n_enc_2, n_z)

        self.dec_1 = nn.Linear(n_z, n_dec_2)
        self.dec_2 = nn.Linear(n_dec_2, n_dec_1)
        self.dec_3 = nn.Linear(n_dec_1, n_input)

        self.discriminator = nn.Sequential(
            nn.Linear(n_z, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, data, edge_index):
        data = data + torch.randn(data.shape).to(device) * 0.1
        edge_index = edge_index.to(device)
        for i in range(data.shape[0]):
            x = torch.as_tensor(data[i].reshape(data.shape[1], 1), dtype=torch.float)
            y = self.gcn(x, edge_index)
            y = F.leaky_relu(y).t()
            if i == 0:
                gcn_out = y
            else:
                gcn_out = torch.cat([gcn_out, y], dim=0)

        enc_h1 = F.leaky_relu(self.enc_1(gcn_out))
        enc_h2 = F.leaky_relu(self.enc_2(enc_h1))
        enc_h3 = F.leaky_relu(self.enc_3(enc_h2))

        dec_h1 = F.leaky_relu(self.dec_1(enc_h3))
        dec_h2 = F.leaky_relu(self.dec_2(dec_h1))
        dec_h3 = F.leaky_relu(self.dec_3(dec_h2))

        z = enc_h3
        d = self.discriminator(z)

        return dec_h3, z, d


class GDEC(nn.Module):

    def __init__(self,
                 n_enc_1,
                 n_enc_2,
                 n_dec_1,
                 n_dec_2,
                 n_z,
                 n_input,
                 n_clusters,
                 alpha=1.0,
                 pretrain_path='data/aae_gcn_pre.pkl'):
        super(GDEC, self).__init__()
        self.alpha = 1.0
        self.pretrain_path = pretrain_path
        self.aae_gcn = AAE_GCN(
            n_enc_1=n_enc_1,
            n_enc_2=n_enc_2,
            n_dec_1=n_dec_1,
            n_dec_2=n_dec_2,
            n_z=n_z,
            n_input=n_input)

        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_z))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

    def pretrain(self, path=''):
        if path == '':
            pretrain_aae_gcn(self.aae_gcn)
        self.aae_gcn.load_state_dict(torch.load(self.pretrain_path))
        print('load pretrained aae_gcn from', path)

    def forward(self, x ,edge):
        x_bar, z ,d= self.aae_gcn(x,edge)
        z_real = torch.randn(x.size(0),self.aae_gcn.n_z)
        z_real = z_real.to(device)

        d_real = self.aae_gcn.discriminator(z_real)
        d_fake = self.aae_gcn.discriminator(z)

        q = 1.0 / (1.0 + torch.sum(
            torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.alpha)
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return x_bar,q,d_real,d_fake


def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


def pretrain_aae_gcn(model):

    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    df = pd.read_csv('edge_index.csv', header=None)
    edge_index = torch.tensor(df.values)
    edge_index = edge_index.t().contiguous()
    print(edge_index)
    print("??")
    print(model)


    optimizer = Adam(model.parameters(), lr=args.lr)
    for epoch in range(args.pretrain_epochs):
        for batch_idx, (x, _) in enumerate(train_loader):
            optimizer.zero_grad()
            x = x.to(device)
            x_bar, z, _ = model(x,edge_index)
            z_real = torch.randn(x.size(0), model.n_z)
            z_real = z_real.to(device)
            d_real = model.discriminator(z_real)
            d_fake = model.discriminator(z)

            encoder_decoder_loss = F.mse_loss(x_bar, x)
            generator_loss = nn.BCELoss()(d_fake, torch.ones_like(d_fake))
            discriminator_loss = nn.BCELoss()(d_fake, torch.zeros_like(d_fake)) + \
                                 nn.BCELoss()(d_real, torch.ones_like(d_real))
            tot_loss = discriminator_loss + generator_loss

            all_loss=tot_loss+encoder_decoder_loss
            all_loss.backward()
            optimizer.step()

        print(
            f'Epoch [{epoch + 1}], encoder_decoder_loss Loss: {encoder_decoder_loss.item():.8f},  generator_loss Loss: {generator_loss.item():.8f},  discriminator_loss Loss: {discriminator_loss.item():.8f},  tot_loss Loss: {tot_loss.item():.8f}')
        torch.save(model.state_dict(), args.pretrain_path)

    print("model saved to {}.".format(args.pretrain_path))


def train_gdec():
    model = GDEC(
        n_enc_1=512,
        n_enc_2=256,
        n_dec_1=512,
        n_dec_2=256,
        n_input=args.n_input,
        n_z=args.n_z,
        n_clusters=args.n_clusters,
        alpha=1.0,
        pretrain_path=args.pretrain_path).to(device)

    model.pretrain()

    train_loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False)
    df = pd.read_csv('edge_index.csv', header=None)
    edge_index = torch.tensor(df.values).to(device)
    edge_index = edge_index.t().contiguous()
    print(edge_index)
    optimizer = Adam(model.parameters(), lr=args.lr)

    data = dataset.x

    data = torch.Tensor(data).to(device)
    x_bar, hidden ,_= model.aae_gcn(data,edge_index)

    kmeans = KMeans(n_clusters=args.n_clusters, n_init=20)
    y_pred = kmeans.fit_predict(hidden.data.cpu().numpy())
    silhouetteScore = silhouette_score(hidden.data.cpu().numpy(), y_pred, metric='euclidean')
    davies_bouldinScore = davies_bouldin_score(hidden.data.cpu().numpy(), y_pred)
    print("silhouetteScore={:.4f}".format(silhouetteScore),', davies_bouldinScore {:.4f}'.format(davies_bouldinScore))

    hidden = None
    x_bar = None

    y_pred_last = y_pred
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)

    model.train()
    for epoch in range(args.train_epochs):

        if epoch % args.update_interval == 0:
            _, tmp_q, _, _ = model(data,edge_index)
            _, hidden,_ = model.aae_gcn(data,edge_index)

            tmp_q = tmp_q.data
            p = target_distribution(tmp_q)

            y_pred = tmp_q.cpu().numpy().argmax(1)

            while len(set(y_pred)) < 2:
                kmeans = KMeans(n_clusters=args.n_clusters,  n_init=20)
                y_pred = kmeans.fit_predict(hidden.data.cpu().numpy())

            delta_label = np.sum(y_pred != y_pred_last).astype(
                np.float32) / y_pred.shape[0]
            y_pred_last = y_pred
            silhouetteScore = silhouette_score(hidden.data.cpu().numpy(), y_pred, metric='euclidean')
            davies_bouldinScore = davies_bouldin_score(hidden.data.cpu().numpy(), y_pred)
            print('Iter {}'.format(epoch), ':silhouetteScore {:.4f}'.format(silhouetteScore),
                  ', davies_bouldinScore {:.4f}'.format(davies_bouldinScore),
                  ', delta_label {:.4f}'.format(delta_label))
            d = np.column_stack((hidden.data.cpu().numpy(), y_pred))
            np.savetxt(r'x_tsne.csv', d, delimiter=',')

            x1 = d[d[:, -1] == 0]
            x2 = d[d[:, -1] == 1]
            x3 = d[d[:, -1] == 2]
            x4 = d[d[:, -1] == 3]
            x5 = d[d[:, -1] == 4]
            x6 = d[d[:, -1] == 5]
            x7 = d[d[:, -1] == 6]
            x8 = d[d[:, -1] == 7]
            x9 = d[d[:, -1] == 8]

            plt.scatter(hidden.data.cpu().numpy()[:, 0], hidden.data.cpu().numpy()[:, 1], c="red", marker='o',
                        label='see')
            plt.scatter(x1[:, 0], x1[:, 1], c="red", marker='o', label='label0')
            plt.scatter(x2[:, 0], x2[:, 1], c="green", marker='*', label='label1')
            plt.scatter(x3[:, 0], x3[:, 1], c="blue", marker='+', label='label2')
            plt.scatter(x4[:, 0], x4[:, 1], c="yellow", marker='o', label='label3')
            plt.scatter(x5[:, 0], x5[:, 1], c="purple", marker='*', label='label4')
            plt.scatter(x6[:, 0], x6[:, 1], c="brown", marker='+', label='label5')
            plt.scatter(x7[:, 0], x7[:, 1], c="pink", marker='o', label='label6')
            plt.scatter(x8[:, 0], x8[:, 1], c="yellowgreen", marker='*', label='label7')
            plt.scatter(x9[:, 0], x9[:, 1], c="skyblue", marker='+', label='label8')
            plt.xlabel('petal length')
            plt.ylabel('petal width')
            plt.legend(loc=2)
            plt.savefig('brca.png')
            plt.show()


            if epoch > 0 and delta_label < args.tol:
                print('delta_label {:.4f}'.format(delta_label), '< tol',
                      args.tol)
                print('Reached tolerance threshold. Stopping training.')
                break
        for batch_idx, (x, idx) in enumerate(train_loader):

            x = x.to(device)

            idx = idx.to(device)
            idx = idx.long()

            x_bar, q,d_real,d_fake = model(x,edge_index)

            reconstr_loss = F.mse_loss(x_bar, x)
            generator_loss = nn.BCELoss()(d_fake, target=torch.ones_like(d_fake))
            discriminator_loss = nn.BCELoss()(d_fake, target=torch.zeros_like(d_fake)) + \
                                 nn.BCELoss()(d_real, target=torch.ones_like(d_real))

            tot_loss = discriminator_loss + generator_loss
            kl_loss = F.kl_div(q, p[idx])
            loss =  kl_loss + args.gamma * (reconstr_loss+tot_loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--n_clusters', default=3, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--n_z', default=2, type=int)
    parser.add_argument('--dataset', type=str, default='data')
    parser.add_argument('--pretrain_path', type=str, default='data/aae_gcn_pre')
    parser.add_argument(
        '--gamma',
        default=0.5,
        type=float,
        help='coefficient of clustering loss')
    parser.add_argument('--update_interval', default=1, type=int)
    parser.add_argument('--tol', default=0.001, type=float)
    parser.add_argument('--pretrain_epochs', default=50, type=int)
    parser.add_argument('--train_epochs', default=500, type=int)
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))
    device = torch.device("cuda" if args.cuda else "cpu")

    if args.dataset == 'data':
        args.pretrain_path = 'data/aae_gcn_pre.pkl'
        args.n_clusters = 2
        dataset = GeneDataset()
        args.pretrain_epochs = 350
        args.train_epochs = 50
        args.n_z = 10
        args.gamma = 0.1
        args.n_input = dataset.x.shape[1]
        args.tol = 0.001
        args.lr = 1e-5
        args.batch_size = 16
    print(args)
    train_gdec()



