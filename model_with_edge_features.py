# model
import torch.optim

from modules_with_edge_features import *

class U_MainModel(nn.Module):
    def __init__(self, dr, lr, nlayers, lamda, alpha, atten_time,nfeats):
        super(U_MainModel, self).__init__()

        self.drop1 = nn.Dropout(p=dr)
        self.fc1 = nn.Linear(640 * atten_time, 256)  # for attention
        #self.fc1 = nn.Linear(640, 256)  # for attention
        self.drop2 = nn.Dropout(p=dr)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 1)

        self.rgn_egnn = U_GCN_EGNN(nlayers=2, nfeat=nfeats, nhidden=512, nclass=1, dropout=dr,
                                 lamda=lamda, alpha=alpha, variant=True, heads=1, attention='MTP')
        # self.rgn_gcn2 = RGN_GCN(nlayers=nlayers, nfeat=nfeats, nhidden=128, nclass=1,
        #                         dropout=dr,
        #                         lamda=lamda, alpha=alpha, variant=True, heads=1)
        self.get = GETModelRunner()

        self.multihead_attention = nn.ModuleList(
            [Attention_1(hidden_size=512+128, num_attention_heads=32
                         ) for _ in range(atten_time)])

        # self.block = EfficientAdditiveAttnetion().cuda()

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=1e-16)
        self.fc_get = nn.Linear(6205, 128)

    def forward(self, G, h, x, adj, efeats,device):
        h = torch.squeeze(h)
        x = torch.squeeze(x)
        h = h.to(torch.float32)

        fea1 = self.rgn_egnn(G, h, x, adj, efeats)
        fea1 = torch.unsqueeze(fea1, dim=0)
        fea2 = self.get(G, h, x,efeats,device)
        fea2 = torch.unsqueeze(fea2, dim=0)
        fea2 = self.fc_get(fea2)

        # fea2 = self.rgn_gcn2(h, adj)
        # fea2 = torch.unsqueeze(fea2, dim=0)

        # fea = torch.cat([fea2], dim=2)
        # fea = torch.cat([fea1], dim=2)
        fea = torch.cat([fea1, fea2], dim=2)
        # embeddings = self.block(fea)

        # gated self-attention
        attention_outputs = []
        for i in range(len(self.multihead_attention)):
            multihead_output, _ = self.multihead_attention[i](fea)
            attention_outputs.append(multihead_output)
        embeddings = torch.cat(attention_outputs, dim=2)

        out = self.drop1(embeddings)
        out = self.fc1(out) #dim:256
        out = self.drop2(out)
        out = self.relu1(out)

        out = self.fc2(out)

        return out


