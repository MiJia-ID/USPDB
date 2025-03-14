import torch
from torch.utils.data import Dataset
from sklearn.metrics import matthews_corrcoef,  precision_score, recall_score, f1_score,roc_auc_score, average_precision_score,confusion_matrix
from torch_geometric.utils import dense_to_sparse
import dgl
import pickle
import pandas as pd
from torch.autograd import Variable
from torch.utils import data

from model_with_edge_features import U_MainModel, MainModel

from feature.create_node_feature import create_dataset
from feature.create_graphs import get_coor_test,get_adj_predicted
from feature.create_edge import create_dis_matrix

import argparse

import warnings

import numpy as np

warnings.filterwarnings("ignore")
seed_value = 1995
th=17

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", type=str, default='/home/mijia/EGPDI/data/')
parser.add_argument("--edgefeats_path", type=str, default='/home/mijia/EGPDI/data/Train_Test129/edge_features/EdgeFeats_predicted_SC_17_129.pkl')
parser.add_argument("--model_path", type=str, default='/home/mijia/EGPDI/Model/')
args = parser.parse_args()


features = []
labels = []

class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    def __len__(self):
        return len(self.features)
    def __getitem__(self, index):
        return self.features[index], self.labels[index]


# predicted pdb
root_dir = args.dataset_path
train_path= root_dir + 'DNA-573_Train.txt'
test_path= root_dir + 'DNA-129_Test.txt'
all_702_path = root_dir +  'DNA-702.txt'
pkl_path= root_dir + 'PDNA_residue_feas_PSA.pkl' # 71 dim
esm2_5120_path= root_dir + 'esm2/'
#esm2_33_path = root_dir + 'ESM2-t36/'
dis_path= root_dir + 'PDNA_psepos_SC.pkl'
ProtTrans_path = root_dir +  'prottrans/'
#mas_path = root_dir + 'MSA/'


query_ids = []
with open(train_path, 'r') as f:
    train_text = f.readlines()
    for i in range(0, len(train_text), 3):
        query_id = train_text[i].strip()[1:]
        query_ids.append(query_id)
with open(test_path, 'r') as f:
    train_text = f.readlines()
    for i in range(0, len(train_text), 3):
        query_id = train_text[i].strip()[1:]
        query_ids.append(query_id)


# X,y = create_dataset(query_ids,train_path, test_path,all_702_path, pkl_path,esm2_33_path,esm2_5120_path,ProtTrans_path,mas_path,residue=True,one_hot=True,esm2_33=True,esm_5120=True,prottrans=True,msa=True)
# distance_matrixs=create_dis_matrix(dis_path,query_ids)
X,y = create_dataset(query_ids,train_path, test_path,all_702_path, pkl_path,esm2_5120_path,ProtTrans_path,residue=True,one_hot=True,esm_5120=True,prottrans=True)
distance_matrixs=create_dis_matrix(dis_path,query_ids)

X_test = X[573:]
y_test = y[573:]

NUMBER_EPOCHS = 35
dr=0.3
lr=0.0001
nlayers=4
lamda=1.1
alpha=0.1
atten_time=8

IDs = query_ids[573:]
sequences = []
labels = []
with open(all_702_path,'r') as f:
    train_text = f.readlines()
    for i in range(0, len(train_text), 3):
        seq = train_text[i+1].strip()
        label = train_text[i+2].strip()
        sequences.append(seq)
        labels.append(label)

sequences = sequences[573:]
labels = y_test
features = X_test

coors = get_coor_test(dis_path, query_ids)
adjs = get_adj_predicted(IDs)

graphs = []
for adj in adjs:
    edge_index, _ = dense_to_sparse(adj)
    G = dgl.graph((edge_index[0], edge_index[1])).to(device)
    graphs.append(G)

# edge features
save_edgefeats_path = args.edgefeats_path
with open(save_edgefeats_path, 'rb') as f:
    efeats = pickle.load(f)

test_dic = {"ID": IDs, "sequence": sequences, "label": labels,'features':features,'coors':coors,'adj':adjs,'graph':graphs,'efeats':efeats}

dataframe = pd.DataFrame(test_dic)

class dataSet(data.Dataset):
    def __init__(self,dataframe,adjs):
        self.names = dataframe['ID'].values
        self.sequences = dataframe['sequence'].values
        self.labels = dataframe['label'].values
        self.features = dataframe['features'].values
        self.coors = dataframe['coors'].values
        self.graphs =  dataframe['graph'].values
        self.efeats = dataframe['efeats'].values
        self.adj = dataframe['adj'].values

    def __getitem__(self,index):
        sequence_name = self.names[index]
        sequence = self.sequences[index]
        label = self.labels[index]
        label = self.labels[index]
        node_features = self.features[index]
        coors = self.coors
        coor = coors[index]
        graphs = self.graphs
        graph = graphs[index]
        adj = self.adj[index]

        efeat = self.efeats[index]

        return sequence_name,sequence,label,node_features,graph,efeat,adj,coor

    def __len__(self):
        return len(self.labels)

def graph_collate(samples):
    _,_,label_batch, node_features_batch, graph_batch,efeat_batch,adj_batch,coors_batch = map(list, zip(*samples))
    graph_batch = dgl.batch(graph_batch)
    return label_batch, node_features_batch, graph_batch,efeat_batch,adj_batch,coors_batch

def evaluate(model,data_loader):
    model.eval()
    y_pred_roc =[]
    y_true_roc = []
    y_pred_dot = []
    epoch_loss = 0.0
    n = 0
    valid_pred = []
    valid_true = []

    for label_batch, node_features_batch, graph_batch,efeat_batch,adj_batch,coors_batch in data_loader:

        with torch.no_grad():
            node_features_batch = torch.tensor(node_features_batch)
            coors_batch = torch.tensor(coors_batch)
            adj_batch = adj_batch[0]
            label_batch = label_batch[0]
            efeat_batch = efeat_batch[0]

            if torch.cuda.is_available():
                node_features_batch = Variable(node_features_batch.cuda())
                graph_batch = graph_batch.to(device)
                efeat_batch = efeat_batch.to(device)
                adj_batch = Variable(adj_batch.cuda())
                coors_batch = Variable(coors_batch.cuda())
                y_true = label_batch
            else:
                node_features_batch = Variable(node_features_batch)
                graph_batch = graph_batch
                adj_batch = Variable(adj_batch)
                coors_batch = Variable(coors_batch)
                y_true = label_batch
                efeat_batch = efeat_batch

            y_pred = model(graph_batch, node_features_batch,coors_batch,adj_batch,efeat_batch, device).to(device)  # ori
            softmax = torch.nn.Softmax(dim=1)
            y_pred = softmax(y_pred)
            y_pred = torch.squeeze(y_pred)
            y_pred_dot.append(y_pred)

            y_true_int = [int(label) for label in y_true]
            y_true = torch.tensor(y_true_int, dtype=torch.float32, device=device)

            loss = model.criterion(y_pred,y_true)

            y_pred = y_pred.cpu().detach().numpy()
            y_true = y_true.cpu().detach().numpy()
            y_pred_roc.append(y_pred)
            y_true_roc.append(y_true)

            valid_pred += [pred for pred in y_pred]
            valid_true += list(y_true)

            epoch_loss += loss.item()
            n += 1

    epoch_loss_avg = epoch_loss / n
    return epoch_loss_avg, valid_true, valid_pred


def analysis(y_true,y_pred,best_threshold = None):
    if best_threshold == None:
        best_mcc = 0
        best_threshold = 0

        for j in range(0, 100):
            threshold = j / 100000
            binary_pred = [1 if pred >= threshold else 0 for pred in y_pred]
            binary_true = y_true
            mcc = matthews_corrcoef(binary_true, binary_pred)
            if mcc > best_mcc:
                best_mcc = mcc
                best_threshold = threshold

    binary_pred = [1.0 if pred >= best_threshold else 0.0 for pred in y_pred]
    print(binary_pred)
    pre = precision_score(y_true, binary_pred, zero_division=0)
    recall = recall_score(y_true, binary_pred, zero_division=0)
    f1 = f1_score(y_true, binary_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, binary_pred)

    auc = roc_auc_score(y_true, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_true, binary_pred).ravel()
    spe = tn / (tn + fp)

    results = {
        'spe':spe,
        'precision': pre,
        'recall': recall,
        'f1':f1,
        'mcc': mcc,
        'auc':auc,
        'thred':best_threshold }

    return results


def test_129(Model_Path):
    model = U_MainModel(dr, lr, nlayers, lamda, alpha, atten_time, nfeats=41 + 20 + 1024 + 5120).to(device)
    model.load_state_dict(torch.load(Model_Path,map_location=device), strict=False)

    test_dataSet = dataSet(dataframe=dataframe, adjs=adjs)

    test_loader = torch.utils.data.DataLoader(test_dataSet, batch_size=1, shuffle=False, collate_fn=graph_collate)

    _, test_true, test_pred = evaluate(model, test_loader)

    test_results = analysis(test_true, test_pred)

    return test_results


# trained_model save path
Model_Path_1 = args.model_path + 'Fold1predicted_edgeFeats_best_AUPR_model.pkl'
Model_Path_2 = args.model_path + 'Fold2predicted_edgeFeats_best_AUPR_model.pkl'
Model_Path_3 = args.model_path + 'Fold3predicted_edgeFeats_best_AUPR_model.pkl'
Model_Path_4 = args.model_path + 'Fold4predicted_edgeFeats_best_AUPR_model.pkl'
Model_Path_5 = args.model_path + 'Fold5predicted_edgeFeats_best_AUPR_model.pkl'


print('begin model_fold1 prediction')
test_results_1 = test_129(Model_Path=Model_Path_1)

print('begin model_fold2 prediction')
test_results_2 = test_129(Model_Path=Model_Path_2)

print('begin model_fold3 prediction')
test_results_3 = test_129(Model_Path=Model_Path_3)

print('begin model_fold4 prediction')
test_results_4 = test_129(Model_Path=Model_Path_4)

print('begin model_fold5 prediction')
test_results_5= test_129(Model_Path=Model_Path_5)


spe = max(test_results_1['spe'],test_results_2['spe'],test_results_3['spe'],test_results_4['spe'],test_results_5['spe'])
pre = max(test_results_1['precision'],test_results_2['precision'],test_results_3['precision'],test_results_4['precision'],test_results_5['precision'])
recall = max(test_results_1['recall'],test_results_2['recall'],test_results_3['recall'],test_results_4['recall'],test_results_5['recall'])
f1 = max(test_results_1['f1'],test_results_2['f1'],test_results_3['f1'],test_results_4['f1'],test_results_5['f1'])
mcc = max(test_results_1['mcc'],test_results_2['mcc'],test_results_3['mcc'],test_results_4['mcc'],test_results_5['mcc'])
auc = max(test_results_1['auc'],test_results_2['auc'],test_results_3['auc'],test_results_4['auc'],test_results_5['auc'])
thred = (test_results_1['thred']+test_results_2['thred']+test_results_3['thred']+test_results_4['thred']+test_results_5['thred'])/5

print("Test_129 performance on our method")
print("final spe: {:.3f} ".format(spe))
print("final pre: {:.3f}".format(pre))
print("final recall: {:.3f} ".format(recall))
print("final f1: {:.3f} ".format(f1))
print("final mcc: {:.3f} ".format(mcc))
print("final auc: {:.3f} ".format(auc))


