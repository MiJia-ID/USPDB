import pickle
import numpy as np
import torch

train_list=[]
seqanno= {}
Query_ids=[]
query_seqs=[]
query_annos=[]


def one_hot_encode(sequence):
    # create one-hot
    # print('len of one-hot',len(sequence))

    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    aa_to_int = {aa: i for i, aa in enumerate(amino_acids)}
    encoded_sequence = np.zeros((len(sequence), len(amino_acids)), dtype=np.float32)
    for i, aa in enumerate(sequence):
        encoded_sequence[i, aa_to_int[aa]] = 1
    return encoded_sequence


def create_features(query_ids,test_path,pkl_path,esm2_5120_path,ProtTrans_path):
    '''
        load node features :
        1) Esm2-t48(5120dim),Esm2-t36(33dim),ProtTrans(1024dim),
        2) Residual_feats(HMM-20dim,PSSM-30dim,AF-7dim,DSSP-14dim)
    '''

    with open(test_path, 'r') as f:
        train_text = f.readlines()
        for i in range(0, len(train_text), 3):
            query_id = train_text[i].strip()[1:]
            if query_id[-1].islower():
                #query_id += query_id[-1]
                query_id = query_id[:-1] + query_id[-1].upper()
                print(query_id,'-'*1000)
            query_seq = train_text[i + 1].strip()
            query_anno = train_text[i + 2].strip()
            train_list.append(query_id)
            seqanno[query_id] = {'seq': query_seq, 'anno': query_anno}
            Query_ids.append(query_id)
            query_seqs.append(query_seq)

    # load one-hot
    query_seqs_181 = []
    with open(test_path, 'r') as f1:
        text_181 = f1.readlines()
        for i in range(1, len(text_181), 3):     
            query_seq_181 = text_181[i].strip()
            #if query_seq_181 != 'RRNRRLSSASVYRYYLKRISMNIGTTGHVNGLSIAGNPEIMRAIARLSEQETYNWVTDYAPSHLAKEVVKQISGKYNIPGAYQGLLMAFAEKVLANYILDYKGEPLVEIHHNFLWELMQGFIYTFVRKDGKPVTVDMSKVLTEIEDALFKLVKK':
            query_seqs_181.append(query_seq_181)
    encoded_proteins = [one_hot_encode(sequence) for sequence in query_seqs_181]

    # load Residual_feats-71dim
    PDNA_residue_load=open(pkl_path,'rb')
    PDNA_residue=pickle.load(PDNA_residue_load)

    # load esm2-t36 embeddings-33dim
    ESM2_33 = []
    paths = []
    # for i in query_ids:
    #     file_paths = esm2_33_path + '{}'.format(i) + '.npy'
    #     paths.append(file_paths)
    for file_path in paths:
        ESM2_33_embedding = np.load(file_path)
        ESM2_33.append(ESM2_33_embedding)

    # load esm2-t48 embeddings-5120dim
    ESM2_5120 = []
    paths_5120 = []
    for i in query_ids:
        # file_paths = esm2_5120_path + '{}'.format(i) + '.rep_5120.npy'
        if i == '6wq2_aa':
            i = '6wq2_a'
        file_paths = esm2_5120_path + '{}'.format(i) + '.npy'
        paths_5120.append(file_paths)
    for file_path in paths_5120:
        # print(file_path)
        # ESM2_5120_embedding = np.load(file_path,allow_pickle=True)
        ESM2_5120_embedding = np.load(file_path,allow_pickle=True)
        ESM2_5120.append(ESM2_5120_embedding)

    # load MSA embeddings-256dim
    MSA_256 = []
    paths_256 = []
    # for i in query_ids:
    #     # file_paths = esm2_5120_path + '{}'.format(i) + '.rep_5120.npy'
    #     file_paths = msa_256_path + '{}'.format(i) + 'msa_first_row.npy'
    #     paths_256.append(file_paths)
    for file_path in paths_256:
        # print(file_path)
        msa_256_embedding = np.load(file_path, allow_pickle=True)
        MSA_256.append(msa_256_embedding)

    # load ProtTrans embeddings-1024dim
    ProTrans_1024=[]
    paths_1024 = []
    for i in query_ids:
        if i == '6wq2_aa':
            i = '6wq2_a'
        file_paths = ProtTrans_path + '{}'.format(i) + '.npy'
        paths_1024.append(file_paths)
    for file_path in paths_1024:
        ProTrans_1024_embedding = np.load(file_path, allow_pickle=True)
        ProTrans_1024.append(ProTrans_1024_embedding)

    
    # load residue features-71dim and labels
    data = {}
    for i in query_ids:
        data[i] = []
        if i == '6wq2_aa':
            i = '6wq2_a'
        residues = PDNA_residue[i]
        labels = seqanno[i]['anno']
        data[i].append({'features': residues,'label': labels})


    feature1=[]
    feature2=[]
    feature3=[]
    feature4 = []
    feature5 = []
    feature6 = []

    protein_labels=[]

    for i in query_ids:
        if i == '6wq2_aa':
            i = '6wq2_a'
        residues=data[i]
        feature1.append(residues[0]['features'])
        protein_labels.append((residues[0]['label']))

    for j in range(len(query_ids)):
        if 0 <= j < len(encoded_proteins):  # 确保 j 在有效范围内
            feature2.append(encoded_proteins[j])
        else:
            print(len(query_ids))
            print(len(encoded_proteins))
            print("警告：索引 j 超出了 encoded_proteins 的范围。")
        #feature2.append(encoded_proteins[j])   # 20 dim one-hot
        feature3.append(ESM2_5120[j])    # 5120 dim bert_5120
        feature4.append(ProTrans_1024[j])  # 1024 dim protrans
        # feature5.append(ESM2_33[j])
        # feature6.append(MSA_256[j])  # 256 dim MSA


    node_features={}
    for i in range(len(query_ids)):
        node_features[query_ids[i]]={'seq': i+1,'residue_fea': feature1[i],'esm2_5120':feature3[i],'prottrans_1024':feature4[i],'one-hot':feature2[i],'label':protein_labels[i]}

    return node_features


def create_dataset(query_ids,test_path,pkl_path,esm2_5120_path,ProtTrans_path,residue,one_hot,esm_5120,prottrans):
    '''
    :param query_ids: all protein ids
    :param train_path: training set file path
    :param test_path: test_129 set file path
    :param all_702_path: train_573 and test_129 file path
    :param pkl_path: residue features path
    :param esm2_33_path: esm2-t36 embeddings path
    :param esm2_5120_path: esm2-t48 embeddings path
    :param ProtTrans_path: ProtTrans embeddings path
    :param residue: add residue features or not
    :param one_hot: add one-hot features or not
    :param esm2_33: add esm2-t36 features or not
    :param esm_5120: add esm2-t48 features or not
    :param prottrans: add ProtTrans features or not
    :return: X and y, involving training and validation set
    '''

    X=[]
    y=[]
    features={}

    # all 702 protein information
    # query_ids,test_path,pkl_path,msa_256_path,esm2_5120_path,ProtTrans_path
    node_features = create_features(query_ids,test_path,pkl_path,esm2_5120_path,ProtTrans_path)
    for i in query_ids:
        protein = node_features[i]

        mat1 = (protein['residue_fea'])
        if i == 1:
            protein = node_features[i+1]
        mat2 = (protein['one-hot'])
        # mat3 = (protein['esm2_33'])
        mat4 = (protein['esm2_5120'])
        mat5 = (protein['prottrans_1024'])
        # mat6 = (protein['msa_256'])

        mat4 = torch.Tensor(mat4)
        mat4 = torch.squeeze(mat4)

        mat5 = torch.Tensor(mat5)
        mat5 = torch.squeeze(mat5)


        # different feature combinations
        # handy drafted features and protein language model embeddings
        if residue == True and one_hot == True and esm_5120 == True and prottrans == True :  # all embeddings
            try:
                # 尝试拼接数组
                features[i] = np.hstack((mat1, mat2, mat4, mat5))
            except ValueError as e:
                # 捕获 ValueError 并跳过
                print(f"跳过拼接操作：{e}")
            #features[i] = np.hstack((mat1, mat2, mat4, mat5))
        # handy drafted features
        elif residue == True and one_hot == True and esm_5120 == False and prottrans == False :  # only hand drafted features protein language model embeddings
            features[i] = np.hstack((mat1, mat2))
        # protein language model embeddings
        elif residue == False and one_hot == False and esm_5120 == True and prottrans == True :  # only protein language model embeddings
            features[i] = np.hstack((mat4, mat5))
        elif residue == True and one_hot == True and esm_5120 == False and prottrans == False:
            features[i] = np.hstack((mat1, mat2))
            # features[i] = np.hstack((mat1,mat3,mat5))
        elif residue == True and one_hot == True and esm_5120 == False and prottrans == True:
            features[i] = np.hstack((mat1, mat2, mat5))

        labels = protein['label']
        y.append(labels)

    for key in query_ids:
        # try:
        #     X.append(features[key])
        # except KeyError:
        #     # 捕获 KeyError 并跳过
        #     print(key,"不存在，跳过处理。")
        if key == '6wq2_aa':
            key = '6wq2_a'
        X.append(features[key])


    return X,y




