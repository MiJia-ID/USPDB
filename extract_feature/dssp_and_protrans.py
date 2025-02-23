import torch
from transformers import T5EncoderModel, T5Tokenizer
import re, argparse
import numpy as np
from tqdm import tqdm
import gc
import multiprocessing
import os, datetime
from Bio import pairwise2
import pickle

def get_prottrans(fasta_file, output_path):
    # 设置环境变量以限制OpenMP使用的线程数
    os.environ["OMP_NUM_THREADS"] = "1"
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default='1')
    args = parser.parse_args()
    gpu = args.gpu

    ID_list = []
    seq_list = []
    with open(fasta_file, "r") as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()  # 移除行末尾的换行符和空白字符
        if line and line[0] == ">":  # 检查行是否非空且是否为ID行
            ID_list.append(line[1:])  # 去除ID行的大于号
        elif line:  # 检查行是否非空
            seq_list.append(" ".join(list(line)))  # 将非空行的序列加入序列列表中

    for id, seq in zip(ID_list[:9], seq_list[:9]):  # 仅作为示例，打印前5个序列及其ID
        print(f"ID: {id}")
        print(f"Sequence: {seq[:]}...")  # 打印序列的前50个字符（加空格后）作为示例
        print("len:",len(seq))

    model_path = "/home/mijia/app/prottrans/Prot-T5-XL-U50"
    tokenizer = T5Tokenizer.from_pretrained(model_path, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(model_path)
    gc.collect()

    # 设置设备
    device = torch.device('cuda:' + gpu if torch.cuda.is_available() and gpu else 'cpu')
    model = model.eval().to(device)

    print(next(model.parameters()).device)
    print('starttime')
    starttime = datetime.datetime.now()
    print(starttime)
    batch_size = 1

    for i in tqdm(range(0, len(ID_list), batch_size)):
        batch_ID_list = ID_list[i:i + batch_size]
        batch_seq_list = seq_list[i:i + batch_size]

        # 检查当前批次的所有输出文件是否已经存在
        all_files_exist = True
        for seq_id in batch_ID_list:
            out_file_path = os.path.join(output_path, seq_id + ".npy")
            if not os.path.exists(out_file_path):
                all_files_exist = False
                break  # 如果发现有文件不存在，则无需继续检查

        # 如果当前批次所有输出文件都存在，跳过此批次
        if all_files_exist:
            print(f"批次 {i // batch_size + 1} 已处理，跳过。")
            continue

        # 处理序列
        batch_seq_list = [re.sub(r"[UZOB]", "X", sequence) for sequence in batch_seq_list]

        # 编码序列
        ids = tokenizer.batch_encode_plus(batch_seq_list, add_special_tokens=True, padding=True)
        input_ids = torch.tensor(ids['input_ids']).to(device)
        attention_mask = torch.tensor(ids['attention_mask']).to(device)

        # 提取特征
        with torch.no_grad():
            embedding = model(input_ids=input_ids, attention_mask=attention_mask)
        embedding = embedding.last_hidden_state.cpu().numpy()

        # 打印特征的尺寸大小
        print("特征尺寸大小:", embedding.shape)

        # 保存特征
        for seq_num in range(len(embedding)):
            seq_len = (attention_mask[seq_num] == 1).sum()
            seq_emd = embedding[seq_num][:seq_len - 1]

            # 打印特征的尺寸大小
            print(f"蛋白质特征{seq_num + 1}protrans的尺寸大小:", seq_emd.shape)

            # 打印部分内容
            # print("蛋白质特征的部分内容:")
            # print(seq_emd[:5])  # 假设您只想查看前5行的内容

            np.save(os.path.join(output_path, batch_ID_list[seq_num]), seq_emd)

    endtime = datetime.datetime.now()
    print('endtime')
    print(endtime)


def get_dssp(fasta_file, pdb_path, dssp_path):
    DSSP = '/home/mijia/app/DSSP/dssp'

    def process_dssp(dssp_file):
        aa_type = "ACDEFGHIKLMNPQRSTVWY"
        SS_type = "HBEGITSC"
        rASA_std = [115, 135, 150, 190, 210, 75, 195, 175, 200, 170,
                    185, 160, 145, 180, 225, 115, 140, 155, 255, 230]

        with open(dssp_file, "r") as f:
            lines = f.readlines()

        seq = ""
        dssp_feature = []

        p = 0
        while lines[p].strip()[0] != "#":
            p += 1
        for i in range(p + 1, len(lines)):
            aa = lines[i][13]
            if aa == "!" or aa == "*":
                continue
            seq += aa
            SS = lines[i][16]
            if SS == " ":
                SS = "C"
            SS_vec = np.zeros(9)  # The last dim represents "Unknown" for missing residues
            SS_vec[SS_type.find(SS)] = 1
            PHI = float(lines[i][103:109].strip())
            PSI = float(lines[i][109:115].strip())
            ACC = float(lines[i][34:38].strip())
            ASA = min(100, round(ACC / rASA_std[aa_type.find(aa)] * 100)) / 100
            dssp_feature.append(np.concatenate((np.array([PHI, PSI, ASA]), SS_vec)))

        return seq, dssp_feature

    def match_dssp(seq, dssp, ref_seq):
        alignments = pairwise2.align.globalxx(ref_seq, seq)
        ref_seq = alignments[0].seqA
        seq = alignments[0].seqB

        SS_vec = np.zeros(9)  # The last dim represent "Unknown" for missing residues
        SS_vec[-1] = 1
        padded_item = np.concatenate((np.array([360, 360, 0]), SS_vec))

        new_dssp = []
        for aa in seq:
            if aa == "-":
                new_dssp.append(padded_item)
            else:
                new_dssp.append(dssp.pop(0))

        matched_dssp = []
        for i in range(len(ref_seq)):
            if ref_seq[i] == "-":
                continue
            matched_dssp.append(new_dssp[i])

        return matched_dssp

    def transform_dssp(dssp_feature):
        dssp_feature = np.array(dssp_feature)
        angle = dssp_feature[:, 0:2]
        ASA_SS = dssp_feature[:, 2:]

        radian = angle * (np.pi / 180)
        dssp_feature = np.concatenate([np.sin(radian), np.cos(radian), ASA_SS], axis=1)

        return dssp_feature

    def get_dssp(data_path, dssp_path, ID, ref_seq):
        try:
            os.system("{} -i {}.pdb -o {}.dssp".format(DSSP, data_path + ID, dssp_path + ID))

            # dssp_seq, dssp_matrix = process_dssp(dssp_path + ID + ".dssp")
            # if dssp_seq != ref_seq:
            #     dssp_matrix = match_dssp(dssp_seq, dssp_matrix, ref_seq)
            # np.save(dssp_path + ID + "_dssp.npy", transform_dssp(dssp_matrix))
            # print(ID,"    dssp特征：")
            #
            # print(transform_dssp(dssp_matrix).shape)
            # print(len(ref_seq))
            #
            # os.system('rm {}.dssp'.format(dssp_path + ID))
            # return 0
        except Exception as e:
            print(e)
            return None

    pdbfasta = {}
    with open(fasta_file) as r1:
        fasta_ori = r1.readlines()
    for i in range(len(fasta_ori)):
        if fasta_ori[i][0] == ">":
            name = fasta_ori[i].split('>')[1].replace('\n', '')
            seq = fasta_ori[i + 1].replace('\n', '')
            pdbfasta[name] = seq

    fault_name = []
    for name in pdbfasta.keys():
        sign = get_dssp(pdb_path, dssp_path, name, pdbfasta[name])
        if sign == None:
            fault_name.append(name)
    # if fault_name != []:
    #     np.save('../Example/structure_data/dssp_fault.npy', fault_name)



import argparse

def main():
    parser = argparse.ArgumentParser(description="Generate features from protein sequences.")

    parser.add_argument("--fasta_file", type=str, default='/home/mijia/EGPDI/data/DNA-129_Test_Extracted.fasta')
    #parser.add_argument("--fasta_file", type=str, default='/home/mijia/EGPDI/data/3cum_A.fasta')
    parser.add_argument("--prottrans_output_path", type=str, default='/home/mijia/EGPDI/data/prottrans/')
    parser.add_argument('--pdb_dir', type=str, default='/home/mijia/EGPDI/data/pdb_dir/')
    parser.add_argument("--dssp_output_path", type=str, default='/home/mijia/EGPDI/data/SS/')

    args = parser.parse_args()

    # 调用之前定义的函数生成特征
    #get_prottrans(args.fasta_file, args.prottrans_output_path)

    get_dssp(args.fasta_file, args.pdb_dir, args.dssp_output_path)


if __name__ == "__main__":
    main()