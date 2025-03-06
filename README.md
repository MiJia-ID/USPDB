# USPDB: A Novel U-Shaped Equivariant Graph Neural Network with Subgraph Sampling for Protein-DNA Binding Site Prediction


## Abstract
Protein-DNA binding directly influences the normal functioning of biological processes by regulating gene expression. Accurate identification of binding sites can reveal the mechanisms of protein-DNA interactions and provide a clear direction for drug target development. However, traditional experimental methods are time-consuming and costly, necessitating the development of efficient computational methods. Although existing computational methods have made significant progress in the field of protein binding site prediction, they have difficulty extracting key residue features and atomic-level features. To address this, we propose a novel method, USPDB, based on a U-shaped Equivariant Graph Neural Network(U-EGNNet) and Subgraph Sampling for Protein-DNA Binding Site Prediction. USPDB reformulates the binding site prediction task by converting the protein into a graph and performing a binary classification for each residue. It leverages protein large language models, such as Protrans, ESM2, and ESM3, to extract sequence and structural features. The General Equivariant Transformer (GET) module is employed to capture geometric features of residues and atoms. Additionally, the U-EGNNet, composed of EGNN and Subgraph Sampling, is utilized to preserve more global information while sampling subgraphs that contain key residues for further computation. Experimental results on DNA\_test\_181 and DNA\_test\_129 datasets demonstrate that USPDB achieves prediction accuracies of 0.532 and 0.361, respectively, outperforming all baseline methods. Through interpretability analysis, we observed that USPDB effectively focuses on residues within DNA-binding domains without requiring prior knowledge, thereby enhancing the performance of DNA-binding protein prediction.

<div align=center>
<img src="USPDB.jpg" width=75%>
</div>


## Preparation
### Environment Setup
```python 
   git clone https://github.com/MiJia-ID/USPDB.git
   conda env create -f uspdb_environment.yml
```
You also need to install the relative packages to run ESM2, ProtTrans and ESM3 protein language model. 

## Experimental Procedure
### Create Dataset
**Firstly**, you need to use ESM3 to obtain the PDB files of proteins in the tran and test datasets(DNA_train_573, DNA_129_Test and DNA_181_Test ). More details can be found here:https://github.com/evolutionaryscale/esm

Then, run the script below to create node features (PSSM, SS, AF, One-hot encoding). The file is located in the scripts folder.
```python 
python3 data_io.py 
```

**Secondly** , run the script below to create node features(ESM2 embeddings and ProtTrans embeddings). The file can be found in feature folder.</br>

```python 
python3 ESM2_5120.py 
```
```python 
python3 ProtTrans.py 
```
We choose the esm2_t48_15B_UR50D() pre-trained model of ESM-2 which has the most parameters. More details about it can be found at: https://huggingface.co/facebook/esm2_t48_15B_UR50D   </br>
We also choose the prot_t5_xl_uniref50 pre-trained model of ProtTrans, which uses a masked language modeling(MLM). More details about it can be found at: https://huggingface.co/Rostlab/prot_t5_xl_uniref50    </br>

**Thirdly**, run the script below to create edge features. The file can be found in feature folder.
```python 
python3 create_edge.py 
```

### Model Training
Run the following script to train the model.
```python
python3 train_val_bestAUPR_predicted.py 
```
**We also provide pre-trained models at** https://drive.google.com/drive/my-drive  </br>

### Inference on Pretrained Model
Run the following script to test the model. Both test datasets, DNA_129_Test and DNA_181_Test , were included in the testing of the model.
```python
python3 test_129_final.py 
```
```python
python3 test_181_final.py 
```

