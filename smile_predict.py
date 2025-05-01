import pickle
import sys

import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from transformers import pipeline
from transformers import BertTokenizer, AutoTokenizer,BertModel,RobertaTokenizer,RobertaModel
from model_create import Net
import argparse
import os

def smiles_to_vector(seq):
    inputs = tokenizer_(seq, return_tensors="pt")
    with torch.no_grad():
        outputs = model_(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze()

def predict_smiles(smiles):
    input_file = [smiles_to_vector(smiles).numpy()]
    input_data = np.stack(input_file)
    with open("./data/data_X.pkl",'rb') as f:
        X = pickle.load(f)
    with open("./data/data_Y.pkl",'rb') as f:
        y = pickle.load(f)
    dz_X_train, dz_X_test, dz_y_train, dz_y_test = train_test_split(X, y, random_state=42)
    scaler = StandardScaler()
    y_train = scaler.fit_transform(dz_y_train)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    fp_tensor = torch.tensor(input_data, device=device).float()
    prediction = predict_model(fp_tensor)
    dz = prediction.item()
    cc = np.array(dz).reshape(1,1)
    a = scaler.inverse_transform(cc)
    return a[0][0]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate the IC50 of drugs (Smile format) based on the esophageal cancer drug model')
    parser.add_argument('--input','-i',
                        required=True,
                        help='Drug file (Smile format)')
    parser.add_argument('--outdir',
                        type=str,
                        default='out',
                        help='The output directory to which all files should be written')
    parser.add_argument('--outfile','-o',
                        type=str,
                        default='result.txt',
                        help='The output file name')
    args = parser.parse_args()

    predict_model = torch.load("./model/ecsc_drug_5.pth")
    predict_model.eval()

    pipe = pipeline("fill-mask", model="DeepChem/ChemBERTa-77M-MLM")
    tokenizer_ = RobertaTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MLM", truncation=True)
    model_ = RobertaModel.from_pretrained("DeepChem/ChemBERTa-77M-MLM")

    input_file = args.input
    out_dir = args.outdir
    out_file = args.outfile

    df = pd.read_csv(input_file)
    logic50=[]
    for i in df.iloc[:,0]:
        temp_value = predict_smiles(i)
        logic50.append(temp_value)

    input_abs_path = os.path.abspath(input_file)
    input_path = os.path.dirname(input_abs_path)
    if out_dir == "out":
        out_dir = os.path.join(input_path, "out")
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
    else:
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

    output_final = os.path.join(out_dir,out_file)

    with open(output_final,'a') as f:
        f.write("Smiles" + "\t" + "pIC50" + "\n")
        for i in range(len(logic50)):
            f.write(df.iloc[i,0] + "\t")
            final_ic50 = str(logic50[i].item())
            f.write(final_ic50 + "\n")

