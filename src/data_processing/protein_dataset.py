from torch.utils.data import DataLoader, random_split
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer


# Preprocessing input for tokenizer (i.e. from list to string)
def parse(list_string):
    parsed = ""
    for s in list_string:
        parsed = parsed + s + " "
    parsed = parsed + "\n"
    return parsed


class ProteinDataset(Dataset):
    def __init__(self, df, smiles_tokenizer, protein_tokenizer, smiles_model, protein_model, max_length=512):
        self.data = df
        self.smiles_tokenizer = smiles_tokenizer
        self.protein_tokenizer = protein_tokenizer
        self.smiles_model = smiles_model
        self.protein_model = protein_model
        self.max_length = max_length
        # idxs = list(range(len(self.data)))
        print("Loaded {} rows of data".format(len(self.data)))

    def __len__(self):
        return len(self.data)



    def __getitem__(self, idx):
        sample_ligand = self.data.iloc[idx]['smiles']
        sample_target = self.data.iloc[idx]['target']
        label = self.data.iloc[idx]['interaction']

        # Encode samples
        encoded_smiles = self.smiles_tokenizer.encode_plus(sample_ligand,
                                                           max_length=self.max_length,  # Pad & truncate all sentences.
                                                           padding='max_length',
                                                           truncation=True,
                                                           return_tensors="pt")
        encoded_target = self.protein_tokenizer.encode_plus(sample_target,
                                                            max_length=self.max_length,  # Pad & truncate all sentences.
                                                            padding='max_length',
                                                            truncation=True,
                                                            return_tensors="pt")
        # Extract two embedding
        smiles_embedding = self.__extract_embedding__(encoded_smiles, 'smiles')
        protein_embedding = self.__extract_embedding__(encoded_target, 'protein')

        # Append two embedding
        input = torch.cat((smiles_embedding, protein_embedding), 1)

        return (input, label)