from torch.utils.data import Dataset

max_lenght = 512


# Preprocessing input for tokenizer (i.e. from list to string)
def parse(list_string):
    parsed = ""
    for s in list_string:
        parsed = parsed + s + " "
    parsed = parsed + "\n"
    return parsed


class trialDataset(Dataset):
    def __init__(self, df):
        self.data = df

        # idxs = list(range(len(self.data)))
        print("Loaded {} rows of data".format(len(self.data)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]