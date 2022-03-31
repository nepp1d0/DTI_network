from torch.utils.data import Dataset

# Preprocessing input for tokenizer (i.e. from list to string)
def parse(list_string):
    parsed = ""
    for s in list_string:
        parsed = parsed + s + " "
    parsed = parsed + "\n"
    return parsed


class DatasetBase(Dataset):
    def __init__(self, df):
        self.data = df
        print("Loaded {} rows of data".format(len(self.data)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]