from torch.utils.data import DataLoader, Dataset


class MyDataset(Dataset):
    def __init__(self):
        self.l = [i for i in range(1_000_000_000)]

    def __getitem__(self, index):
        return self.l[index]

    def __len__(self):
        return len(self.l)


loader = DataLoader(MyDataset(), batch_size=200, num_workers=0, shuffle=True)

# for x in loader:
#     time.sleep(0.001)
#     print(x)
