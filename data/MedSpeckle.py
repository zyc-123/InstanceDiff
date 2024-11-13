import torch.utils.data as data
import torch
import numpy as np
import json
from torchvision.transforms import Resize, Normalize
from torchvision.transforms import Compose

def clip_transform(img, resolution=224):
    return Compose([
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])(img)

class SpeckleMedDataset(data.Dataset):
    def __init__(self, data_flist, phase="train", max_dataset_size=1000000, opt=None,
                 use_artifact_type=[]) -> None:
        self.use_artifact_type = use_artifact_type
        self.opt = opt
        self.df = []
        with open(data_flist, 'r') as f:
            self.df += json.load(f)[phase]
    
        new_df = []
        for i in range(len(self.df)):
            item = self.df[i]
            if item['name'] in use_artifact_type:# and item["region"] in ["Fetus"]: # and item['region'] in ["Fetus", "Kidney", "FASCICLE", "Thyroid"]:
                new_df.append(item)
# ["Brain", "Pancreas"]
        self.df = new_df

        if max_dataset_size<len(self.df):
            self.df = self.df[:max_dataset_size]

        self.resizer = Resize((224, 224), antialias=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index, resolution=224):
        item = self.df[index]
        A_path = item['A']
        B_path = item['B']
        A_emb_path = item['A_emb']
        name = item['name']

        a_img = np.fromfile(A_path, dtype=np.float32).reshape(1, 224, 224)
        b_img = np.fromfile(B_path, dtype=np.float32).reshape(1, 224, 224)
        A_emb = np.fromfile(A_emb_path, dtype=np.float32).reshape(1, -1)

        # a_img = np.clip(a_img, 0, 1)
        # b_img = np.clip(b_img, 0, 1)

        A = torch.from_numpy(a_img)
        B = torch.from_numpy(b_img)
        A_emb = torch.from_numpy(A_emb)

        if name=='scatter artifact in CT':
            A[A<0] = 0
            A[A>1800] = 1800
            B[B<0] = 0
            B[B>1800] = 1800
            A = A / 1800.0
            B = B / 1800.0

        if name=='noise in cryo-EM image':
            A = np.clip(A, 0.0, 255.0)
            B = np.clip(B, 0.0, 255.0)
            A = A / 255.0
            B = B / 255.0

        A = A * 2.0 -1.0
        B = B * 2.0 -1.0

        # lq4clip = A # clip_transform(A)
        return {'LQ': A, 'GT': B, 'LQ_path': A_path, 'GT_path': B_path, 'name': name, 'A_emb': A_emb}#, "region": item["region"]}
    
    
def create_SpeckleMedDataset(params=None):
    dataset_file = params['dataset_file']
    import platform
    if platform.system()=="Windows":
        dataset_file = params['dataset_file_win']
    phase = params['name'].split('_')[0]
    max_dataset_size = params['max_dataset_size']
    use_artifact_type = params['use_artifact_type']
    dataset = SpeckleMedDataset(dataset_file,
                                phase=phase,
                                max_dataset_size=max_dataset_size,
                                opt=params,
                                use_artifact_type=use_artifact_type)
    return dataset
