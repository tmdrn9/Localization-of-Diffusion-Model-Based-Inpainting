from torch.utils.data import Dataset
import cv2
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, csv, transforms=None):
        self.csv = csv.reset_index(drop=True)
        self.transforms = transforms

    def __getitem__(self, index):
        row = self.csv.iloc[index]

        image = cv2.imread(row.imagepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(row.maskpath, cv2.IMREAD_GRAYSCALE)
        mask = np.where(mask>=255,0,255)

        if self.transforms is not None:
            augmented = self.transforms(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        mask = np.expand_dims(mask, axis=2)
        image = image.astype(np.float32).transpose(2, 0, 1)
        mask = mask.astype(np.float32).transpose(2, 0, 1)

        image /= 255
        mask /= 255

        return image, mask

    def __len__(self):
        return self.csv.shape[0]


  class EvalDataset_jpeg(Dataset): #jpeg 강인성 실험 때 사용
    def __init__(self, qf, csv, transforms=None):
        self.csv = csv.reset_index(drop=True)
        self.transforms = transforms
        self.qualityFactor = qf

    def __getitem__(self, index):
        row = self.csv.iloc[index]
        image_name = row.imagepath.split('/')[8]

        image = cv2.imread(row.imagepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.qualityFactor]
        _, encimg = cv2.imencode('.jpg', image, encode_param)
        image = cv2.imdecode(encimg, 1)

        mask = cv2.imread(row.maskpath, cv2.IMREAD_GRAYSCALE)
        mask = np.where(mask>=255,0,255)

        if self.transforms is not None:
            augmented = self.transforms(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        mask = np.expand_dims(mask, axis=2)
        image = image.astype(np.float32).transpose(2, 0, 1)
        mask = mask.astype(np.float32).transpose(2, 0, 1)

        image /= 255
        mask /= 255

        return image, mask, image_name

    def __len__(self):
        return self.csv.shape[0]
