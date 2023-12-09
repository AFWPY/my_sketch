from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
class CustomDataset(Dataset):
    def __init__(self, root, transform=None):
        self.image_paths = root
        self.imgs = self.read_file(self.image_paths)
        if transform is None:
            self.transform = transforms.Compose([transforms.Resize((256, 256)),
                                          transforms.ToTensor()])
        else:
            self.transform = transform

        self.dir_A = os.path.join(root, 'train' + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(root, 'train' + 'B')  # create a path '/path/to/data/trainB'

        self.A_paths = self.read_file(self.dir_A)   # load images from '/path/to/data/trainA'
        self.B_paths = self.read_file(self.dir_B)    # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B

    def read_file(self, path):
        """从文件夹中读取数据"""
        files_list = os.listdir(path)
        file_path_list = [os.path.join(path, img) for img in files_list]
        file_path_list.sort()  # 如果你需要特定的顺序则保留这一行
        return file_path_list

    def __len__(self):
        return len(self.imgs)  # 返回图片列表的长度

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        # if self.opt.serial_batches:   # make sure index is within then range
        #     index_B = index % self.B_size
        # else:   # randomize the index for domain B to avoid fixed pairs.
        # index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index % self.B_size]
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        # apply image transformation
        A = self.transform(A_img)
        B = self.transform(B_img)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}