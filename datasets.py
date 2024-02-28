from torch.utils import data
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from PIL import Image, ImageOps
import torch
import os
import random
import numpy as np
import torchvision.transforms as transforms


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".bmp", ".png", ".jpg", ".jpeg", ".JPG", ".PNG"])


def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    # y, _, _ = img.split()
    return img


def get_patch(imgs, patch_size, ix=-1, iy=-1):
    (ih, iw) = imgs[0].size

    ix = random.randrange(0, iw - patch_size + 1)
    iy = random.randrange(0, ih - patch_size + 1)

    outs = []
    for img in imgs:
        outs.append(img.crop((iy, ix, iy + patch_size, ix + patch_size)))

    return tuple(outs)
def resize_img(imgs):
    outs = []
    for img in imgs:
        w, h = img.size
        new_w, new_h = w, h
        if (w / 4) % 1 != 0:
            new_w = w // 4 * 4
        if (h / 4) % 1 != 0:
            new_h = h // 4 * 4
        if new_w != w or new_h != h:
            img = img.resize((new_w, new_h))
        outs.append(img)
    return tuple(outs)


def augmentation(imgs, flip_h=True, rot=True):
    info_aug = {'flip_h': False, 'flip_v': False, 'trans': False}
    # random.seed(123)
    if random.random() < 0.5 and flip_h:
        for i in range(len(imgs)):
            imgs[i] = ImageOps.flip(imgs[i])
        info_aug['flip_h'] = True
    if rot:
        # random.seed(123)
        if random.random() < 0.5:
            for i in range(len(imgs)):
                imgs[i] = ImageOps.mirror(imgs[i])
            info_aug['flip_v'] = True
        # random.seed(123)
        if random.random() < 0.5:
            for i in range(len(imgs)):
                imgs[i] = imgs[i].rotate(180)
        info_aug['trans'] = True
    return tuple(imgs), info_aug



class DegradationTrainDataset(data.Dataset):
    def __init__(self, n_classes=3, transforms_=None, patch_size=256, num_cl=5, mode='train'):

        data_paths = [r'D:/AllWeather/train/raindrop/input',
                      r'D:/AllWeather/train/rainstreak/input',
                      r'D:/AllWeather/train/snow/input',
                      ]
        gt_path = [r'D:/AllWeather/train/raindrop/gt',
                   r'D:/AllWeather/train/rainstreak/gt',
                   r'D:/AllWeather/train/snow/gt'
                   ]

        """
        data_paths = [r'D:/Haze/OTS_BETA/haze/train',
                      r'D:/Low_Light_Rainy/Rainy/Rain1400_Fu/training/rainy_image',
                      r'D:/AllWeather/train/snow/input',
                      ]
        gt_path = [r'D:/Haze/OTS_BETA/clear/train',
                   r'D:/Low_Light_Rainy/Rainy/Rain1400_Fu/training/ground_truth',
                   r'D:/AllWeather/train/snow/gt'
                   ]
        """

        if mode == 'val':

            data_paths = [r'D:/AllWeather/test/raindrop/input',
                          r'D:/AllWeather/test/rainstreak/input',
                          r'D:/AllWeather/test/snow/input',
                          ]
            gt_path = [r'D:/AllWeather/test/raindrop/gt',
                       r'D:/AllWeather/test/rainstreak/gt',
                       r'D:/AllWeather/test/snow/gt'
                       ]

            """
            data_paths = [r'D:/Haze/OTS_BETA/haze/test',
                          r'D:/Low_Light_Rainy/Rainy/Rain1400_Fu/testing/rainy_image',
                          r'D:/AllWeather/test/snow/input',
                          ]
            gt_path = [r'D:/Haze/OTS_BETA/clear/test',
                       r'D:/Low_Light_Rainy/Rainy/Rain1400_Fu/testing/ground_truth',
                       r'D:/AllWeather/test/snow/gt'
                       ]
            """
        self.mode = mode
        self.n_classes = n_classes
        self.transform = transforms.Compose(transforms_)
        self.patch_size = patch_size
        self.imgInputPath = []
        self.imgGTPath = []
        self.labels = []
        self.len = 0
        self.num_cl = num_cl
        for i in range(data_paths.__len__()):
            path = data_paths[i]
            in_data = []
            gt_data = []
            for x in os.listdir(path):
                if is_image_file(x):
                    in_data.append(os.path.join(path, x))
                    if i == 1 and (mode == 'train' or mode == 'test'):
                        xs = x.split('_')
                        #x = xs[0] + '_' + xs[1] + '.png'
                        x = xs[0] +'.jpg'
                    if i == 0:
                        pass
                        #x = x.replace('rain', 'clean')
                    gt_data.append(os.path.join(gt_path[i], x))
                else:
                    continue

            self.imgInputPath.append(in_data)
            self.len += in_data.__len__()
            self.imgGTPath.append(gt_data)
            label_template = np.zeros(n_classes)
            label_template[i] = 1
            self.labels.append([label_template for i in range(len(in_data))])

    def __getitem__(self, index):
        if self.mode == 'val':

            num = int(random.random() * 3)
            img_in_path = self.imgInputPath[num][index % len(self.imgInputPath[num])]
            img_gt_path = self.imgGTPath[num][index % len(self.imgInputPath[num])]
            label = self.labels[num][index % len(self.imgInputPath[num])]

            img_in = load_img(img_in_path)
            img_gt = load_img(img_gt_path)
            return {"Img_In": self.transform(img_in), "Img_GT": self.transform(img_gt),
                    "label": torch.FloatTensor(label)}
        else:

            num = (index//9000)%self.n_classes
            main_idxs = [index%9000, (index+random.randint(0, 9000))%9000]
            # img_in_path = []
            # img_gt_path = []
            img_in = []
            img_gt = []
            label = []
            cl_imgs = []
            for main_idx in main_idxs:
                img_in_path = self.imgInputPath[num][main_idx % len(self.imgInputPath[num])]
                gt_path = self.imgGTPath[num][main_idx % len(self.imgInputPath[num])]
                img_in.append(load_img(img_in_path))
                img_gt.append(load_img(gt_path))
                label.append(self.labels[num][main_idx % len(self.imgInputPath[num])])

                if self.mode == 'train':
                    (img_in_patch, img_gt_patch) = get_patch(imgs=[img_in[-1], img_gt[-1]],
                                                             patch_size=self.patch_size)
                    (img_in_aug, img_gt_aug), _ = augmentation(imgs=[img_in_patch, img_gt_patch])
                    img_in[-1] = self.transform(img_in_aug)
                    img_gt[-1] = self.transform(img_gt_aug)
                else:
                    img_in[-1] = self.transform(img_in[-1])
                    img_gt[-1] = self.transform(img_gt[-1])
                imgs = []
                for i in range(self.num_cl):
                    if i==0:
                        idx = random.randint(0, len(self.imgInputPath[num]))
                        img_cl = load_img(self.imgInputPath[num][idx % len(self.imgInputPath[num])])
                    else:
                        arr = np.arange(self.n_classes).tolist()
                        del arr[num]
                        cls = random.sample(arr, 1)
                        idx = random.randint(0, len(self.imgInputPath[cls[0]]) -1)
                        img_cl = load_img(self.imgInputPath[cls[0]][idx])

                    (img_cl_patch) = get_patch(imgs=[img_cl], patch_size=self.patch_size)
                    (img_cl_aug), _ = augmentation(imgs=[img_cl_patch[0]])
                    imgs += [self.transform(img_cl_aug[0])]
                cl_imgs.append(imgs)
            return {"Img_In": img_in, "Img_GT": img_gt, "label": torch.FloatTensor(label), "Img_CL": cl_imgs}

    def __len__(self):
        return 9000 * self.n_classes
        #return (self.len // 2) * 2
        # return self.imgInputPath[2].__len__() * 3



class DegradationTestDataset(data.Dataset):
    def __init__(self, transforms_=None, dataset_path="train", patch_size=128, resmaplingRatio=16):
        super(DegradationTestDataset, self).__init__()
        self.transform = transforms.Compose(transforms_)
        self.files = [os.path.join(dataset_path, x) for x in
                      os.listdir(dataset_path) if is_image_file(x)]
        # sorted(glob.glob( + "/*.*"))
        self.patch_size = patch_size
        self.resmaplingRatio = resmaplingRatio

    def __getitem__(self, index):

        img = load_img(self.files[index % len(self.files)])

        w, h = img.size
        new_w, new_h = w, h
        if (w / self.resmaplingRatio) % 1 != 0:
            new_w = w // self.resmaplingRatio * self.resmaplingRatio
        if (h / self.resmaplingRatio) % 1 != 0:
            new_h = h // self.resmaplingRatio * self.resmaplingRatio
        if new_w != w or new_h != h:
            img = img.resize((new_w, new_h))

        img = self.transform(img)

        return {"img": img, 'name': self.files[index % len(self.files)]}

    def __len__(self):
        return 40#len(self.files)
