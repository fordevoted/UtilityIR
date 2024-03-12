import argparse
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import torchvision

from loss import QualityAssessment, SSIM, MarginalQualityRankingLoss
import sys

import time

import cv2
import numpy as np
from torch.autograd import Variable
from torchvision.utils import save_image
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from models import *
from datasets import *

import torch
from tqdm import tqdm

RSDGT = [r'D:\20230119test\rainstreak/gt']
RDDGT = [r'D:\AllWeather\test\raindrop\gt']
SWDGT = [r'D:\20230119test/snow/gt']

RSD = [r'D:\20230119test\rainstreak/input']
RDD = [r'D:\AllWeather\test\raindrop\input']
SWD = [r'D:\20230119test/snow/input']

test_name = 'RDD[0]'


def findfolder(test_name, isTest=False):
    index = int(test_name[4])
    if 'RD' == test_name[:2]:
        if isTest:
            return os.path.join(opt.out_dir, opt.exp_name, 'test_image', test_name), RDDGT[
                index]
        else:
            return RDD[index]
    elif 'RS' == test_name[:2]:
        if isTest:
            return os.path.join(opt.out_dir, opt.exp_name, 'test_image', test_name), RSDGT[
                index]
        else:
            return RSD[index]
    else:
        if isTest:
            return os.path.join(opt.out_dir, opt.exp_name, 'test_image', test_name), SWDGT[
                index]
        else:
            return SWD[index]


parser = argparse.ArgumentParser()
parser.add_argument("--exp_name", type=str, default="UtilityIR_RSSWRD",
                    # parser.add_argument("--exp_name", type=str, default="release",
                    help="name of the experiment")
parser.add_argument("--test_dir", type=str, default=findfolder(test_name), help="path to test image directory")
parser.add_argument("--out_dir", type=str, default=r'./output', help="path to output image directory")
parser.add_argument("--testing", type=str, default=r'test_image', help="testing function. All list: ['test_image', 'test_plot_latent_tsne', 'test_latent_manipulation', 'eval_objective', 'test_QA']")
parser.add_argument("--checkpoint", type=int, default=30, help="number of epoch of checkpoint")

parser.add_argument("--n_downsample", type=int, default=2, help="number downsampling layers in encoder")
parser.add_argument("--n_residual", type=int, default=5, help="number of residual blocks in encoder / decoder")
parser.add_argument("--dim", type=int, default=64, help="number of filters in first encoder layer")
parser.add_argument("--style_dim", type=int, default=128, help="dimensionality of the style code")

parser.add_argument("--gpu", type=str, default='1', help="set GPU")
parser.add_argument("--print_model_complexity", type=bool, default=True,
                    help="Print number of params and run time speed")
parser.add_argument("--seed", type=int, default=42, help="Random state")
opt = parser.parse_args()
# print(opt)
cuda = torch.cuda.is_available()
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)
    cudnn.benchmark = True

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor


def worker_init(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def test_image(epoch):
    out_path = os.path.join(opt.out_dir, opt.exp_name, 'test_image', test_name)
    # out_cls_path = os.path.join(opt.out_dir, opt.exp_name, 'test_REAL_cls_image', str(epoch), test_name)
    os.makedirs(out_path, exist_ok=True)
    # os.makedirs(out_cls_path, exist_ok=True)

    transforms_test = [
        transforms.ToTensor(),
        # transforms.Normalize(0.5, 0.5),
    ]

    val_dataloader = DataLoader(
        DegradationTestDataset(transforms_=transforms_test, dataset_path=opt.test_dir),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        worker_init_fn=worker_init,
        pin_memory=True,
    )

    if opt.print_model_complexity:
        num_params = 0
        models = [utilityIR]
        for model in models:
            for param in model.parameters():
                num_params += param.numel()
            # print(net)
        print('Total number of parameters: %d, %s' % (num_params, opt.exp_name))
    # exit(0)
    time_all = 0
    for i, batch in enumerate(tqdm(val_dataloader)):
        img = batch["img"]
        name = batch["name"][0].split(os.sep)[-1]

        with torch.no_grad():
            # Create copies of image
            img = Variable(img.type(Tensor))
            if cuda:
                img = img.cuda()

            # Generate samples
            start = time.time()

            img_restored = utilityIR(img, training=False)
            if opt.print_model_complexity and i != 0:
                time_all += time.time() - start
                # print("time: ", time.time() - start)

            ndarr = img_restored.squeeze().mul(255).add_(0.0).clamp_(0, 255).permute(1, 2, 0).to('cpu',
                                                                                                  torch.uint8).numpy()
            im = Image.fromarray(ndarr)
            ori_im = Image.open(batch["name"][0])
            im = im.resize(ori_im.size)

            im.save(os.path.join(out_path, name))


    print("Total time: %f, average time: %f, FPS: %f, dataloader: %d" % (
        time_all, time_all / (len(val_dataloader) - 2), (len(val_dataloader) - 2) / time_all, len(val_dataloader)))


def test_plot_latent_tsne():
    transforms_test = [
        transforms.ToTensor(),
        # transforms.Normalize(0.5, 0.5),
    ]

    dataloader = DataLoader(
        DegradationTrainDataset(n_classes=3,
                                patch_size=opt.img_size,
                                transforms_=transforms_test, mode='train'),
        batch_size=1,
        shuffle=True,
        num_workers=0,
        worker_init_fn=worker_init,
        pin_memory=True)

    feature = []
    label = []
    bound = 500
    for i, batch in enumerate(dataloader):
        img_in_A = Variable(batch["Img_In"][0].type(Tensor))
        img_in_B = Variable(batch["Img_In"][1].type(Tensor))

        label_A = Variable(batch["label"][:, 0].type(Tensor)).cpu()
        label_B = Variable(batch["label"][:, 1].type(Tensor)).cpu()
        sys.stdout.write("\r Processing: %d/%d" % (i, len(dataloader)))
        # Create copies of image
        with torch.no_grad():
            # Create copies of image
            for img_in, label_in in zip([img_in_A, img_in_B], [label_A, label_B]):
                X_in = img_in.cuda()

                # Generate samples
                deg_type, _, deg_severity = utilityIR.die(X_in, psnr_weight=50)
                label_in = (label_in.squeeze() == 1).nonzero(as_tuple=True)[0].numpy()[0]
                # Generate samples
                feature.append(deg_type.cpu().detach().numpy().flatten().squeeze())

                if label_in == 0:  # RD
                    label.append('type 0: RD')
                elif label_in == 1:  # RS
                    label.append('type 1: RS')
                elif label_in == 2:  # SW
                    label.append('type 2: SW')

                # feature.append(en_s_code_in.cpu().detach().numpy().squeeze())
                # label.append('clean')
                # label.append('real-world → clean')
        if i >= bound:
            break

    feature = np.array(feature)
    label = np.array(label)
    X = TSNE(perplexity=30.0, random_state=123, verbose=1).fit_transform(feature)
    plt.style.use("Solarize_Light2")
    sns.scatterplot(X[:, 0], X[:, 1], hue=label, legend='full', palette='hls')
    # plt.axis("off")
    plt.show()
    plt.savefig("tsne.png")


def test_QA(bound=500):
    transforms_test = [
        transforms.ToTensor(),
        # transforms.Normalize(0.5, 0.5),
    ]

    dataloader = DataLoader(
        DegradationTrainDataset(n_classes=3,
                                patch_size=opt.img_size,
                                transforms_=transforms_test,
                                num_cl=5, mode='train'),
        batch_size=1,
        shuffle=True,
        num_workers=1,
        worker_init_fn=worker_init,
        pin_memory=True)

    count = 0
    correct = 0
    qa_loss = QualityAssessment().cuda()
    ranking_loss = MarginalQualityRankingLoss().cuda()
    for i, batch in tqdm(enumerate(dataloader)):
        img_in_A = Variable(batch["Img_In"][0].type(Tensor))
        img_in_B = Variable(batch["Img_In"][1].type(Tensor))

        img_gt_A = Variable(batch["Img_GT"][0].type(Tensor))
        img_gt_B = Variable(batch["Img_GT"][1].type(Tensor))

        with torch.no_grad():
            # Create copies of image
            gt_qa_a, gt_qa_b = qa_loss(img_in_A, img_gt_A), qa_loss(img_in_B, img_gt_B)
            _, qa_A, __ = utilityIR.die(img_in_A, gt_qa_a, psnr_weight=50)
            _, qa_B, __ = utilityIR.die(img_in_B, gt_qa_b, psnr_weight=50)
            # Generate samples
            rkloss = ranking_loss(qa_A, qa_B, gt_qa_a, gt_qa_b)
            count += 1
            if rkloss == 0:
                correct += 1
        if i >= bound:
            break

    # print("Accuracy: %.2f" % (correct/count)*100)
    return (correct / count) * 100


def test_latent_manipulation():
    out_path = os.path.join(opt.out_dir, opt.exp_name, 'test_latent_manipulation')
    os.makedirs(out_path, exist_ok=True)
    # You can manually adjust alphas value to obtain satisfactory result
    # alphas = np.linspace(-1, 2, num=10)
    # alphas = np.array([0, 1.0, 1.25, 1.8, 2.5, 5.0])
    alphas = np.array([5.0, 1.5, 1.1, 0.9, 0.6, 0.0])
    # alphas = np.array([0, 0.15, 0.25, 0.35, 0.4, 0.45, 1.0])
    # alphas = np.array([1.0, 0.45, 0.4, 0.35, 0.25, 0.15, 0])
    # alphas = np.array([0, 0.25, 0.6, 0.8, 0.9, 1.0, 1.2, 1.33, 1.35, 1.40])
    # alphas = np.array([0, 0.25, 0.45, 0.65, 0.75, 1.0, 1.3, 1.35, 1.45])

    transforms_val = [
        transforms.ToTensor(),
    ]
    transforms_val = transforms.Compose(transforms_val)

    img_list = os.listdir(opt.test_dir)
    for img_name in img_list:
        # if not img_name.__contains__('im_0339_s85_a04'):
        #     continue
        img_path = os.path.join(opt.test_dir, img_name)
        img = load_img(img_path)
        w, h = img.size
        new_w, new_h = w, h
        if (w / 4) % 1 != 0:
            new_w = w // 4 * 4
        if (h / 4) % 1 != 0:
            new_h = h // 4 * 4
        if new_w != w or new_h != h:
            img = img.resize((new_w, new_h))

        img = transforms_val(img).unsqueeze(0).cuda()
        item_list = []
        img_enhanceds = None
        with torch.no_grad():
            # Create copies of image
            img = Variable(img.type(Tensor))
            if cuda:
                img = img.cuda()

            # Generate samples
            start = time.time()

            # gt = transforms.Compose(transforms_test)(load_img(os.path.join(RDDGT[0], name.replace("rain", "clean"))).resize((XReal.shape[3], XReal.shape[2]))).cuda()
            # qa_gt = qa_loss(XReal, gt)
            deg_type, _, deg_severity, img_en = utilityIR(img)
            _, __, deg_severity_iter2 = utilityIR.die(img_en, psnr_weight=50)

            for alpha in alphas:
                s_code = deg_severity + alpha * (deg_severity_iter2 - deg_severity)
                img_en = utilityIR(img, deg_type, s_code, training=False)

                item_list.append(img_en)

            img = None
            for item in item_list:
                img = item if img is None else torch.cat((img, item), -1)

            img_enhanceds = img if img_enhanceds is None else torch.cat((img_enhanceds, img), -2)
            save_image(img_enhanceds, os.path.join(out_path, img_name), nrow=1, normalize=True, range=(0, 1))
            print(img_name)


def eval_objective(test_folder, test_GT_folder, test_name):
    def align_to_four(img):
        # print ('before alignment, row = %d, col = %d'%(img.shape[0], img.shape[1]))
        # align to four
        a_row = int(img.shape[0] / 4) * 4
        a_col = int(img.shape[1] / 4) * 4
        img = img[0:a_row, 0:a_col]
        # print ('after alignment, row = %d, col = %d'%(img.shape[0], img.shape[1]))
        return img

    from skimage.metrics._structural_similarity import structural_similarity as compare_ssim
    from skimage.metrics.simple_metrics import peak_signal_noise_ratio as compare_psnr
    print("input folder:", test_folder, test_GT_folder)
    test_est_list = [x for x in sorted(os.listdir(test_folder)) if is_image_file(x)]
    my = False

    PSNR = 0
    SSIM = 0
    tStart = time.time()
    counter = 0
    for i in range(test_est_list.__len__()):
        sys.stdout.write("\r Processing: %d/%d" % (i, test_est_list.__len__()))
        x = test_est_list[i]
        est = cv2.imread(os.path.join(test_folder, x))

        if 'RD' == test_name[:2]:
            x = x.replace('rain', 'clean')
        gt = cv2.imread(os.path.join(test_GT_folder, x))
        est = cv2.resize(est, (gt.shape[1], gt.shape[0]))
        # uiqm_val, uciqe_val = eval_uw_metrics(est)
        # print(x)
        est = cv2.cvtColor(est, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2YCR_CB)[:, :, 0]

        psnr_val = compare_psnr(gt, est, data_range=255)
        ssim_val = compare_ssim(gt, est, multichannel=True)

        PSNR += psnr_val
        SSIM += ssim_val
        # counter +=1
        # print(psnr_val, ssim_val)
    PSNR /= test_est_list.__len__()
    SSIM /= test_est_list.__len__()
    print("")
    print("test: %.2f sceonds ==>" % (time.time() - tStart), end=" ")
    print("PSNR: %.3f, SSIM: %.3f" % (PSNR, SSIM))


if __name__ == '__main__':

    testing = opt.testing
    testing_list = ['test_image', 'test_plot_latent_tsne', 'test_latent_manipulation', 'eval_objective',
                    'test_QA']

    utilityIR = UtilityIR(dim=opt.dim, n_downsample=opt.n_downsample, n_residual=opt.n_residual,
                          deg_dim=opt.style_dim)

    if cuda:
        utilityIR = utilityIR.cuda()

    if cuda:
        utilityIR.load_state_dict(torch.load("saved_models/%s/utilityIR_best.pth" % (opt.exp_name)))
    else:

        utilityIR.load_state_dict(
            torch.load("saved_models/%s/utilityIR_best.pth" % (opt.exp_name), map_location=torch.device('cpu')))

    if testing.__contains__('test_image'):
        test_image(epoch=opt.checkpoint)
        (img_folder, gt_folder) = findfolder(test_name, isTest=True)
        eval_objective(img_folder, gt_folder, test_name)
    elif testing == 'test_plot_latent_tsne':
        test_plot_latent_tsne()
    elif testing == 'test_latent_manipulation':
        test_latent_manipulation()
    elif testing == 'test_QA':
        record = []
        for i in range(10):
            record.append(test_QA(bound=1000))
        record = np.array(record)
        print("Record: ", record)
        print("Mean, std: %.5f, %.5f" % (record.mean(), record.std()))
    elif testing == 'eval_objective':
        print(test_name)
        (img_folder, gt_folder) = findfolder(test_name, isTest=True)
        eval_objective(img_folder, gt_folder, test_name)
