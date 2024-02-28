import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import time

import datetime
import itertools
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from datasets import *
from loss import SSIM, Contrastive, QualityAssessment, MarginalQualityRankingLoss
from models import *
from history import log_history

"""
You can try different random seed for init weight if the performance can not match to the paper.
-----------------------
RS = rain streak + fog
SW = snow
RD = rain drop
---------------------- 
"""
parser = argparse.ArgumentParser()
parser.add_argument("--exp_name", type=str, default="UtilityIR_RSSWRD", help="name of the dataset")
parser.add_argument("--data_root", type=str, default=r'D:\\', help="dataset")

parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=40, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=2, help="size of the batches")
parser.add_argument("--lr", type=float, default=1e-4, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=18, help="epoch from which to start lr decay")

parser.add_argument("--num_workers", type=int, default=2, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_size", type=int, default=256, help="size of image width")
parser.add_argument("--n_downsample", type=int, default=2, help="number downsampling layers in encoder")
parser.add_argument("--n_residual", type=int, default=5, help="number of residual blocks in encoder / decoder")
parser.add_argument("--dim", type=int, default=64, help="number of filters in first encoder layer")
parser.add_argument("--deg_dim", type=int, default=128, help="dimensionality of the degle code")
parser.add_argument('--num_CL', type=int, default=5, help='Number of Cl instance')
parser.add_argument('--test_dir', type=str, default=r'./AllWeather/test/raindrop/input',
                    help='RD_val_path')
parser.add_argument('--test_GT_folder', type=str, default=r'./AllWeather/test/raindrop/gt',
                    help='RD_GT_path')
parser.add_argument("--eval_interval", type=int, default=1, help="interval eval enhanced result")
parser.add_argument("--checkpoint_interval", type=int, default=10, help="interval between saving model checkpoints")

parser.add_argument("--gpu", type=str, default='1', help="set GPU")
parser.add_argument("--seed", type=int, default=123, help="Random state")
opt = parser.parse_args()
cuda = torch.cuda.is_available()
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu


def set_seed(seed, cuda):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True


def worker_init(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def train():
    os.makedirs("./images/%s" % (opt.exp_name), exist_ok=True)
    os.makedirs("./saved_models/%s" % (opt.exp_name), exist_ok=True)

    set_seed(opt.seed, cuda)
    ###################
    #    Loss
    ###################
    l1_loss = nn.L1Loss()
    ssim_loss = SSIM()
    contrastive_loss = Contrastive()
    qa_loss = QualityAssessment()
    ranking_loss = MarginalQualityRankingLoss()
    if cuda:
        l1_loss = l1_loss.cuda()
        ssim_loss = ssim_loss.cuda()
        contrastive_loss = contrastive_loss.cuda()
        qa_loss = qa_loss.cuda()
        ranking_loss = ranking_loss.cuda()

    # Loss Weight

    lambda_enhanced = 10
    lambda_ssim = 5
    lambda_latent = 0.5
    lambda_qa = 0.3

    ###################
    #   Model
    ###################
    utilityIR = UtilityIR(dim=opt.dim, n_downsample=opt.n_downsample, n_residual=opt.n_residual,
                          deg_dim=opt.deg_dim)

    if cuda:
        utilityIR = utilityIR.cuda()

    if opt.epoch != 0:
        utilityIR.load_state_dict(torch.load("saved_models/%s/utilityIR_best.pth" % (opt.exp_name)))

    else:
        # Initialize weights
        utilityIR.apply(weights_init_normal)

    ###################
    #   Optimizer & scheduler
    ###################
    optimizer_G = torch.optim.AdamW(
        itertools.chain(utilityIR.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2),
        amsgrad=True)
    # Schedulers
    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
        optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
    )
    if opt.epoch != 0:
        for _ in range(opt.epoch-1):
            lr_scheduler_G.step()

    ###################
    #  DataLoader
    ###################

    transforms_train = [
        transforms.ToTensor(),
        # transforms.Normalize(0.5, 0.5),
    ]
    set_seed(opt.seed, cuda)
    dataloader = DataLoader(
        DegradationTrainDataset(n_classes=3,
                                patch_size=opt.img_size,
                                transforms_=transforms_train,
                                num_cl=opt.num_CL, mode='train'),
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        worker_init_fn=worker_init,
        pin_memory=True)
    set_seed(opt.seed, cuda)
    val_dataloader = DataLoader(
        DegradationTestDataset(patch_size=opt.img_size, transforms_=transforms_train, dataset_path=opt.test_dir),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        worker_init_fn=worker_init,
        pin_memory=True,
    )
    # =============================#
    #         History              #
    # =============================#
    best_psnr, best_ssim = 0, 0
    best_record = 0
    best_loss = np.inf
    loss_history = []
    psnr_history = []
    ssim_history = []
    epochs = []

    ###################
    #  Training
    ###################
    def eval_model(epoch):
        from skimage.metrics._structural_similarity import structural_similarity as compare_ssim
        from skimage.metrics.simple_metrics import peak_signal_noise_ratio as compare_psnr
        out_path = "images/%s" % opt.exp_name
        os.makedirs(out_path, exist_ok=True)
        for i, batch in enumerate(val_dataloader):
            img = batch["img"]
            name = batch["name"][0].split(os.sep)[-1]
            with torch.no_grad():
                # Create copies of image
                img = Variable(img.type(Tensor)).cuda()
                sys.stdout.write("\r Processing: %d/%d" % (i, len(val_dataloader)))

                img_en = utilityIR(img, training=False)

                # enhanced_Real = (enhanced_Real + 1) / 2
                ndarr = img_en.squeeze().mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu',
                                                                                               torch.uint8).numpy()
                im = Image.fromarray(ndarr)

                ori_im = Image.open(batch["name"][0])
                im = im.resize(ori_im.size)

                im.save(os.path.join(out_path, name))

        # objective
        test_est_list = [x for x in sorted(os.listdir(out_path)) if is_image_file(x)]
        PSNR = 0
        SSIM = 0
        for i in range(test_est_list.__len__()):
            sys.stdout.write("\r Processing: %d/%d" % (i, test_est_list.__len__()))
            x = test_est_list[i]
            est = cv2.imread(os.path.join(out_path, x))

            x = x.replace('rain', 'clean')  # RD dataset
            gt = cv2.imread(os.path.join(opt.test_GT_folder, x))

            psnr_val = compare_psnr(gt, est, data_range=255)
            ssim_val = compare_ssim(gt, est, multichannel=True)

            PSNR += psnr_val
            SSIM += ssim_val
            # print(psnr_val, ssim_val)
        PSNR /= test_est_list.__len__()
        SSIM /= test_est_list.__len__()
        print("epoch:%d => PSNR: %.3f, SSIM: %.3f" % (epoch, PSNR, SSIM))
        return PSNR, SSIM

    # Adversarial ground truths
    valid = 1
    fake = 0
    prev_time = time.time()
    eval_model(0)
    for epoch in range(opt.epoch + 1 if opt.epoch > 0 else 0, opt.n_epochs + 1):
        epoch_loss = 0
        for i, batch in enumerate(dataloader):
            optimizer_G.zero_grad()

            # Set model input
            img_in_A = Variable(batch["Img_In"][0].type(Tensor))
            img_in_B = Variable(batch["Img_In"][1].type(Tensor))
            img_gt_A = Variable(batch["Img_GT"][0].type(Tensor))
            img_gt_B = Variable(batch["Img_GT"][1].type(Tensor))

            if cuda:
                img_in_A = img_in_A.cuda()
                img_gt_A = img_gt_A.cuda()

                img_in_B = img_in_B.cuda()
                img_gt_B = img_gt_B.cuda()

            gt_qa_a, gt_qa_b = qa_loss(img_in_A, img_gt_A), qa_loss(img_in_B, img_gt_B)
            deg_cls_A, qa_A, deg_qual_A, img_en_A = utilityIR(img_in_A)
            deg_cls_B, qa_B, deg_qual_B, img_en_B = utilityIR(img_in_B)

            # -----------------------
            #  Train Generator
            # -----------------------
            optimizer_G.zero_grad()

            # loss_ID = lambda_id * (l1_loss(img_in_A, img_recon_A) + l1_loss(img_in_B, img_recon_B))
            loss_enhanced = lambda_enhanced * (l1_loss(img_en_A, img_gt_A) + l1_loss(img_en_B, img_gt_B))  #
            loss_ssim = lambda_ssim * (1 - ssim_loss(img_en_A, img_gt_A) + 1 - ssim_loss(img_en_B, img_gt_B))  #
            loss_qa = lambda_qa * (ranking_loss(qa_A, qa_B, gt_qa_a, gt_qa_b))  # 1

            deg_A_neg = []
            deg_B_neg = []
            deg_A_pos = utilityIR.die(batch["Img_CL"][0][0].cuda(), psnr_weight=50)[0]
            deg_B_pos =  utilityIR.die(batch["Img_CL"][1][0].cuda(), psnr_weight=50)[0]
            for idx in range(1, opt.num_CL):
                deg_A_neg.append( utilityIR.die(batch["Img_CL"][0][idx].cuda(), psnr_weight=50)[0])
                deg_B_neg.append( utilityIR.die(batch["Img_CL"][1][idx].cuda(), psnr_weight=50)[0])
            loss_latent_s = lambda_latent * (contrastive_loss(deg_cls_B, deg_B_pos, deg_B_neg, type='deg_r')
                                             + contrastive_loss(deg_cls_A,
                                                                deg_A_pos,
                                                                deg_A_neg,
                                                                type='deg_r'))  # + c
            loss_latent_e_i = lambda_latent * (
                    contrastive_loss(img_en_A, img_gt_A, img_in_A, type='img_per') + contrastive_loss(img_en_B,
                                                                                                      img_gt_B,
                                                                                                      img_in_B,
                                                                                                      type='img_per'))  #
            loss_latent = loss_latent_s + 0.01 * loss_latent_e_i
            # Total Loss
            loss_G = (loss_enhanced + loss_ssim + loss_latent + loss_qa)
            loss_G.backward()
            optimizer_G.step()

            # --------------
            #  Log Progress
            # --------------

            # Determine approximate time left
            batches_done = epoch * len(dataloader) + i
            batches_left = opt.n_epochs * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            if i % 500 == 0:
                E = loss_enhanced.item() + loss_ssim.item()
                print(
                    "[Epoch %d/%d][Batch %d/%d] [LR: %f]"
                    " [G loss: %f -- {CL: %f, Edge: %f, QA: %f} ]"
                    " [Enhanced loss %f: [L1: %f, ssim: %f]] ETA: %s"
                    % (
                        epoch, opt.n_epochs, i, len(dataloader), optimizer_G.param_groups[0]['lr'],
                        loss_G.item(), loss_latent.item(), 0.0, loss_qa.item(),
                        E, loss_enhanced.item(), loss_ssim.item(),
                        time_left),

                )
            epoch_loss += loss_G.item()

        epoch_loss /= len(dataloader)
        loss_history.append(epoch_loss)
        epochs.append(epoch)
        # eval_image per each epoch
        if epoch % opt.eval_interval == 0:
            eval_psnr, eval_ssim = eval_model(epoch)
            psnr_history.append(eval_psnr)
            ssim_history.append(eval_ssim)
            save_best = False
            if best_psnr < eval_psnr:
                best_psnr = eval_psnr
                print("Epoch %d ==> Best PSNR save: %f " % (epoch, best_psnr))
                save_best = True
            if best_ssim < eval_ssim:
                best_ssim = eval_ssim
                print("Epoch %d ==> Best SSIM save: %f " % (epoch, best_ssim))
                save_best = True
            if save_best:
                print("Epoch %d ==> Best Record save: %f, %f " % (epoch, best_psnr, best_ssim))
                # Save model checkpoints
                torch.save(utilityIR.state_dict(), "saved_models/%s/utilityIR_best.pth" % (opt.exp_name))
                # print("Save model for best record: %d" % epoch)

        if epoch % opt.checkpoint_interval == 0 and epoch > opt.n_epochs // 3:
            # Save model checkpoints
            torch.save(utilityIR.state_dict(), "saved_models/%s/utilityIR_%d.pth" % (opt.exp_name, epoch))

        # Update learning rates
        lr_scheduler_G.step()

    log_history(loss_history=loss_history, metric_historys=[psnr_history, ssim_history], epochs=epochs, opt=opt)
    print("Best PSNR: %f. Best SSIM: %f" % (best_psnr, best_ssim))


if __name__ == "__main__":
    train()
