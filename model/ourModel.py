import os
from torch.optim import AdamW
from common.size_adapter import ImgAndEventSizeAdapter
from model.ourNet import *
from model.loss import *
from common.laplacian import *
import lpips

from util.utils_func import pyramid_Img

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Model:
    def __init__(self, args):
        self.my_net = MyNet(args)
        self.args = args
        self.device()
        self.device_num = torch.cuda.device_count()
        self.is_multiple()
        self.optimG = AdamW(self.my_net.parameters(), lr=1e-6, weight_decay=1e-3)
        self.lap = LapLoss()
        self.perc = lpips.LPIPS(net='alex').to(device)
        self.patch_ssim = SSIM_with_patch_mask(args.mask_patch_size)
        self.size_adapter = ImgAndEventSizeAdapter()

    def is_multiple(self):
        if self.device_num > 1:
            print("Start with", self.device_num, "GPUs!")
            self.my_net = nn.DataParallel(self.my_net, device_ids=list(range(self.device_num)))

    def train(self):
        self.my_net.train()

    def eval(self):
        self.my_net.eval()

    def device(self):
        self.my_net.to(device)

    def load_checkpoint(self, path):
        def convert(param):
            return {
                k.replace("module.", ""): v
                for k, v in param.items()
                if "module." in k
            }

        checkpoint = torch.load(path)
        if self.device_num == 1:
            checkpoint['net'] = convert(checkpoint['net'])
        self.my_net.load_state_dict(checkpoint['net'])

    def save_model(self, path):
        torch.save(self.my_net.state_dict(), '{}/work.pkl'.format(path))

    def save_model_min_loss(self, path):
        torch.save(self.my_net.state_dict(), '{}/min_loss.pkl'.format(path))

    def save_checkpoint(self, type, epoch):
        checkpoint = {
            "net": self.my_net.state_dict(),
            'optimizer': self.optimG.state_dict(),
            "epoch": epoch
        }
        if not os.path.isdir("./train/checkpoint"):
            os.mkdir("./train/checkpoint")
        torch.save(checkpoint, './train/checkpoint/ckpt_%s.pth' % (str(epoch)))

    def inference(self, imgs, voxels, mask):
        imgs, voxels, mask = self.size_adapter.pad(imgs, voxels, mask)
        self.eval()
        _, _, _, pred, _, _ = self.my_net(imgs, voxels, mask, self.args.bins)
        pred = self.size_adapter.unpad(pred[0])
        return pred