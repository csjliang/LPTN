import random
from torch.utils import data as data
from torchvision.transforms.functional import normalize

from codes.data.data_util import (paths_from_folder, paths_from_lmdb)
from codes.data.transforms import augment, unpaired_random_crop
from codes.utils import FileClient, imfrombytes, img2tensor

class UnPairedImageDataset(data.Dataset):

    def __init__(self, opt):
        super(UnPairedImageDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths_lq = paths_from_lmdb(self.lq_folder)
            self.paths_gt = paths_from_lmdb(self.gt_folder)

        elif self.io_backend_opt['type'] == 'disk':
            self.paths_lq = paths_from_folder(self.lq_folder)
            self.paths_gt = paths_from_folder(self.gt_folder)
        else:
            raise ValueError(
                f'io_backend not supported')

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.

        lq_path = self.paths_lq[index % len(self.paths_lq)]
        img_bytes = self.file_client.get(lq_path, 'lq')
        img_lq = imfrombytes(img_bytes, float32=True)

        gt_path = self.paths_gt[index % len(self.paths_gt)]
        img_bytes = self.file_client.get(gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True)

        img_ref = img_gt

        # augmentation for training
        if self.opt['phase'] == 'train':
            if_fix = self.opt['if_fix_size']
            gt_size = self.opt['gt_size']
            if not if_fix and self.opt['batch_size_per_gpu'] != 1:
                raise ValueError(
                    f'Param mismatch. Only support fix data shape if batchsize > 1 or num_gpu > 1.')
            # random crop
            img_lq, img_ref = unpaired_random_crop(img_lq, img_ref, if_fix, gt_size)
            # flip, rotation
            img_lq, img_ref = augment([img_lq, img_ref], self.opt['use_flip'], self.opt['use_rot'])

        # BGR to RGB, HWC to CHW, numpy to tensor

        img_lq, img_ref = img2tensor([img_lq, img_ref], bgr2rgb=True, float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_ref, self.mean, self.std, inplace=True)

        return {
            'lq': img_lq,
            'ref': img_ref,
            'lq_path': lq_path,
            'ref_path': gt_path,
        }

    def __len__(self):
        return len(self.paths_lq)
        # return 100
