import logging
import os
import torch
from collections import OrderedDict
from torch.nn.parallel import DataParallel, DistributedDataParallel

from codes.models import lr_scheduler as lr_scheduler
from codes.utils.dist_util import master_only

logger = logging.getLogger('codes')

class BaseModel():
    """Base model."""

    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device('cuda' if opt['num_gpu'] != 0 else 'cpu')
        self.is_train = opt['is_train']
        self.schedulers = []
        self.optimizers = []

    def feed_data(self, data):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        pass

    def save(self, epoch, current_iter):
        """Save networks and training state."""
        pass

    def validation(self, dataloader, current_iter, tb_logger, save_img=False):
        """Validation function.

        Args:
            dataloader (torch.utils.data.DataLoader): Validation dataloader.
            current_iter (int): Current iteration.
            tb_logger (tensorboard logger): Tensorboard logger.
            save_img (bool): Whether to save images. Default: False.
        """
        if self.opt['dist']:
            self.dist_validation(dataloader, current_iter, tb_logger, save_img)
        else:
            self.nondist_validation(dataloader, current_iter, tb_logger,
                                    save_img)

    def validation_speed(self, dataloader, times_per_img=50, num_imgs=20, size=None):
        """Validation function.

        Args:
            dataloader (torch.utils.data.DataLoader): Validation dataloader.
            current_iter (int): Current iteration.
            tb_logger (tensorboard logger): Tensorboard logger.
            save_img (bool): Whether to save images. Default: False.
        """
        self.nondist_validation_speed(dataloader, times_per_img, num_imgs, size)

    def get_current_log(self):
        return self.log_dict

    def model_to_device(self, net):
        """Model to device. It also warps models with DistributedDataParallel
        or DataParallel.

        Args:
            net (nn.Module)
        """
        net = net.to(self.device)
        if self.opt['dist']:
            find_unused_parameters = self.opt.get('find_unused_parameters',
                                                  False)
            net = DistributedDataParallel(
                net,
                device_ids=[torch.cuda.current_device()],
                find_unused_parameters=find_unused_parameters)
        elif self.opt['num_gpu'] > 1:
            net = DataParallel(net)
        return net

    def setup_schedulers(self):
        """Set up schedulers."""
        train_opt = self.opt['train']
        scheduler_type = train_opt['scheduler'].pop('type')
        if scheduler_type in ['MultiStepLR', 'MultiStepRestartLR']:
            for optimizer in self.optimizers:
                self.schedulers.append(
                    lr_scheduler.MultiStepRestartLR(optimizer,
                                                    **train_opt['scheduler']))
        elif scheduler_type == 'CosineAnnealingRestartLR':
            for optimizer in self.optimizers:
                self.schedulers.append(
                    lr_scheduler.CosineAnnealingRestartLR(
                        optimizer, **train_opt['scheduler']))
        else:
            raise NotImplementedError(
                f'Scheduler {scheduler_type} is not implemented yet.')

    def get_bare_model(self, net):
        """Get bare model, especially under wrapping with
        DistributedDataParallel or DataParallel.
        """
        if isinstance(net, (DataParallel, DistributedDataParallel)):
            net = net.module
        return net

    @master_only
    def print_network(self, net):
        """Print the str and parameter number of a network.

        Args:
            net (nn.Module)
        """
        if isinstance(net, (DataParallel, DistributedDataParallel)):
            net_cls_str = (f'{net.__class__.__name__} - '
                           f'{net.module.__class__.__name__}')
        else:
            net_cls_str = f'{net.__class__.__name__}'

        net = self.get_bare_model(net)
        net_str = str(net)
        net_params = sum(map(lambda x: x.numel(), net.parameters()))

        logger.info(
            f'Network: {net_cls_str}, with parameters: {net_params:,d}')
        logger.info(net_str)

    def _set_lr(self, lr_groups_l):
        """Set learning rate for warmup.

        Args:
            lr_groups_l (list): List for lr_groups, each for an optimizer.
        """
        for optimizer, lr_groups in zip(self.optimizers, lr_groups_l):
            for param_group, lr in zip(optimizer.param_groups, lr_groups):
                param_group['lr'] = lr

    def _get_init_lr(self):
        """Get the initial lr, which is set by the scheduler.
        """
        init_lr_groups_l = []
        for optimizer in self.optimizers:
            init_lr_groups_l.append(
                [v['initial_lr'] for v in optimizer.param_groups])
        return init_lr_groups_l

    def update_learning_rate(self, current_iter, warmup_iter=-1):
        """Update learning rate.

        Args:
            current_iter (int): Current iteration.
            warmup_iter (int)： Warmup iter numbers. -1 for no warmup.
                Default： -1.
        """
        if current_iter > 1:
            for scheduler in self.schedulers:
                scheduler.step()
        # set up warm-up learning rate
        if current_iter < warmup_iter:
            # get initial lr for each group
            init_lr_g_l = self._get_init_lr()
            # modify warming-up learning rates
            # currently only support linearly warm up
            warm_up_lr_l = []
            for init_lr_g in init_lr_g_l:
                warm_up_lr_l.append(
                    [v / warmup_iter * current_iter for v in init_lr_g])
            # set learning rate
            self._set_lr(warm_up_lr_l)

    def get_current_learning_rate(self):
        return [
            param_group['lr']
            for param_group in self.optimizers[0].param_groups
        ]

    def get_bare_model(self, net):
        """Get bare model, especially under wrapping with
        DistributedDataParallel or DataParallel. (needed when saving and loading models using DistributedDataParallel)
        """
        if isinstance(net, (DataParallel, DistributedDataParallel)):
            net = net.module
        return net

    @master_only
    def save_network(self, net, net_label, current_iter):
        """Save networks.

        Args:
            net (nn.Module | list[nn.Module]): Network(s) to be saved.
            net_label (str): Network label.
            current_iter (int): Current iter number.
            param_key (str | list[str]): The parameter key(s) to save network.
                Default: 'params'.
        """
        if current_iter == -1:
            current_iter = 'latest'
        save_filename = f'{net_label}_{current_iter}.pth'
        save_path = os.path.join(self.opt['path']['models'], save_filename)

        net = self.get_bare_model(net)
        state_dict = net.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, save_path)

    def load_network(self, net, load_path, strict=True):
        net = self.get_bare_model(net)
        logger.info(f'Loading {net.__class__.__name__} model from {load_path}.')
        load_net = torch.load(load_path)
        net.load_state_dict(load_net, strict=strict)

    @master_only
    def save_training_state(self, epoch, current_iter):
        """Save training states during training, which will be used for
        resuming.

        Args:
            epoch (int): Current epoch.
            current_iter (int): Current iteration.
        """
        if current_iter != -1:
            state = {
                'epoch': epoch,
                'iter': current_iter,
                'optimizers': [],
                'schedulers': []
            }
            for o in self.optimizers:
                state['optimizers'].append(o.state_dict())
            for s in self.schedulers:
                state['schedulers'].append(s.state_dict())
            save_filename = f'{current_iter}.state'
            save_path = os.path.join(self.opt['path']['training_states'],
                                     save_filename)
            torch.save(state, save_path)

    def resume_training(self, resume_state):
        """Reload the optimizers and schedulers for resumed training.

        Args:
            resume_state (dict): Resume state.
        """
        resume_optimizers = resume_state['optimizers']
        resume_schedulers = resume_state['schedulers']
        assert len(resume_optimizers) == len(
            self.optimizers), 'Wrong lengths of optimizers'
        assert len(resume_schedulers) == len(
            self.schedulers), 'Wrong lengths of schedulers'
        for i, o in enumerate(resume_optimizers):
            self.optimizers[i].load_state_dict(o)
        for i, s in enumerate(resume_schedulers):
            self.schedulers[i].load_state_dict(s)

    def reduce_loss_dict(self, loss_dict):
        """reduce loss dict.

        In distributed training, it averages the losses among different GPUs .

        Args:
            loss_dict (OrderedDict): Loss dict.
        """
        with torch.no_grad():
            if self.opt['dist']:
                keys = []
                losses = []
                for name, value in loss_dict.items():
                    keys.append(name)
                    losses.append(value)
                losses = torch.stack(losses, 0)
                torch.distributed.reduce(losses, dst=0)
                if self.opt['rank'] == 0:
                    losses /= self.opt['world_size']
                loss_dict = {key: loss for key, loss in zip(keys, losses)}

            log_dict = OrderedDict()
            for name, value in loss_dict.items():
                log_dict[name] = value.mean().item()

            return log_dict
