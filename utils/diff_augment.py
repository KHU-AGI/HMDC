import numpy as np
import torch
import torch.nn.functional as F

class Transform:
    prob_flip = 0.5
    aug_mode = 'S' #'multiple or single'
    ratio_scale = 1.2
    ratio_rotate = 15.0
    ratio_crop_pad = 0.125
    ratio_cutout = 0.5 # the size would be 0.5x0.5
    ratio_noise = 0.01 ### original 0.05
    brightness = 1.0
    saturation = 2.0
    contrast = 0.5
    latestseed = -1

    @classmethod
    def diff_augment_idc(cls, images, labels, strategy='color_crop_cutout_flip_scale_rotate', dsa_iter=1, seed=None):
        auges_images = []
        for _ in range(dsa_iter):
            if seed is None:
                cls.Siamese = False
            else:
                cls.Siamese = True
                cls.latestseed = seed
            if strategy == 'None' or strategy == 'none':
                return images
            elif strategy:
                x = images; pbties = strategy.split('_')
                if 'flip' in pbties: pbties.remove('flip');x=Transform.AUGMENT_FNS['flip'][0].__func__(cls, x)
                if 'color' in pbties : pbties.remove('color');x=Transform.AUGMENT_FNS['color'][0].__func__(cls, x)
                is_cutout=True if 'cutout' in pbties else False; 
                if 'cutout' in pbties : pbties.remove('cutout');
                cls.set_seed_DiffAug()
                p = pbties[torch.randint(0, len(pbties), size=(1,)).item()]
                for f in Transform.AUGMENT_FNS[p]:
                    x = f.__func__(cls, x)
                if is_cutout:
                    x = Transform.AUGMENT_FNS['cutout'][0].__func__(cls, x)
                x = x.contiguous()
            auges_images.append(x)
        images = torch.cat(auges_images)
        labels = labels.repeat(dsa_iter)
        return images, labels

    @classmethod
    def diff_augment(cls, images, labels, strategy='color_crop_cutout_flip_scale_rotate', dsa_iter=1, seed=None):
        auges_images = []
        for _ in range(dsa_iter):
            if seed is None:
                cls.Siamese = False
            else:
                cls.Siamese = True
                cls.latestseed = seed
            x = images
            if strategy == 'None' or strategy == 'none':
                return images
            elif strategy:
                pbties = strategy.split('_')
                cls.set_seed_DiffAug()
                p = pbties[torch.randint(0, len(pbties), size=(1,)).item()]
            for f in Transform.AUGMENT_FNS[p]:
                x = f.__func__(cls, x)
            x = x.contiguous()
            auges_images.append(x)
        images = torch.cat(auges_images)
        labels = labels.repeat(dsa_iter)
        return images, labels

    @classmethod
    def cutmix(cls, images, labels, num_classes, alpha=1.0):
        device = images.device
        r = torch.rand(1).item()
        if r < 0.5:
            lam = np.random.beta(alpha, alpha)
            rand_index = torch.randperm(images.size()[0]).to(device)
            label_b = labels[rand_index]
            bbx1, bby1, bbx2, bby2 = cls.rand_bbox(images.size(), lam)
            images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
            rate = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2])).to(device)
            targets = F.one_hot(labels, num_classes).float() * rate.to(device) + \
                    F.one_hot(label_b, num_classes).float() * (1. - rate)
        else:
            targets = F.one_hot(labels, num_classes).float()
        return images, targets

    @classmethod
    def rand_bbox(cls, size, lam):
        W = size[2]
        H = size[3]
        cut_rat = torch.sqrt(torch.tensor(1. - lam))
        cut_w = W * cut_rat
        cut_h = H * cut_rat
        # uniform
        cx = torch.randint(W, (1,))
        cy = torch.randint(H, (1,))
        bbx1 = torch.clip(cx - cut_w // 2, 0, W).long()
        bby1 = torch.clip(cy - cut_h // 2, 0, H).long()
        bbx2 = torch.clip(cx + cut_w // 2, 0, W).long()
        bby2 = torch.clip(cy + cut_h // 2, 0, H).long()
        return bbx1, bby1, bbx2, bby2

    @classmethod
    def set_seed_DiffAug(cls):
        if cls.latestseed == -1:
            return
        else:
            torch.random.manual_seed(cls.latestseed)
            cls.latestseed += 1

    # We implement the following differentiable augmentation strategies based on the code provided in https://github.com/mit-han-lab/data-efficient-gans.
    @classmethod
    def rand_scale(cls, x):
        # x>1, max scale
        # sx, sy: (0, +oo), 1: orignial size, 0.5: enlarge 2 times
        ratio = cls.ratio_scale
        with torch.no_grad():
            cls.set_seed_DiffAug()
            sx = torch.rand(x.shape[0]) * (ratio - 1.0/ratio) + 1.0/ratio
            cls.set_seed_DiffAug()
            sy = torch.rand(x.shape[0]) * (ratio - 1.0/ratio) + 1.0/ratio
            theta = [[[sx[i], 0,  0],
                    [0,  sy[i], 0],] for i in range(x.shape[0])]
            theta = torch.tensor(theta, dtype=torch.float)
            if cls.Siamese: # Siamese augmentation:
                _theta = theta.clone()
                theta[:] = _theta[0]
        grid = F.affine_grid(theta, x.shape, align_corners=True).to(x.device)
        x = F.grid_sample(x, grid)
        return x

    @classmethod
    def rand_rotate(cls, x): # [-180, 180], 90: anticlockwise 90 degree
        ratio = cls.ratio_rotate
        cls.set_seed_DiffAug()
        with torch.no_grad():
            theta = (torch.rand(x.shape[0]) - 0.5) * 2 * ratio / 180 * float(np.pi)
            theta = [[[torch.cos(theta[i]), torch.sin(-theta[i]), 0],
                [torch.sin(theta[i]), torch.cos(theta[i]),  0],]  for i in range(x.shape[0])]
            theta = torch.tensor(theta, dtype=torch.float)
            if cls.Siamese: # Siamese augmentation:
                _theta = theta.clone()
                theta[:] = _theta[0]
        grid = F.affine_grid(theta, x.shape, align_corners=True).to(x.device)
        x = F.grid_sample(x, grid)
        return x

    @classmethod
    def rand_flip(cls, x):
        prob = cls.prob_flip
        with torch.no_grad():
            cls.set_seed_DiffAug()
            randf = torch.rand(x.size(0), 1, 1, 1, device=x.device)
            if cls.Siamese: # Siamese augmentation:
                _rand = randf.clone()
                randf[:] = _rand[0]
        return torch.where(randf < prob, x.flip(3), x)

    @classmethod
    def rand_brightness(cls, x):
        ratio = cls.brightness
        with torch.no_grad():
            cls.set_seed_DiffAug()
            randb = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
            if cls.Siamese:  # Siamese augmentation:
                _rand = randb.clone()
                randb[:] = _rand[0]
        x = x + (randb - 0.5)*ratio
        return x

    @classmethod
    def rand_saturation(cls, x):
        ratio = cls.saturation
        x_mean = x.mean(dim=1, keepdim=True)
        with torch.no_grad():
            cls.set_seed_DiffAug()
            rands = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
            if cls.Siamese:  # Siamese augmentation:
                _rand = rands.clone()
                rands[:] = _rand[0]
        x = (x - x_mean) * (rands * ratio) + x_mean
        return x

    @classmethod
    def rand_contrast(cls, x):
        ratio = cls.contrast
        x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
        with torch.no_grad():
            cls.set_seed_DiffAug()
            randc = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
            if cls.Siamese:  # Siamese augmentation:
                _rand = randc.clone()
                randc[:] = _rand[0]
        x = (x - x_mean) * (randc + ratio) + x_mean
        return x

    @classmethod
    def rand_crop(cls, x):
        # The image is padded on its surrounding and then cropped.
        ratio = cls.ratio_crop_pad
        shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
        with torch.no_grad():
            cls.set_seed_DiffAug()
            translation_x = torch.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device)
            cls.set_seed_DiffAug()
            translation_y = torch.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device)
            if cls.Siamese:  # Siamese augmentation:
                _translation_x = translation_x.clone()
                _translation_y = translation_y.clone()
                translation_x[:] = _translation_x[0]
                translation_y[:] = _translation_y[0]
            grid_batch, grid_x, grid_y = torch.meshgrid(
                torch.arange(x.size(0), dtype=torch.long, device=x.device),
                torch.arange(x.size(2), dtype=torch.long, device=x.device),
                torch.arange(x.size(3), dtype=torch.long, device=x.device),
                indexing='ij'
            )
            grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
            grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
        x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
        x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2)
        return x

    @classmethod
    def rand_cutout(cls, x):
        ratio = cls.ratio_cutout
        cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
        cls.set_seed_DiffAug()
        offset_x = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device)
        cls.set_seed_DiffAug()
        offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device)
        if cls.Siamese:  # Siamese augmentation:
            _offset_x = offset_x.clone()
            _offset_y = offset_y.clone()
            offset_x[:] = _offset_x[0]
            offset_y[:] = _offset_y[0]
        grid_batch, grid_x, grid_y = torch.meshgrid(
            torch.arange(x.size(0), dtype=torch.long, device=x.device),
            torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
            torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
        )
        grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
        grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
        mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
        mask[grid_batch, grid_x, grid_y] = 0
        x = x * mask.unsqueeze(1)
        return x
    
    AUGMENT_FNS = {
        'color': [rand_brightness, rand_saturation, rand_contrast],
        'crop': [rand_crop],
        'cutout': [rand_cutout],
        'flip': [rand_flip],
        'scale': [rand_scale],
        'rotate': [rand_rotate],
    }