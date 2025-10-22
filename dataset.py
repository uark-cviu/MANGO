# Torch libraries
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
# Custom libraries
from utils.datasets import SegDataset as Dataset
from utils.transforms import Normalise, Pad, RandomCrop, RandomMirror, ResizeAndScale, \
                             CropAlignToMask, ResizeAlignToMask, ToTensor, ResizeInputs


def create_loaders(dataset, inputs, train_dir, val_dir, train_list, val_list,
                   shorter_side, crop_size, input_size, low_scale, high_scale,
                   normalise_params, batch_size, num_workers, ignore_label):
    """
    Args:
      train_dir (str) : path to the root directory of the training set.
      val_dir (str) : path to the root directory of the validation set.
      train_list (str) : path to the training list.
      val_list (str) : path to the validation list.
      shorter_side (int) : parameter of the shorter_side resize transformation.
      crop_size (int) : square crop to apply during the training.
      low_scale (float) : lowest scale ratio for augmentations.
      high_scale (float) : highest scale ratio for augmentations.
      normalise_params (list / tuple) : img_scale, img_mean, img_std.
      batch_size (int) : training batch size.
      num_workers (int) : number of workers to parallelise data loading operations.
      ignore_label (int) : label to pad segmentation masks with

    Returns:
      train_loader, val loader

    """
    input_names, input_mask_idxs = ['rgb', 'depth'], [0, 2, 1]

    AlignToMask = CropAlignToMask if dataset == 'nyudv2' else ResizeAlignToMask
    composed_trn = transforms.Compose([
        AlignToMask(),
        ResizeAndScale(shorter_side, low_scale, high_scale),
        Pad(crop_size, [123.675, 116.28 , 103.53], ignore_label),
        RandomMirror(),
        RandomCrop(crop_size),
        ResizeInputs(input_size),
        Normalise(*normalise_params),
        ToTensor()
    ])
    composed_val = transforms.Compose([
        AlignToMask(),
        ResizeInputs(input_size),
        Normalise(*normalise_params),
        ToTensor()
    ])
    # Training and validation sets
    trainset = Dataset(dataset=dataset, data_file=train_list, data_dir=train_dir,
                       input_names=input_names, input_mask_idxs=input_mask_idxs,
                       transform_trn=composed_trn, transform_val=composed_val,
                       stage='train', ignore_label=ignore_label)

    validset = Dataset(dataset=dataset, data_file=val_list, data_dir=val_dir,
                       input_names=input_names, input_mask_idxs=input_mask_idxs,
                       transform_trn=None, transform_val=composed_val, stage='val',
                       ignore_label=ignore_label)
    print_log('Created train set {} examples, val set {} examples'.format(len(trainset), len(validset)))
    # Training and validation loaders
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(validset, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader



