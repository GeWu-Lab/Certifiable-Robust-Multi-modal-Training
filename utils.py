import csv
import random
from functools import partialmethod
import torch
import numpy as np
import os

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 

def save_checkpoint(save_file_path, epoch, video_model, optimizer, scheduler):
    
    if hasattr(video_model, 'module'):
        video_model_state_dict = video_model.module.state_dict()
    else:
        video_model_state_dict = video_model.state_dict()
    
    save_states = {
        'epoch': epoch,
        'state_dict': video_model_state_dict,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
    }
    torch.save(save_states, save_file_path)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Logger(object):

    def __init__(self, path, header):
        # self.log_file = path.open('w')
        self.log_file = open(path, 'w')
        self.logger = csv.writer(self.log_file, delimiter='\t')

        self.logger.writerow(header)
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()


def calculate_accuracy(outputs, targets):
    with torch.no_grad():
        batch_size = targets.size(0)
        _, pred = outputs.topk(1, 1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1))
        n_correct_elems = correct.float().sum().item()

        return n_correct_elems / batch_size




def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_lr(optimizer):
    lrs = []
    for param_group in optimizer.param_groups:
        lr = float(param_group['lr'])
        lrs.append(lr)

    return max(lrs)


def partialclass(cls, *args, **kwargs):

    class PartialClass(cls):
        __init__ = partialmethod(cls.__init__, *args, **kwargs)

    return PartialClass


def write_to_batch_logger(batch_logger, epoch, i, data_loader, losses, accuracies, current_lr):
    if batch_logger is not None:
        batch_logger.log({
            'epoch': epoch,
            'batch': i + 1,
            'iter': (epoch - 1) * len(data_loader) + (i + 1),
            'loss': losses,
            'acc': accuracies,
            'lr': current_lr,
        })


def write_to_epoch_logger(epoch_logger, epoch, losses, accuracies, current_lr):
    if epoch_logger is not None:
        epoch_logger.log({
            'epoch': epoch,
            'loss': losses,
            'acc': accuracies,
            'lr': current_lr
        })

def cosine_sim(x, y):
    return (torch.sum(x * y) / torch.sqrt(torch.sum(x ** 2) * torch.sum(y ** 2))).item()
    
def calculate_cosine_sim(feature):
    cosine = []
    for i in range(feature.shape[0]): 
        for j in range(feature.shape[0]): 
            cosine.append(cosine_sim(feature[i], feature[j]))
    return cosine 

def valid_data_range(data_loader):
    # Test the range of the datasets. 
    audio_min = []
    audio_max = []
    visual_min = []
    visual_max = []
    for i, batch in enumerate(data_loader):
        batch_size = batch['clip'].shape[0]
        visual = batch['clip'].reshape(batch_size, -1)
        audio = batch['audio'].reshape(batch_size, -1)
        audio_min.append(torch.min(audio, axis = 1)[0])
        audio_max.append(torch.max(audio, axis = 1)[0])
        visual_min.append(torch.min(visual, axis = 1)[0])
        visual_max.append(torch.max(visual, axis = 1)[0])
    audio_min = torch.cat(audio_min, 0)
    audio_max = torch.cat(audio_max, 0)
    visual_min = torch.cat(visual_min, 0)
    visual_max = torch.cat(visual_max, 0)
    print("audio_range:{} to {}".format(torch.min(audio_min), torch.max(audio_max)))
    print("visual_range:{} to {}".format(torch.min(visual_min), torch.max(visual_max)))


def calculate_flops(model):
    import thop
    flops, params = thop.profile(model, inputs=(v, a))
    flops, params = thop.clever_format([flops, params], '%.3f')
    print('flops: ', flops, 'params: ', params)
    # from ptflops import get_model_complexity_info
    # flops, params = get_model_complexity_info(model, (, ), as_strings=True, print_per_layer_stat=True)

def get_features(data_loader, model, partial_feature = False):
    
    model.eval()

    v_feature = []
    a_feature = []
    labels = []
    for i, batch in enumerate(data_loader):
        visual = batch['clip'].cuda()
        audio = batch['audio'].cuda()
        targets = batch['target'].cuda()
        v, a, out = model(visual, audio)  
        predict = torch.argmax(out, dim = 1) 
        # idx = predict == targets
        # v, a, targets = v[idx], a[idx], targets[idx]
        labels.append(targets)
        if partial_feature:
            v_feature.append(v.detach()[0].unsqueeze(0))
            a_feature.append(a.detach()[0].unsqueeze(0))
        else:
            v_feature.append(v.detach())
            a_feature.append(a.detach())
    labels = torch.cat(labels)
    v_feature = torch.cat(v_feature, 0)
    a_feature = torch.cat(a_feature, 0)
    # torch.save(v_feature, 'results_KS/Dv_best.pt')
    # torch.save(a_feature, 'results_KS/Da_best.pt')
    return v_feature, a_feature, labels


def get_dataset(opt):
    import hydra 
    train_data = hydra.utils.instantiate(opt.dataset, mode = "train")
    val_data = hydra.utils.instantiate(opt.dataset, mode = "val")
    
    g = torch.Generator()
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=opt.batch_size,
                                               shuffle = True, 
                                               num_workers=opt.n_threads,
                                               pin_memory=True,
                                               worker_init_fn=worker_init_fn,
                                               generator=g)

    val_loader = torch.utils.data.DataLoader(val_data,
                                             batch_size= opt.batch_size,
                                             shuffle=False,
                                             num_workers=opt.n_threads,
                                             pin_memory=True,
                                             worker_init_fn=worker_init_fn,
                                             generator=g)
    return train_loader, val_loader

def get_logger(opt):
    train_logger = Logger(os.path.join(opt.result_path, 'train.log'),
                            ['epoch', 'loss', 'acc', 'lr'])
    train_batch_logger = Logger(
        os.path.join(opt.result_path, 'train_batch.log'),
        ['epoch', 'batch', 'iter', 'loss', 'acc', 'lr'])

    val_logger = Logger(os.path.join(opt.result_path, 'val.log'),
                ['epoch', 'loss', 'acc', 'acc_num'])
    return train_logger, train_batch_logger, val_logger