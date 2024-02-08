import time
import torch
import torch.nn as nn
from utils import *
import os 
import numpy as np

def train(train_loader,val_loader,model,\
            train_logger,val_logger,train_batch_logger,tb_writer,opt,\
            optimizer, scheduler):

    N = len(train_loader.dataset)
    criterion = nn.CrossEntropyLoss().cuda()
    pre_val_acc = 0.0
    train_dict = {"CRMT_JT": train_CRMT_JT, "CRMT_AT": train_CRMT_AT, "CRMT_mix": train_CRMT_mix}   
    filename = opt.method + "_lip.pt"
    for i in range(1, opt.n_epochs + 1): 

        train_alg = train_dict[opt.method]
        train_alg(epoch=i, data_loader=train_loader, model = model, criterion=criterion,
                                optimizer=optimizer, epoch_logger=train_logger, batch_logger=train_batch_logger, tb_writer=tb_writer, opt=opt) 
        scheduler.step()

        if i % opt.val_freq == 0:
            prev_val_loss, val_acc = val_epoch(i, val_loader, model, criterion, val_logger, tb_writer)
            if pre_val_acc < val_acc:
                pre_val_acc = val_acc
                save_file_path = os.path.join(opt.result_path, 'save_best.pth')
                save_checkpoint(save_file_path, i, model, optimizer, scheduler)
    get_lip(model, train_loader, opt, filename)
    
    for epoch in range(1, opt.n_epochs // 3): 
        the_second_step(epoch, train_loader, model, opt, lip_filename = filename)
        if epoch % opt.val_freq == 0:
            prev_val_loss, val_acc = val_epoch(epoch, val_loader, model, criterion, val_logger, tb_writer)
            if pre_val_acc < val_acc:
                pre_val_acc = val_acc
                save_file_path = os.path.join(opt.result_path, 'save_best.pth')
                save_checkpoint(save_file_path, epoch,  model, optimizer, scheduler)

def obtain_input(batch):
    visual = batch['clip'].cuda()
    audio = batch['audio'].cuda()
    targets = batch['target'].cuda()
    return visual, audio, targets

def train_CRMT_JT(epoch, data_loader, model, criterion, optimizer, 
                                 epoch_logger, batch_logger, tb_writer=None, opt=None):
    print('imporved train at epoch {}'.format(epoch))

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_cls = AverageMeter()
    accuracies = AverageMeter()
    start_time = time.time()    
    current_lr = optimizer.param_groups[-1]['lr']

    rho = opt.methods.rho
    num_modal = opt.num_modal  
    model.train()

    for i, batch in enumerate(data_loader):
        visual, audio, labels = obtain_input(batch)
        batch_size = labels.shape[0]
        data_time.update(time.time() - start_time)
        optimizer.zero_grad()
        out_v, out_a, out = model(visual, audio)
        outs = [out_v, out_a]
        rows = torch.arange(batch_size)
        margin_loss = 0.0
        for modality in range(num_modal):
            out_cur = outs[modality]
            margin_loss = margin_loss + (torch.sum(torch.exp(out_cur), dim = 1) * torch.exp(-out_cur[rows, labels]) - 1)
        loss_margin = torch.mean(torch.log(margin_loss + 1e-5)) * rho

        loss = criterion(out, labels)
        loss = loss + loss_margin

        acc = calculate_accuracy(out, labels)
        loss.backward()

        accuracies.update(acc,out.size(0))
        losses_cls.update(loss.item(), out.size(0))
        
        optimizer.step()
        
        batch_time.update(time.time() - start_time)
        start_time = time.time()

        write_to_batch_logger(batch_logger, epoch, i, data_loader, losses_cls.val, accuracies.val, current_lr)

    print('Epoch: [{0}]\t Data_time {data_time.sum:.3f} \t Batch_Time {batch_time.sum:.3f}\t'
            'Loss_cls ({loss_cls.avg:.3f})\t'
            'Acc ({acc.avg:.3f})\t'.
            format(epoch, data_time = data_time, batch_time=batch_time,
                    loss_cls=losses_cls, acc=accuracies), flush=True)

    write_to_epoch_logger(epoch_logger, epoch, losses_cls.avg, accuracies.avg, current_lr)

    if tb_writer is not None:
        tb_writer.add_scalar('train/loss_cls', losses_cls.avg, epoch)
        tb_writer.add_scalar('train/acc', accuracies.avg, epoch)


def get_lip(model, train_dataloader, opt, filename = "Lip.pt"):
    num_class = opt.n_classes
    time_start = time.time()
    Lip = torch.zeros(2, num_class).to(opt.device)
    model.eval()
    
    for i, batch in enumerate(train_dataloader):
        visual, audio, labels = obtain_input(batch)
        data = [visual, audio]
        num_modal = len(data)
        batch_size = labels.shape[0]
        for modality in range(num_modal):
            data[modality].requires_grad = True   
        v, a, out = model(data[0], data[1])
        for t_cls in range(num_class):
            loss = torch.sum(out[:, t_cls])
            loss.backward(retain_graph = True)
            for modality in range(num_modal):    
                grads = data[modality].grad.detach().clone()
                audio.grad.data.zero_()
                norm_grads = torch.norm(grads.view(batch_size, -1), dim = 1)
                Lip[modality, t_cls] = torch.maximum(torch.max(norm_grads), Lip[modality, t_cls])
        if (i + 1) % 100 == 0:        
            print("{} batches, taking time: {}s".format(i, time.time() - time_start))
    torch.save(Lip.cpu().detach(), filename)
    return Lip
    

def train_CRMT_AT(epoch, data_loader, model, criterion, optimizer, 
                                 epoch_logger, batch_logger, tb_writer=None, opt=None):
    print('train at epoch {}'.format(epoch))

    device = opt.device
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_cls = AverageMeter()
    accuracies = AverageMeter()
    end_time = time.time()
    current_lr = optimizer.param_groups[-1]['lr']

    rho = opt.methods.rho
    num_modal = 2
    for i, batch in enumerate(data_loader):
        visual, audio, labels = obtain_input(batch)
        batch_size = labels.shape[0]
        data_time.update(time.time() - end_time)
        visual, audio = generate_AT_sample([visual, audio], labels, model, criterion, epsilon = 0.1)
        optimizer.zero_grad()
        
        out_v, out_a, out = model(visual, audio)
        outs = [out_v, out_a]

        rows = torch.arange(batch_size)
        margin_loss = []
        for v in range(num_modal):
            out_v = outs[v]
            margin_loss.append(torch.sum(torch.exp(out_v), dim = 1) * torch.exp(-out_v[rows, labels]) - 1)

        margin_loss_sum = margin_loss[0] + margin_loss[1]
        loss_margin = torch.mean(torch.log(margin_loss_sum + 1e-5)) * rho

        loss = criterion(out, labels) + loss_margin

        acc = calculate_accuracy(out, labels)
        loss.backward()

        accuracies.update(acc,out.size(0))
        losses_cls.update(loss.item(), out.size(0))
        
        optimizer.step()

        #####################################################################################
        batch_time.update(time.time() - end_time)
        end_time = time.time()

        write_to_batch_logger(batch_logger, epoch, i, data_loader, losses_cls.val, accuracies.val, current_lr)

    print('Epoch: [{0}]\t'
            'Time {batch_time.sum:.3f} ({batch_time.avg:.3f})\t'
            'Loss_cls ({loss_cls.avg:.3f})\t'
            'Acc ({acc.avg:.3f})\t'.
            format(epoch,
                    batch_time=batch_time,
                    loss_cls=losses_cls,
                    acc=accuracies), flush=True)

    write_to_epoch_logger(epoch_logger, epoch, losses_cls.avg, accuracies.avg, current_lr)

    if tb_writer is not None:
        tb_writer.add_scalar('train/loss_cls', losses_cls.avg, epoch)
        tb_writer.add_scalar('train/acc', accuracies.avg, epoch)
    # return result_visual_epoch, result_audio_epoch


def train_CRMT_mix(epoch, data_loader, model, criterion, optimizer, 
                                 epoch_logger, batch_logger, tb_writer=None, opt=None):
    print('CRMT_mix training at epoch {}'.format(epoch))

    device = opt.device
    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_cls = AverageMeter()
    accuracies = AverageMeter()
    end_time = time.time()
    current_lr = optimizer.param_groups[-1]['lr']
    num_class = opt.n_classes
    rho = opt.methods.rho
    num_modal = opt.num_modal
    for i, batch in enumerate(data_loader):
        visual, audio, labels = obtain_input(batch)
        data_time.update(time.time() - end_time)
        batch_size = labels.shape[0]
        data, targets_a, targets_b, lam = mixup_data([visual, audio], labels, alpha=1.0)
        optimizer.zero_grad()
        out_v, out_a, out = model(data[0], data[1])
        outs = [out_v, out_a]
        rows = torch.arange(batch_size)
        same_target = (targets_a != targets_b)
        margin_loss = 0.0
        for modality in range(num_modal):
            numerator = torch.sum(torch.exp(outs[modality]), dim = 1) - outs[modality][rows, targets_a] - outs[modality][rows, targets_b] * same_target
            margin_loss = margin_loss + numerator * torch.exp(-(lam * outs[modality][rows, targets_a] + (1 - lam) * outs[modality][rows, targets_b]))
        loss_margin =torch.mean(torch.log(margin_loss)) * rho
        
        loss = mixup_criterion(criterion, out, targets_a, targets_b, lam) + loss_margin

        acc = calculate_accuracy(out, labels)
        loss.backward()

        accuracies.update(acc,out.size(0))
        losses_cls.update(loss.item(), out.size(0))
        
        optimizer.step()

        #####################################################################################
        batch_time.update(time.time() - end_time)
        end_time = time.time()

        write_to_batch_logger(batch_logger, epoch, i, data_loader, losses_cls.val, accuracies.val, current_lr)

    print('Epoch: [{0}]\t'
            'Time {batch_time.sum:.3f} ({batch_time.avg:.3f})\t'
            'Loss_cls ({loss_cls.avg:.3f})\t'
            'Acc ({acc.avg:.3f})\t'.
            format(epoch,
                    batch_time=batch_time,
                    loss_cls=losses_cls,
                    acc=accuracies), flush=True)

    write_to_epoch_logger(epoch_logger, epoch, losses_cls.avg, accuracies.avg, current_lr)

    if tb_writer is not None:
        tb_writer.add_scalar('train/loss_cls', losses_cls.avg, epoch)
        tb_writer.add_scalar('train/acc', accuracies.avg, epoch)

def the_second_step(epoch, data_loader, model, opt, lip_filename = "Lip.pt"):
    print('The second step training at epoch {}'.format(epoch))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.train()
    a = [model.module.fusion_module.orth_layer.a_0, model.module.fusion_module.orth_layer.a_1]
    optimizer_a = torch.optim.SGD(a, lr=opt.lr * 0.03, momentum=0.9, weight_decay=1e-2)
    Lip_cur = torch.load(lip_filename).to(device)
    num_modal = opt.num_modal
    for i, batch in enumerate(data_loader):
        optimizer_a.zero_grad()
        visual, audio, labels = obtain_input(batch) 
        batch_size = labels.shape[0]
        out_v, out_a, out = model(visual, audio)
        outs = [out_v, out_a]
        mask = torch.ones(out.shape, dtype=torch.bool).to(device)
        rows = torch.arange(batch_size)
        mask[rows, labels] = False  
        masked_out = out.masked_fill(mask == False, float('-inf'))
        _, second_largest_ids = masked_out.max(dim=1)

        denom = []   
        numer = []     
        b = model.module.fusion_module.orth_layer.fc.bias.detach()
        beta = b[labels] - b[second_largest_ids]
        for modality in range(num_modal):
            denom.append(a[modality][labels] * Lip_cur[modality][labels] + a[modality][second_largest_ids]* Lip_cur[modality][second_largest_ids])
            numer.append(a[modality][labels] * outs[modality][rows, labels].detach() - a[modality][second_largest_ids] * outs[modality][rows, second_largest_ids].detach())
        loss_c = - torch.mean((numer[0] + numer[1] + beta) / torch.sqrt(denom[0] ** 2+ denom[1] ** 2))                

        loss_c.backward()
        optimizer_a.step()

def val_epoch(epoch, data_loader, model, criterion, logger, tb_writer=None):
    print('validation at epoch {}'.format(epoch))

    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()
    end_time = time.time()
    acc_num = 0

    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            data_time.update(time.time() - end_time)
            visual, audio, targets = obtain_input(batch)
            a, v, out = model(visual, audio)
            loss = criterion(out, targets)
            acc = calculate_accuracy(out, targets)          
            acc_num += acc*targets.shape[0]
            losses.update(loss.item(), out.size(0))
            accuracies.update(acc, out.size(0))
            batch_time.update(time.time() - end_time)
            end_time = time.time()

        print('Epoch: [{0}]\t'
            'Time {batch_time.sum:.3f} ({batch_time.avg:.3f})\t'
            'Loss_cls ({loss_cls.avg:.3f})\t'
            'Acc ({acc.avg:.3f})\t'.
            format(epoch,
                    batch_time=batch_time,
                    loss_cls=losses,
                    acc=accuracies), flush=True)

    if logger is not None:
        logger.log({'epoch': epoch, 'loss': losses.avg, 'acc': accuracies.avg, 'acc_num': acc_num})

    if tb_writer is not None:
        tb_writer.add_scalar('val/loss', losses.avg, epoch)
        tb_writer.add_scalar('val/acc', accuracies.avg, epoch)
        tb_writer.add_scalar('val/acc_num', acc_num, epoch)

    return losses.avg, accuracies.avg

def mixup_data(x, y, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x[0].size()[0]
    index = torch.randperm(batch_size).to(x[0].device)

    mixed_x = [lam * x[0] + (1 - lam) * x[0][index, :], lam * x[1] + (1 - lam) * x[1][index, :]]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def generate_AT_sample(data, labels, model, criterion, epsilon):
    alpha = epsilon / 5
    data_adv = []
    num_modal = len(data)
    batch_size = data[0].shape[0]
    for v in range(num_modal):
        data_adv.append(data[v].clone().detach())
    for t in range(5):
        for v in range(num_modal):
            data_adv[v].requires_grad = True
        _, _, preds_iter = model(data_adv[0], data_adv[1])
        model.zero_grad()        
        err = criterion(preds_iter, labels)
        err.backward()
        grad = []
        for v in range(num_modal):
            grad.append(data_adv[v].grad.data.reshape(batch_size, -1)) 
        grad_norm = (torch.norm(grad[0], dim = 1) ** 2).reshape((batch_size, 1)) + (torch.norm(grad[1], dim = 1) ** 2).reshape((batch_size, 1)) 
        grad_norm = torch.sqrt(grad_norm + 1e-6)
        for v in range(num_modal):
            grad[v] = grad[v] / (grad_norm.expand(grad[v].shape))
            grad[v] = grad[v].view(data[v].shape)
            i_adv = data_adv[v].detach() + alpha * grad[v]
            eta = torch.clamp(i_adv - data[v], min=-epsilon * 2, max=epsilon* 2)
            i_adv = (data[v] + eta)  
            data_adv[v] = i_adv.detach()  
    return data_adv[0], data_adv[1]