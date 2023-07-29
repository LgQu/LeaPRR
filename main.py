import argparse
import os
from numpy import save
import torch
import math
from torch.cuda.amp.grad_scaler import GradScaler
from torch import cuda, optim
import torch.nn.functional as F
from tqdm import tqdm
import sys
import tensorboard_logger as tb_logger
import time
import numpy as np
import pickle

from data import get_loader
from model.MutualRerankTransformer import MutualRerankTransformer
from utils import MetricLogger, AverageMeter, WarmupCos_Scheduler, WarmupStep_Scheduler 
from model.loss import TripletLoss, ContrastiveLoss
from utils import create_optimizer, is_main_process, print_options
from evaluation import evalrank_single


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpuid', type=int, default=0)
    parser.add_argument('--logger_path', default='runs/runX', help='Path to save the model and Tensorboard log.')
    parser.add_argument('--dataset', type=str, default='flickr')
    parser.add_argument('--direction', type=str, default='both', choices=['both', 'I2T', 'T2I'])
    parser.add_argument('--anchor', type=str, default='both', choices=['both', 'inter', 'intra'])
    parser.add_argument('--topk_i2t', type=int, default=32)
    parser.add_argument('--topk_t2i', type=int, default=8)
    parser.add_argument('--sim_len', type=int, default=64)
    parser.add_argument('--data_path', default='/data', help='path to datasets')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--only_test', type=str, default=None)
    parser.add_argument('--workers', default=4, type=int, help='Number of data loader workers.')
    parser.add_argument('--print_freq', default=10, type=int)
    parser.add_argument('--base_model', type=str, default='CAMERA')
    # training
    parser.add_argument('--batch_size', default=512, type=int, help='Size of a training mini-batch.')
    parser.add_argument('--num_epochs', default=30, type=int, help='Number of training epochs.')
    parser.add_argument('--stop_at_epoch', type=int, default=None)
    parser.add_argument('--base_lr', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--warmup_epochs', type=int, default=0)
    parser.add_argument('--warmup_lr', type=float, default=0)
    parser.add_argument('--update_every', type=int, default=1)
    parser.add_argument('--final_lr', type=float, default=0)
    parser.add_argument('--lr_step', type=float, default=999, help='Epoch to decay')
    parser.add_argument('--lr_gamma', type=float, default=0.1, help='decaying rate')
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--clip_max_norm', type=float, default=0)
    parser.add_argument('--val_step', default=200, type=int, help='Number of steps to run validation.')
    # model
    parser.add_argument('--embed_dim', type=int, default=768)
    parser.add_argument('--depth', type=int, default=2)
    parser.add_argument('--num_heads', type=int, default=24)
    parser.add_argument('--mlp_ratio', type=int, default=4)
    parser.add_argument('--qkv_bias', action='store_true', default=True)
    parser.add_argument('--drop_rate', type=float, default=0.0)
    parser.add_argument('--drop_path_rate', type=float, default=0.0)
    parser.add_argument('--attn_drop_rate', type=float, default=0.0)
    # loss
    parser.add_argument('--alpha', type=float, default=0.0)
    parser.add_argument('--Hinge_trade_off', type=float, default=1.0)
    parser.add_argument('--weight_decay', type=float, default=0.0) 
    parser.add_argument('--margin', type=float, default=0.2)
    parser.add_argument('--max_violation', action='store_true', default=False)
    parser.add_argument('--KL_trade_off', type=float, default=1.0)
    parser.add_argument('--Trip_trade_off', type=float, default=1.0)
    opt = parser.parse_args()

    if opt.gpuid >= 0:
        opt.device = torch.device('cuda:{}'.format(opt.gpuid))

    if opt.stop_at_epoch is not None:
        if opt.stop_at_epoch > opt.num_epochs:
            raise Exception
    else:
        opt.stop_at_epoch = opt.num_epochs

    print_options(opt)
    tb_logger.configure(opt.logger_name, flush_secs=5)

    # Define Model
    topk_dim = opt.sim_len * 2 if opt.anchor == 'both' else opt.sim_len
    model = MutualRerankTransformer(embed_dim=opt.embed_dim,
                              topk_dim=topk_dim,
                              topk_i2t=opt.topk_i2t, 
                              topk_t2i=opt.topk_t2i, 
                              depth=opt.depth,
                              num_heads=opt.num_heads,
                              mlp_ratio=opt.mlp_ratio,
                              qkv_bias=opt.qkv_bias,
                              drop_rate=opt.drop_rate,
                              attn_drop_rate=opt.attn_drop_rate,
                              drop_path_rate=opt.drop_path_rate).to(opt.device)
    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model_without_ddp.parameters() if p.requires_grad)
    print('>> number of params:{:.2f}M'.format(n_parameters / 1e6))

    test_split = 'test' if opt.dataset in ['flickr', 'flickr_offline'] else 'testall'
    test_loader, test_dset = get_loader(test_split, opt.data_path, opt.topk_i2t-1, opt.topk_t2i-1, opt.sim_len-1, opt.batch_size, opt.direction, opt.anchor, \
                                            shuffle=False, num_workers=opt.workers, base_model=opt.base_model, dataset_name=opt.dataset, opt=opt)

    if opt.only_test is not None:
        test(model, opt, test_loader, test_dset, ckpt_name=opt.only_test)
        exit()

    # Loss
    cal_contrastive_loss = ContrastiveLoss()
    cal_triplet_loss = TripletLoss(opt.margin, opt.max_violation)


    # DataLoader
    train_loader, train_dset = get_loader('train', opt.data_path, opt.topk_i2t-1, opt.topk_t2i-1, opt.sim_len-1, opt.batch_size, opt.direction, opt.anchor, \
                                            shuffle=True, num_workers=opt.workers, base_model=opt.base_model, dataset_name=opt.dataset, opt=opt)
    val_loader, val_dset = get_loader('dev', opt.data_path, opt.topk_i2t-1, opt.topk_t2i-1, opt.sim_len-1, opt.batch_size, opt.direction, opt.anchor, \
                                            shuffle=False, num_workers=opt.workers, base_model=opt.base_model, dataset_name=opt.dataset, opt=opt)

    # Define optimizer
    param_dicts = create_optimizer(opt.weight_decay, model_without_ddp)
    optimizer = optim.SGD(param_dicts, lr=opt.base_lr * opt.batch_size / 256, weight_decay=opt.weight_decay, momentum=opt.momentum)

    start_epoch = 0
    if opt.resume is not None:
        model_path = os.path.join(opt.logger_name, opt.resume + '.pth')
        if os.path.isfile(model_path):
            print(">> Loading checkpoint:\n>> '{}'".format(model_path))
            checkpoint = torch.load(model_path, map_location='cpu')
            start_epoch = checkpoint['epoch']
            model_without_ddp.load_state_dict(checkpoint['state_dict'], strict=False)
            print(">>>> loaded checkpoint:\n>>>> '{}' (epoch {})".format(model_path, checkpoint['epoch']))
        else:
            print(">> No checkpoint found at '{}'".format(model_path))

    lr_scheduler = WarmupCos_Scheduler(optimizer=optimizer,
                                       warmup_epochs=opt.warmup_epochs,
                                       warmup_lr=opt.warmup_lr * opt.batch_size * opt.update_every / 256,
                                       num_epochs=opt.num_epochs,
                                       base_lr=opt.base_lr * opt.batch_size * opt.update_every / 256,
                                       final_lr=opt.final_lr * opt.batch_size * opt.update_every / 256,
                                       iter_per_epoch=int(len(train_loader) / opt.update_every))

    lr_scheduler.iter = max(int(len(train_loader) * start_epoch / opt.update_every), 0)

    # Start training
    metric_logger = MetricLogger(delimiter=" ")
    data_time = AverageMeter()
    scaler = GradScaler()
    model_path = None
    best_rsum = 0

    for epoch in range(start_epoch, opt.stop_at_epoch):
        header = '>> Train Epoch: [{}]'.format(epoch)
        optimizer.zero_grad()
        end = time.time()
        for idx, ((sim_mat_i2t, targets_i2t,  permute_index_i2t, _, _, A_mat_i2t), (sim_mat_t2i, targets_t2i, permute_index_t2i, _, _, A_mat_t2i), sim_mat_map) in \
                enumerate(metric_logger.log_every(train_loader, opt.print_freq, header)):

            metric_logger.meters['Data'].update(time.time() - end)
            model.Eiters += 1
            model.train()
            targets_i2t = targets_i2t.to(opt.device)
            targets_t2i = targets_t2i.to(opt.device)
            sim_mat_i2t = sim_mat_i2t.to(opt.device)
            sim_mat_t2i = sim_mat_t2i.to(opt.device)
            A_mat_i2t = A_mat_i2t.to(opt.device)
            A_mat_t2i = A_mat_t2i.to(opt.device)

            model.set_mode('normal')
            (exp_sim_i2t, MSE_loss_i2t, Hinge_loss_i2t, sim_i2t, norm_x_i2t), \
            (exp_sim_t2i, MSE_loss_t2i, Hinge_loss_t2i, sim_t2i, norm_x_t2i) \
                                = model(sim_mat_i2t.half(), sim_mat_t2i.half(), A_mat_i2t.half(), A_mat_t2i.half(), permute_index_i2t, permute_index_t2i)

            Contrastive_loss_i2t = cal_contrastive_loss(exp_sim_i2t, targets_i2t)
            Contrastive_loss_t2i = cal_contrastive_loss(exp_sim_t2i, targets_t2i)
            Contrastive_loss = Contrastive_loss_i2t + Contrastive_loss_t2i
            MSE_loss = MSE_loss_i2t + MSE_loss_t2i
            Hinge_loss = Hinge_loss_i2t + Hinge_loss_t2i
            loss = Contrastive_loss + opt.alpha * MSE_loss + opt.Hinge_trade_off * Hinge_loss
            # triplet
            Triplet_loss_i2t = cal_triplet_loss(sim_i2t, targets_i2t)
            Triplet_loss_t2i = cal_triplet_loss(sim_t2i, targets_t2i)
            Triplet_loss = Triplet_loss_i2t + Triplet_loss_t2i
            metric_logger.meters['Trip'].update(Triplet_loss.item())
            loss += Triplet_loss * opt.Trip_trade_off
            # cross alignment
            sim_mat_i2t_map, sim_mat_t2i_map = sim_mat_map
            sim_mat_i2t_map = sim_mat_i2t_map.to(opt.device)
            sim_mat_t2i_map = sim_mat_t2i_map.to(opt.device)
            model.set_mode('align')
            A_mat_t2i = A_mat_t2i[:, :opt.topk_t2i, :opt.topk_t2i]
            A_mat_i2t = A_mat_i2t[:, :opt.topk_i2t, :opt.topk_i2t]
            (exp_sim_i2t_map, MSE_loss_i2t_map, Hinge_loss_i2t_map, sim_i2t_map, norm_x_i2t_map), \
            (exp_sim_t2i_map, MSE_loss_t2i_map, Hinge_loss_t2i_map, sim_t2i_map, norm_x_t2i_map) \
                                = model(sim_mat_i2t_map.half(), sim_mat_t2i_map.half(), A_mat_t2i.half(), A_mat_i2t.half(), permute_index_t2i, permute_index_i2t)

            model.set_mode('normal')
            Contrastive_loss_i2t_map = cal_contrastive_loss(exp_sim_i2t_map, targets_t2i)
            Contrastive_loss_t2i_map = cal_contrastive_loss(exp_sim_t2i_map, targets_i2t)
            Contrastive_loss_map = Contrastive_loss_i2t_map + Contrastive_loss_t2i_map
            MSE_loss_map = MSE_loss_i2t_map + MSE_loss_t2i_map
            Hinge_loss_map = Hinge_loss_i2t_map + Hinge_loss_t2i_map
            loss_map = Contrastive_loss_map + opt.alpha * MSE_loss_map + opt.Hinge_trade_off * Hinge_loss_map

            temperature = opt.temperature
            exp_sim_i2t, exp_sim_t2i = torch.exp(sim_i2t / temperature), torch.exp(sim_t2i / temperature)
            exp_sim_i2t_map, exp_sim_t2i_map = torch.exp(sim_i2t_map / temperature), torch.exp(sim_t2i_map / temperature)
            dist_i2t = exp_sim_i2t / exp_sim_i2t.sum(dim=-1, keepdim=True)
            dist_t2i = exp_sim_t2i / exp_sim_t2i.sum(dim=-1, keepdim=True)
            dist_i2t_map = exp_sim_i2t_map / exp_sim_i2t_map.sum(dim=-1, keepdim=True)
            dist_t2i_map = exp_sim_t2i_map / exp_sim_t2i_map.sum(dim=-1, keepdim=True)

            kl_i2t = F.kl_div(dist_t2i_map.log(), dist_i2t, reduction='batchmean')
            kl_t2i = F.kl_div(dist_i2t_map.log(), dist_t2i, reduction='batchmean')
            kl_loss = kl_i2t + kl_t2i
            metric_logger.meters['KL'].update(kl_loss.item())

            Contrastive_loss += Contrastive_loss_map
            MSE_loss += MSE_loss_map
            Hinge_loss += Hinge_loss_map
            loss += loss_map
            loss += kl_loss * opt.KL_trade_off 
            metric_logger.meters['Loss'].update(loss.item())

            if not math.isfinite(Contrastive_loss.item()):
                print(">> Contrastive loss is nan, skip to the next iteration")
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                continue

            if not math.isfinite(MSE_loss.item()):
                print(">> MSE loss is nan, skip to the next iteration")
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                continue

            scaler.scale(loss).backward()
            metric_logger.meters['Contr'].update(Contrastive_loss.item())
            metric_logger.meters['MSE'].update(MSE_loss.item())
            metric_logger.meters['Hinge'].update(Hinge_loss.item())
            metric_logger.meters['LR'].update(optimizer.param_groups[0]['lr'])
            tb_logger.log_value('epoch', epoch, step=model.Eiters)
            metric_logger.tb_log(tb_logger, step=model.Eiters)

            if (idx + 1) % opt.update_every == 0:
                if opt.clip_max_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), opt.clip_max_norm)
                lr = lr_scheduler.step()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            if model.Eiters % opt.val_step == 0:
                recall = evaluate(model, val_loader, val_dset.sim_i2t, opt.device, opt.direction, update_log=True)
                rsum = recall['rsum']
                is_best = rsum > best_rsum
                best_rsum = max(rsum, best_rsum)
                if is_best:
                    save_ckpt(epoch, opt.logger_name, model_without_ddp, 'checkpoint_best.pth', recall)

            end = time.time()

        recall = evaluate(model, val_loader, val_dset.sim_i2t, opt.device, opt.direction, update_log=True)
        rsum = recall['rsum']
        is_best = rsum > best_rsum
        best_rsum = max(rsum, best_rsum)
        if is_best:
            save_ckpt(epoch, opt.logger_name, model_without_ddp, 'checkpoint_best.pth', recall)
    
    save_ckpt(epoch, opt.logger_name, model_without_ddp, 'checkpoint_last.pth', recall)
    # Evaluation on Test Set
    test(model, opt, test_loader, test_dset, ckpt_name='checkpoint_best')

def test(model, opt, test_loader, test_dset, ckpt_name='checkpoint_best'):
    print('#'*25 + ' Test ' + '#'*25)
    model_path = os.path.join(opt.logger_name, ckpt_name+'.pth')
    print(">> Loading checkpoint:\n>> '{}'".format(model_path))
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    print(">>>> loaded checkpoint:\n>>>> '{}' (epoch {})".format(model_path, checkpoint['epoch']))
    print('Best Result on Dev Set: ')
    recall = checkpoint['recall']
    r = tuple([recall['r1'], recall['r5'], recall['r10']])
    rt = tuple([recall['rt1'], recall['rt5'], recall['rt10']])
    ar = (r[0] + r[1] + r[2]) / 3
    ari = (rt[0] + rt[1] + rt[2]) / 3
    rsum = r[0] + r[1] + r[2] + rt[0] + rt[1] + rt[2]
    print("rsum: %.1f" % rsum)
    print("Average i2t Recall: %.1f" % ar)
    print("Image to text: %.1f %.1f %.1f " % r)
    print("Average t2i Recall: %.1f" % ari)
    print("Text to image: %.1f %.1f %.1f " % rt)
    print('-'*55)
    evaluate(model, test_loader, test_dset.sim_i2t, opt.device, opt.direction)


@torch.no_grad()
def evaluate(model, loader, sim_i2t, device, direction, update_log=False, prefix='', without_mapping=False):
    model.eval()
    end = time.time()

    sim_i2t, sim_t2i = sim_i2t.copy(), sim_i2t.T.copy()
    ranks_i2t = np.argsort(-sim_i2t, axis=1)
    ranks_t2i = np.argsort(-sim_t2i, axis=1)
    for _, ((sim_mat_i2t, _,  permute_index_i2t, img_index, tok_index_i2t, A_mat_i2t), \
            (sim_mat_t2i, _,  permute_index_t2i, txt_index, tok_index_t2i, A_mat_t2i), _) in enumerate(loader):
        sim_mat_i2t = sim_mat_i2t.to(device)
        sim_mat_t2i = sim_mat_t2i.to(device)
        A_mat_i2t = A_mat_i2t.to(device)
        A_mat_t2i = A_mat_t2i.to(device)

        if without_mapping:
            (rerank_scores_i2t, _), (rerank_scores_t2i, _) = model.forward_feature_without_mapping(sim_mat_i2t.half(), sim_mat_t2i.half(), permute_index_i2t, permute_index_t2i)
        else:    
            (rerank_scores_i2t, _), (rerank_scores_t2i, _) = model.forward_feature(sim_mat_i2t.half(), sim_mat_t2i.half(), A_mat_i2t.half(), A_mat_t2i.half(), permute_index_i2t, permute_index_t2i)
        bs, topk_i2t = rerank_scores_i2t.size()
        bs, topk_t2i = rerank_scores_t2i.size()

        rerank_scores_i2t = rerank_scores_i2t.detach().cpu()
        rerank_scores_t2i = rerank_scores_t2i.detach().cpu()

        rows_i2t = np.expand_dims(img_index, axis=1).repeat(topk_i2t, axis=1)
        sim_i2t[rows_i2t, tok_index_i2t] = rerank_scores_i2t.numpy().copy()
        rerank_indices_i2t = np.argsort(-rerank_scores_i2t.numpy(), axis=1)
        ranks_i2t[img_index, :topk_i2t] = ranks_i2t[rows_i2t, rerank_indices_i2t]

        rows_t2i = np.expand_dims(txt_index, axis=1).repeat(topk_t2i, axis=1)
        sim_t2i[rows_t2i, tok_index_t2i] = rerank_scores_t2i.numpy().copy()
        rerank_indices_t2i = np.argsort(-rerank_scores_t2i.numpy(), axis=1)
        ranks_t2i[txt_index, :topk_t2i] = ranks_t2i[rows_t2i, rerank_indices_t2i]

    sim = (sim_i2t + sim_t2i.T) / 2
    r1, r5, r10, rt1, rt5, rt10, rsum = evalrank_single(sim)

    if update_log:
        tb_logger.log_value(prefix+'r1', r1, step=model.Eiters)
        tb_logger.log_value(prefix+'r5', r5, step=model.Eiters)
        tb_logger.log_value(prefix+'r10', r10, step=model.Eiters)
        tb_logger.log_value(prefix+'rt1', rt1, step=model.Eiters)
        tb_logger.log_value(prefix+'rt5', rt5, step=model.Eiters)
        tb_logger.log_value(prefix+'rt10', rt10, step=model.Eiters)
        tb_logger.log_value(prefix+'rsum', rsum, step=model.Eiters)
    r_dict =  {'r1': r1, 'r5': r5, 'r10': r10, 'rt1': rt1, 'rt5': rt5, 'rt10': rt10, 'rsum': rsum}
    return r_dict

def save_ckpt(epoch, logger_name, model_without_ddp, name, recall):
    if is_main_process():
        model_path = os.path.join(logger_name, name)
        torch.save({'epoch': epoch, 'state_dict': model_without_ddp.state_dict(), 'recall': recall}, model_path)

if __name__ == '__main__':
    main()





