import os
import torch
import torch.utils.data as data
import numpy as np 
import pickle
import random
import time
import torch.nn.functional as F


class OfflineMutualRerankDataset(data.Dataset):
    def __init__(self, data_dir, split, topk_i2t, topk_t2i, sim_len, anchor, base_model='CAMERA', dataset='flickr', sim_mat=None, 
                    ti_ratio=5, num_neighbor_to_consider=10, coef=0.8):
        self.topk_i2t = topk_i2t
        self.topk_t2i = topk_t2i
        self.sim_len = sim_len
        self.ti_ratio = ti_ratio
        self.num_neighbor_to_consider = num_neighbor_to_consider
        self.coef = coef
        self.anchor = anchor
        self.split = split
        assert anchor in ['inter', 'intra', 'both']
        end = time.time()

        if split == 'train_subset':
            assert False, 'sim_i2t is unavailable on train set!'
            subset_ration = 1/30
            self.batch_data = {}
            for k, v in sim_mat.items():
                self.batch_data[k] = sim_mat[:int(len(v) * subset_ration)]
        else:
            self.batch_data = torch.load(os.path.join(data_dir, f'{base_model}/{dataset}_batch_data_{split}_float16.pt'))

        assert topk_i2t <= self.batch_data['targets_i2t'].shape[1] - 1
        assert topk_t2i <= self.batch_data['targets_t2i'].shape[1] - 1
        assert sim_len <= self.batch_data['sim_mat_i2t'].shape[-1] // 2 - 1
        assert topk_t2i <= (self.batch_data['sim_mat_i2t_map'].shape[1] - 1) // 5
        print('Loading {} ({}) spends {:.2f}s.'.format(dataset, split, time.time() - end))

        if split in ['test', 'dev', 'testall']:
            self.sim_i2t = np.load(os.path.join(data_dir, f'{base_model}/{dataset}_sim_i2t_{split}_float16.npy'))

        self.sored_id = torch.load(os.path.join(data_dir, f'{base_model}/{dataset}_sorted_id_{split}_100.pt'))
        self.i2t_sorted_id = self.sored_id['sorted_ins_i2t_cut']
        self.i2i_sorted_id = self.sored_id['sorted_ins_i2i_cut']
        self.t2i_sorted_id = self.sored_id['sorted_ins_t2i_cut']
        self.t2t_sorted_id = self.sored_id['sorted_ins_t2t_cut']

    def get_sim_mat(self):
        return self.batch_data

    def __len__(self):
        return self.batch_data['sim_mat_t2i'].shape[0]

    def __getitem__(self, txt_index):
        img_index = txt_index // self.ti_ratio
        # txt query, t2i 
        sim_mat_t2i = self.batch_data['sim_mat_t2i'][txt_index]
        label_t2i = self.batch_data['targets_t2i'][txt_index]
        topk_ins_t2i = self.batch_data['topk_ins_t2i'][txt_index]
        permute_index_t2i = None
        # img query, i2t 
        sim_mat_i2t = self.batch_data['sim_mat_i2t'][img_index]
        label_i2t = self.batch_data['targets_i2t'][img_index]
        topk_ins_i2t = self.batch_data['topk_ins_i2t'][img_index]
        permute_index_i2t = None
        # filter intra-neighbor out 
        sim_mat_t2i = sim_mat_t2i[:self.topk_t2i + 1]
        label_t2i = label_t2i[:self.topk_t2i + 1]
        topk_ins_t2i = topk_ins_t2i[:self.topk_t2i]
        sim_mat_i2t = sim_mat_i2t[:self.topk_i2t + 1]
        label_i2t = label_i2t[:self.topk_i2t + 1]
        topk_ins_i2t = topk_ins_i2t[:self.topk_i2t]
        # select anchors
        num_anc_raw = sim_mat_t2i.shape[1]
        inter_anc = torch.arange(self.sim_len + 1)
        intra_anc = torch.arange(self.sim_len + 1) + num_anc_raw // 2
        if self.anchor == 'inter':
            anc_ids = inter_anc
        elif self.anchor == 'intra':
            anc_ids = intra_anc
        else:
            anc_ids = torch.cat([inter_anc, intra_anc])

        sim_mat_t2i = sim_mat_t2i[:, anc_ids]
        sim_mat_i2t = sim_mat_i2t[:, anc_ids]
        #txt query map for i2t 
        sim_mat_i2t_map = self.batch_data['sim_mat_i2t_map'][txt_index]
        idx_i2t_map = torch.cat([torch.tensor([0]), torch.arange(self.topk_t2i) * self.ti_ratio + torch.randint(0, self.ti_ratio, size=(self.topk_t2i, )) + 1])
        sim_mat_i2t_map = sim_mat_i2t_map[idx_i2t_map]
        #img query map for t2i 
        sim_mat_t2i_map = self.batch_data['sim_mat_t2i_map'][txt_index]
        sim_mat_i2t_map = sim_mat_i2t_map[:, anc_ids]
        sim_mat_t2i_map = sim_mat_t2i_map[:, anc_ids]

        # construct edge matrix 
        def cal_intersect(nbs):
            n = len(nbs)
            A_mat = np.zeros((n, n))
            for i in range(n):
                for j in range(i+1, n):
                    A_mat[i, j] = len(np.intersect1d(nbs[i], nbs[j]))
            A_mat = A_mat + A_mat.T
            A_mat = A_mat + np.eye(n) * nbs.shape[1]
            return A_mat
            
        num_neighbor_to_consider = self.num_neighbor_to_consider

        def A_mat_online():
            ''' i2t graph '''
            t_img_nb = self.t2i_sorted_id[[txt_index]][:, :num_neighbor_to_consider]
            t2i_img_nbs = self.i2i_sorted_id[topk_ins_t2i][:, :num_neighbor_to_consider]
            t_img_nbs = np.concatenate([t_img_nb, t2i_img_nbs], axis=0)
            # intra-neighbor (inter-anchor)
            A_mat_t2i_img_nb = cal_intersect(t_img_nbs)
            t_txt_nb = self.t2t_sorted_id[[txt_index]][:, :num_neighbor_to_consider]
            t2i_txt_nbs = self.i2t_sorted_id[topk_ins_t2i][:, :num_neighbor_to_consider]
            t_txt_nbs = np.concatenate([t_txt_nb, t2i_txt_nbs], axis=0)
            # intra-neighbor (intra-anchor)
            A_mat_t2i_txt_nb = cal_intersect(t_txt_nbs)
            A_mat_t2i = A_mat_t2i_img_nb + A_mat_t2i_txt_nb
            A_mat_t2i_norm = A_mat_t2i / A_mat_t2i.sum(axis=1, keepdims=True)
            sparse_factor = 1./8 * self.coef
            A_mat_t2i_norm = A_mat_t2i_norm * (A_mat_t2i_norm > sparse_factor)
            A_mat_t2i_norm = A_mat_t2i_norm / A_mat_t2i_norm.sum(axis=1, keepdims=True)

            A_mat_t2i = torch.tensor(A_mat_t2i_norm)

            ''' i2t graph '''
            i_txt_nb = self.i2t_sorted_id[[img_index]][:, :num_neighbor_to_consider]
            i2t_txt_nbs = self.t2t_sorted_id[topk_ins_i2t][:, :num_neighbor_to_consider]
            i_txt_nbs = np.concatenate([i_txt_nb, i2t_txt_nbs], axis=0)
            # intra-neighbor (inter-anchor)
            A_mat_i2t_txt_nb = cal_intersect(i_txt_nbs)
            i_img_nb = self.i2i_sorted_id[[img_index]][:, :num_neighbor_to_consider]
            i2t_img_nbs = self.t2i_sorted_id[topk_ins_i2t][:, :num_neighbor_to_consider]
            i_img_nbs = np.concatenate([i_img_nb, i2t_img_nbs], axis=0)
            # intra-neighbor (intra-anchor)
            # calculate intersection (intra-anchor)
            A_mat_i2t_img_nb = cal_intersect(i_img_nbs)
            A_mat_i2t = A_mat_i2t_txt_nb + A_mat_i2t_img_nb
            A_mat_i2t_norm = A_mat_i2t / A_mat_i2t.sum(axis=1, keepdims=True)
            sparse_factor = 1./32 * self.coef
            A_mat_i2t_norm = A_mat_i2t_norm * (A_mat_i2t_norm > sparse_factor)
            A_mat_i2t_norm = A_mat_i2t_norm / A_mat_i2t_norm.sum(axis=1, keepdims=True)

            A_mat_i2t = torch.tensor(A_mat_i2t_norm)
            return A_mat_i2t, A_mat_t2i

        def A_mat_offline():
            A_mat_i2t = self.A_mat['A_mat_i2t'][img_index]
            A_mat_t2i = self.A_mat['A_mat_t2i'][txt_index]
            return A_mat_i2t, A_mat_t2i

        A_mat_i2t, A_mat_t2i = A_mat_online()
        return sim_mat_i2t, label_i2t, permute_index_i2t, img_index, topk_ins_i2t, A_mat_i2t, \
                sim_mat_t2i, label_t2i, permute_index_t2i, txt_index, topk_ins_t2i, A_mat_t2i, \
                sim_mat_i2t_map, sim_mat_t2i_map


def collate_fn(data):
    sim_mat_i2t, label_i2t, permute_index_i2t, img_index, topk_ins_i2t, A_mat_i2t, \
        sim_mat_t2i, label_t2i, permute_index_t2i, txt_index, topk_ins_t2i, A_mat_t2i, \
        sim_mat_i2t_map, sim_mat_t2i_map = zip(*data)

    return (torch.stack(sim_mat_i2t, dim=0), torch.stack(label_i2t, dim=0), permute_index_i2t, np.array(img_index), np.stack(topk_ins_i2t, axis=0), torch.stack(A_mat_i2t, dim=0)), \
            (torch.stack(sim_mat_t2i, dim=0), torch.stack(label_t2i, dim=0), permute_index_t2i, np.array(txt_index), np.stack(topk_ins_t2i, axis=0), torch.stack(A_mat_t2i, dim=0)), \
            (torch.stack(sim_mat_i2t_map, dim=0), torch.stack(sim_mat_t2i_map, dim=0))


def get_loader(split, data_path, topk_i2t, topk_t2i, sim_len, batch_size, direction, anchor, shuffle=True, num_workers=2, \
                    sim_mat=None, base_model='CAMERA', dataset_name='flickr', opt={}):
    dset = OfflineMutualRerankDataset(data_path, split, topk_i2t, topk_t2i, sim_len, anchor, dataset=dataset_name, sim_mat=sim_mat, base_model=base_model)
    loader = torch.utils.data.DataLoader(dataset=dset, batch_size=batch_size, shuffle=shuffle, pin_memory=True, collate_fn=collate_fn, num_workers=num_workers)
    return loader, dset