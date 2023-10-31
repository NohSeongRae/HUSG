import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tensorboard

class Trainer:
    def __init__(self,model, batch_size,device,max_epoch, loss_dict,sos_idx, eos_idx, pad_idx, d_street, d_unit, d_model, n_layer, n_head,
                 n_building, n_boundary, dropout, use_checkpoint, checkpoint_epoch, use_tensorboard,
                 val_epoch, save_epoch,
                 weight_decay, warmup_steps, n_street,
                 use_global_attn, use_street_attn, use_local_attn, local_rank, save_dir_path):
        super().__init__()
        self.model=model
        self.loss_sum=0
        self.ext_acc=0
        self.iter_ct=0
        self.batch_size=batch_size
        self.device=device
        self.max_epoch=max_epoch
        self.loss_dict=loss_dict
        self.epoch=0
    def train(self,train_loader):
        self.epoch +=1
        loss_sum=0
        self.model.train()
        optimizer = torch.optim.AdamW(lr=3e-5)
        data_len=len(train_loader)
        num_train_steps=in(data_len/self.batch_size * self.max_epoch)
        num_warmup_steps = int(num_train_steps * 0.1)
        self.scheduler=get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps=num_train_steps)
        for batch_idx, data in enumerate(train_loader):
            data=data.to(self.device)
            optimizer.zero_grad()

            exist, pos, size, mu , logvar, acap, iou=self.model(data)
            exist_gt=data.x[:, 0].unsqueeze(1)
            pos_gt=data.org_node_pos
            size_gt=data.org_node_size
            acap_gt=data.acap_gt
            iou_gt=data.iou

            exist_out=torch.ge(F.sigmoid(exist), 0.5).type(torch.uint8)
            exist_sum_loss=self.loss_dict['Exist_sum_loss'](torch.sum(exist_out), torch.sum(exist_gt))
            exist_loss=self.loss_dict['Exist_BCE_loss'](exist, exist_gt)
            pos_loss=self.loss_dict['Pos_loss'](pos, pos_gt)
            size_loss=self.loss_dict['Size_loss'](size, size_gt)
            acap_loss=self.loss_dict['Acap_loss'](acap, acap_gt)
            iou_loss=self.loss_dict['IOU_loss'](iou, iou_gt)
            kld_loss=torch.mean(-0.5 * torch.sum(1+logvar - mu**2 - logvar.exp(), dim=1), dim=0)

            loss=3.0*exist_loss +4.0*pos_loss+0.5*kld_loss+4.0*size_loss+2.0*exist_sum_loss+0.05*acap_loss+1.0*iou_loss
            loss.backward()

            loss_sum+=loss.item()
            optimizer.step()
            self.scheduler.step()

            print('train/all_loss', loss.item(), int(self.epoch * (
                        num_data / train_loader.batch_size) + batch_idx))  # (num_data / batch_size) = how many batches per epoch
            log_value('train/exist_loss', exist_loss.item(), int(epoch * (
                        num_data / train_loader.batch_size) + batch_idx))  # (num_data / batch_size) = how many batches per epoch
            log_value('train/pos_loss', pos_loss.item(), int(epoch * (
                        num_data / train_loader.batch_size) + batch_idx))  # (num_data / batch_size) = how many batches per epoch
            log_value('train/size_loss', size_loss.item(), int(epoch * (
                        num_data / train_loader.batch_size) + batch_idx))  # (num_data / batch_size) = how many batches per epoch
            log_value('train/kld_loss', kld_loss.item(), int(epoch * (
                        num_data / train_loader.batch_size) + batch_idx))  # (num_data / batch_size) = how many batches per epoch
            log_value('train/extsum_loss', extsum_loss, int(epoch * (
                        num_data / train_loader.batch_size) + batch_idx))  # (num_data / batch_size) = how many batches per epoch
            log_value('train/shape_loss', shape_loss.item(), int(epoch * (
                        num_data / train_loader.batch_size) + batch_idx))  # (num_data / batch_size) = how many batches per epoch
            log_value('train/bldg_iou_loss', iou_loss.item(), int(epoch * (
                        num_data / train_loader.batch_size) + batch_idx))  # (num_data / batch_size) = how many batches per epoch

            correct_ext = (exist_out == data.x[:, 0].unsqueeze(1)).sum() / torch.numel(data.x[:, 0])
            ext_acc += correct_ext

            iter_ct += 1


