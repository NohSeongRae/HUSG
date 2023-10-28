import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from attention_message_passing import *


class BlockGenerator(nn.Module):
    def __init__(self, args, n_building=120, n_street=50, iteration=3):
        super(BlockGenerator).__init__()
        self.N_buildings = n_building
        self.N_streets = n_street
        self.N = n_building + n_street
        self.device = torch.device(f'cuda:{self.local_rank}') if torch.cuda.is_available() else torch.device('cpu')
        self.global_pool = torch_geometric.nn.global_max_pool
        self.conv_layer = torch_geometric.nn.GCNConv  # mock-up
        self.latent_dim = 512
        self.feature_dim = 256
        self.n_head = args.n_head
        self.batch_size=args.batch_size
        self.T = iteration  # Message passing iteration
        self.convlayer=torch_geometric.nn.GATConv
        self.global_pool=torch_geometric.nn.global_max_pool

        # Embedding
        self.exist_emb = nn.Linear(2, int(self.feature_dim / 4))
        self.feature_emb = nn.Linear(int(self.feature_dim / 4) + self.N, int(self.feature_dim / 2))
        self.pos_emb = nn.Linear(2, int(self.feature_dim / 2))
        self.size_emb = nn.Linear(2, int(self.feature_dim / 2))

        # Encoder message passing
        self.e_conv1 = self.convlayer(int(self.feature_dim * 2.0), self.feature_dim, heads=self.head)
        self.e_conv2 = self.convlayer(self.feature_dim * self.head, self.feature_dim, heads=self.head)
        self.e_conv3 = self.convlayer(self.feature_dim * self.head, self.feature_dim, heads=self.head)

        # Latent space
        self.aggregate = nn.Linear(int(self.feature_dim) * self.T, self.latent_dim)
        self.z_mu = nn.Linear(self.latent_dim, self.latent_dim)
        self.z_var = nn.Linear(self.latent_dim, self.latent_dim)
        self.dec_feature_init = nn.Linear(self.latent_dim, self.feature_dim * self.N)

        # Decoder
        self.d_conv1 = self.convlayer((-1, self.feature_dim + self.N), self.feature_dim, heads=self.head)
        self.d_conv2 = self.convlayer(self.feature_dim * self.head, self.feature_dim, heads=self.head)
        self.d_conv3 = self.convlayer(self.feature_dim * self.head, self.feature_dim, heads=self.head)

        # Output
        self.dec_exist = nn.Linear(self.feature_dim, self.feature_dim)
        self.out_exist = nn.Linear(self.feature_dim, 1)

        self.dec_pos_x = nn.Linear(self.feature_dim, self.feature_dim)
        self.out_pos_x = nn.Linear(self.feature_dim, 1)

        self.dec_pos_y = nn.Linear(self.feature_dim, self.feature_dim)
        self.out_pos_y = nn.Linear(self.feature_dim, 1)

        self.dec_height = nn.Linear(self.feature_dim, self.feature_dim)
        self.out_height = nn.Linear(self.feature_dim, 1)

        self.dec_width = nn.Linear(self.feature_dim, self.feature_dim)
        self.out_width = nn.Linear(self.feature_dim, 1)

        if args.use_semantic_embedding:  # use if semantic labels exist
            self.enc_semantic = nn.Linear(self.feature_dim, self.feature_dim)
            self.dec_semantic = nn.Linear(self.feature_dim, self.feature_dim)
            self.out_semantic = nn.Linear(self.feature_dim, 1)

        self.enc_iou = nn.Linear(1, int(self.feature_dim / 4))
        self.dec_iou = nn.Linear(self.feature_dim, self.feature_dim)
        self.out_iou = nn.Linear(self.feature_dim, 1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = nn.init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))

    def reparameterize(self, mu, logvar):
        return (torch.exp(0.5 * logvar)) * (torch.randn_like(torch.exp(0.5 * logvar))) + mu

    def Encoder(self,data):
        #node level attrivutes and edge index
        exist, edge_index, pos, size=data.h, data.edge_index, data.node_pos, data.node_size
        iou=data.iou
        print(f'exist: {exist}\n edge_index: {edge_index} \n pos: {pos}\n size:{size}\n iou:{data.iou} ')
        if data.semantic:
            semantic=data.semantic

        h_iou=F.relu(self.enc_iou(iou))
        h_exist=self.exist_emb(exist)
        h_pos=F.relu(self.pos_emb(pos))
        h_size=F.relu(self.size_emb(size))

        graph_pos=torch.eye(self.N, dtype=torch.float32).to(self.device).repeat(self.batch_size, 1)
        h_exist_pos=torch.cat([h_exist, graph_pos], 1)
        ft=F.relu(self.feature_emb(h_exist_pos))

        input_ft=torch.cat((h_iou, h_size, h_pos, ft), dim=1)

        n_embd_1 = F.relu(self.e_conv1(input_ft, edge_index))
        n_embd_2 = F.relu(self.e_conv2(n_embd_1, edge_index))
        n_embd_3 = F.relu(self.e_conv3(n_embd_2, edge_index))

        g_embd_0 = self.global_pool(input_ft, data.batch)
        g_embd_1 = self.global_pool(n_embd_1, data.batch)
        g_embd_2 = self.global_pool(n_embd_2, data.batch)
        g_embd_3 = self.global_pool(n_embd_3, data.batch)

        # g_embd = torch.cat((g_embd_0, g_embd_1, g_embd_2, g_embd_3, org_graph_feature), 1)

    def forward(self, data):
        batch_size = data.ptr.shape[0] - 1
        mu, log_var = self.encode(data)
        z = self.reparameterize(mu, log_var)
        block_condition = data.block_condition.view(batch_size, 2, 64, 64)
        block_condition = self.cnn_encode(block_condition)
        exist, posx, posy, sizex, sizey, b_shape, b_iou = self.decode(z, block_condition, data.edge_index)
        pos = torch.cat((posx, posy), 1)
        size = torch.cat((sizex, sizey), 1)

        return exist, pos, size, mu, log_var, b_shape, b_iou



# class MultiScopeMessagePassing(nn.Module):
#     def __init__(self, args, N_building=120, N_street=50):
