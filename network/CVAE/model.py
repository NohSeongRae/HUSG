import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from attention_message_passing import *


class BlockGenerator(nn.Module):
    def __init__(self, args, n_building=120, n_street=50, iteration=3, frequency_num=32, bottleneck=128, image_size=64,
                 inner_channel=80):
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
        self.batch_size = args.batch_size
        self.T = iteration  # Message passing iteration
        self.convlayer = torch_geometric.nn.GATConv
        self.global_pool = torch_geometric.nn.global_max_pool

        channel_num = int((image_size / 2 ** 4) ** 2 * inner_channel)
        self.inner_channel = 80
        self.image_size = 64
        self.linear_bottleneck = nn.Linear(channel_num, bottleneck)
        self.bottleneck = int(bottleneck)

        # CNN Encoder
        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(2, int(self.inner_channel / 8), 3, stride=1, padding=1),  # b, 16, 10, 10
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # b, 16, 5, 5
            nn.Conv2d(int(self.inner_channel / 8), int(self.inner_channel / 4), 3, stride=1, padding=1),  # b, 8, 3, 3
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 8, 2, 2
            nn.Conv2d(int(self.inner_channel / 4), int(self.inner_channel / 2), 3, stride=1, padding=1),  # b, 8, 3, 3
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 8, 2, 2
            nn.Conv2d(int(self.inner_channel / 2), int(self.inner_channel), 3, stride=1, padding=1),  # b, 8, 3, 3
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2)  # b, 8, 2, 2
        )
        # Embedding
        self.exist_emb = nn.Linear(2, int(self.feature_dim / 4))
        self.feature_emb = nn.Linear(int(self.feature_dim / 4) + self.N, int(self.feature_dim / 2))
        self.pos_emb = nn.Linear(2, int(self.feature_dim / 2))
        self.size_emb = nn.Linear(2, int(self.feature_dim / 2))
        self.enc_block_scale = nn.Linear(1, 20)

        # Encoder message passing
        self.e_conv1 = self.convlayer(int(self.feature_dim * 2.0), self.feature_dim, heads=self.head)
        self.e_conv2 = self.convlayer(self.feature_dim * self.head, self.feature_dim, heads=self.head)
        self.e_conv3 = self.convlayer(self.feature_dim * self.head, self.feature_dim, heads=self.head)

        # Latent space
        self.aggregate = nn.Linear(int(self.feature_dim * (2.0 + self.n_head * self.T)), self.latent_dim)
        self.z_mu = nn.Linear(self.latent_dim, self.latent_dim)
        self.z_var = nn.Linear(self.latent_dim, self.latent_dim)
        self.dec_feature_init = nn.Linear(self.latent_dim + bottleneck, self.feature_dim * self.N)

        # Decoder
        self.d_conv1 = self.convlayer((-1, self.feature_dim + self.N), self.feature_dim, heads=self.head)
        self.d_conv2 = self.convlayer(self.feature_dim * self.head, self.feature_dim, heads=self.head)
        self.d_conv3 = self.convlayer(self.feature_dim * self.head, self.feature_dim, heads=self.head)

        # Output
        self.dec_exist = nn.Linear(self.feature_dim * self.n_head, self.feature_dim)
        self.out_exist = nn.Linear(self.feature_dim, 1)

        self.dec_pos_x = nn.Linear(self.feature_dim * self.n_head, self.feature_dim)
        self.out_pos_x = nn.Linear(self.feature_dim, 1)

        self.dec_pos_y = nn.Linear(self.feature_dim * self.n_head, self.feature_dim)
        self.out_pos_y = nn.Linear(self.feature_dim, 1)

        self.dec_height = nn.Linear(self.feature_dim * self.n_head, self.feature_dim)
        self.out_height = nn.Linear(self.feature_dim, 1)

        self.dec_width = nn.Linear(self.feature_dim * self.n_head, self.feature_dim)
        self.out_width = nn.Linear(self.feature_dim, 1)

        if args.use_semantic_embedding:  # use if semantic labels exist
            self.enc_semantic = nn.Linear(self.feature_dim, self.feature_dim)
            self.dec_semantic = nn.Linear(self.feature_dim, self.feature_dim)
            self.out_semantic = nn.Linear(self.feature_dim, 1)

        self.enc_iou = nn.Linear(1, int(self.feature_dim / 4))
        self.dec_iou = nn.Linear(self.feature_dim * self.n_head, self.feature_dim)
        self.out_iou = nn.Linear(self.feature_dim, 1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = nn.init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))

    def cnn_encode(self, x):
        x = self.cnn_encoder(x)
        x = torch.flatten(x, 1)
        x = self.linear_bottleneck(x)
        return x

    def reparameterize(self, mu, logvar):
        return (torch.exp(0.5 * logvar)) * (torch.randn_like(torch.exp(0.5 * logvar))) + mu

    def Encoder(self, data):
        # node level attrivutes and edge index
        exist, edge_index, pos, size = data.h, data.edge_index, data.node_pos, data.node_size
        iou = data.iou
        print(f'exist: {exist}\n edge_index: {edge_index} \n pos: {pos}\n size:{size}\n iou:{data.iou} ')
        if data.semantic:
            semantic = data.semantic

        h_iou = F.relu(self.enc_iou(iou))
        h_exist = self.exist_emb(exist)
        h_pos = F.relu(self.pos_emb(pos))
        h_size = F.relu(self.size_emb(size))

        graph_pos = torch.eye(self.N, dtype=torch.float32).to(self.device).repeat(self.batch_size, 1)
        h_exist_pos = torch.cat([h_exist, graph_pos], 1)
        ft = F.relu(self.feature_emb(h_exist_pos))

        input_ft = torch.cat((h_iou, h_size, h_pos, ft), dim=1)

        n_embd_1 = F.relu(self.e_conv1(input_ft, edge_index))
        n_embd_2 = F.relu(self.e_conv2(n_embd_1, edge_index))
        n_embd_3 = F.relu(self.e_conv3(n_embd_2, edge_index))

        g_embd_0 = self.global_pool(input_ft, data.batch)
        g_embd_1 = self.global_pool(n_embd_1, data.batch)
        g_embd_2 = self.global_pool(n_embd_2, data.batch)
        g_embd_3 = self.global_pool(n_embd_3, data.batch)

        g_embd = torch.cat((g_embd_0, g_embd_1, g_embd_2, g_embd_3, org_graph_feature), 1)
        latent = self.aggregate(g_embd)
        mu = self.z_mu(latent)
        log_var = self.z_var(latent)

        return [mu, log_var]

    def Decoder(self, z, block_condition, edge_index):
        batch_size = z.shape[0]

        z = torch.cat((z, block_condition), 1)
        z = self.dec_feature_init(z).view(z.shape[0] * self.N, -1)

        one_hot = torch.eye(self.N, dtype=torch.float32).to(self.device).repeat(batch_size, 1)
        z = torch.cat([z, one_hot], 1)

        d_embd_0 = F.relu(z)
        d_embd_1 = F.relu(self.d_conv1(d_embd_0, edge_index))
        d_embd_2 = F.relu(self.d_conv2(d_embd_1, edge_index))
        d_embd_3 = F.relu(self.d_conv3(d_embd_2, edge_index))

        exist = self.dec_exist(d_embd_3)
        pos_x = F.relu(self.dec_pos_x(d_embd_3))
        pos_x = self.out_pos_x(pos_x)

        pos_y = F.relu(self.dec_pos_y(d_embd_3))
        pos_y = self.out_pos_y(pos_y)

        height = F.relu(self.dec_height(d_embd_3))
        height = self.out_height(height)

        width = F.relu(self.dec_width(d_embd_3))
        width = self.out_width(width)

        iou = F.relu(self.dec_iou(d_embd_3))
        iou = self.out_iou(iou)

        return exist, pos_x, pos_y, height, width, iou

    def forward(self, data):
        batch_size = data.ptr.shape[0] - 1
        mu, log_var = self.encode(data)
        z = self.reparameterize(mu, log_var)
        block_condition = data.block_condition.view(batch_size, 2, 64, 64)
        block_condition = self.cnn_encode(block_condition)
        exist, pos_x, pos_y, height, width, b_shape, iou = self.decode(z, block_condition, data.edge_index)
        pos = torch.cat((pos_x, pos_y), 1)
        size = torch.cat((height, width), 1)

        return exist, pos, size, mu, log_var, b_shape, iou

# class MultiScopeMessagePassing(nn.Module):
#     def __init__(self, args, N_building=120, N_street=50):
