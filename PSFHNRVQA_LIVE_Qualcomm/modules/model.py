import torch
import torch.nn as nn
import torch.nn.init as init
import functools
from modules.fpn_inception import FPNInception

def weigth_init(m):
    if isinstance(m, (nn.Conv2d, nn.Conv3d)):
        init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


class MyConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, device):
        super(MyConv3d, self).__init__()

        self.conv3d = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm3d(out_channels, eps=1e-4, momentum=0.95),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv3d(x)
        return x


class MyFc(nn.Module):
    def __init__(self, in_channels, out_channels, drop):
        super(MyFc, self).__init__()

        self.fc = nn.Sequential(
            nn.Dropout(drop),
            nn.Linear(in_channels, out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        x = self.fc(x)
        return x

class MyGRU(nn.Module):
    # 输入为 n*c*T*1*1， 展开成 n T c的向量
    # 输出为 n T c2

    def __init__(self, input_size, hidden_size, device, batch_first=True):
        # batch_first=True则输入输出的数据格式为 (batch, seq, feature)
        super(MyGRU, self).__init__()
        self.rnn = nn.GRU(input_size, hidden_size, batch_first=batch_first)
        self.pool = nn.AdaptiveMaxPool1d(1)

        self.hidden_size = hidden_size
        self.device = device

    def forward(self, x):
        t = torch.squeeze(x, dim=3)
        t = torch.squeeze(t, dim=3)
        t = t.permute([0, 2, 1])
        # 输入：n T input_size     输出：n T hidden_size
        r, h1 = self.rnn(t, self._get_initial_state(t.size(0), self.device))
        r = r.permute([0, 2, 1])  # n hidden_size T
        f = self.pool(r).squeeze(2)  # 输出：n hidden_size
        return f

    def _get_initial_state(self, batch_size, device):
        h0 = torch.zeros(1, batch_size, self.hidden_size, device=device)
        return h0

# 整体的网络
class MyModel(nn.Module):

    def __init__(self, device):
        super(MyModel, self).__init__()
        self.device = device
        self._init_modules()
        # 预训练模型地址
        weights_path=r'./pretrained/fpn_inception.h5'

        # 加载模型
        model_g = FPNInception(norm_layer=get_norm_layer(norm_type='instance'))
        model = nn.DataParallel(model_g)
        model.load_state_dict(torch.load(weights_path)['model'])
        self.premodel = model.cuda()
        self.premodel.eval()

        # 固定预训练模型
        # k是可训练参数的名字，v是包含可训练参数的一个实体
        for k,v in self.premodel.named_parameters():
            v.requires_grad = False #固定参数

    def _init_modules(self):

        #################################
        # stage
        
        # stage 0
        # c_in:6->c_out:32 t_in:8->t_out:8  256*256->128*128
        self.stage_0 = nn.Sequential(
            MyConv3d(6, 16, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), device=self.device),
            MyConv3d(16, 32, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), device=self.device),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=0),
        )
        # c_in:128->c_out:32 t_in:8->t_out:8  128*128->128*128
        self.stage_0_side = nn.Sequential(
            MyConv3d(128, 64, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0), device=self.device),
            MyConv3d(64, 32, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), device=self.device),
        )

        # stage 1
        # c_in:64->c_out:32 t_in:8->t_out:4  128*128->64*64
        self.stage_1 = nn.Sequential(
            MyConv3d(64, 32, kernel_size=(2, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1), device=self.device),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=0),
        )
        # c_in:256->c_out:32 t_in:8->t_out:4  64*64->64*64
        self.stage_1_side = nn.Sequential(
            MyConv3d(256, 64, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0), device=self.device),
            MyConv3d(64, 32, kernel_size=(2, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1), device=self.device),
        )

        # stage 2
        # c_in:64->c_out:32 t_in:4->t_out:2  64*64->32*32
        self.stage_2 = nn.Sequential(
            MyConv3d(64, 32, kernel_size=(2, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1), device=self.device),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=0),
        )
        # c_in:256->c_out:32 t_in:8->t_out:2  32*32->32*32
        self.stage_2_side = nn.Sequential(
            MyConv3d(256, 64, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0), device=self.device),
            MyConv3d(64, 32, kernel_size=(2, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1), device=self.device),
            MyConv3d(32, 32, kernel_size=(2, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1), device=self.device),
        )

        # stage 3
        # c_in:64->c_out:32 t_in:2->t_out:1  32*32->16*16
        self.stage_3 = nn.Sequential(
            MyConv3d(64, 32, kernel_size=(2, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1), device=self.device),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=0),
        )
        # c_in:256->c_out:32 t_in:8->t_out:1  16*16->16*16
        self.stage_3_side = nn.Sequential(
            MyConv3d(256, 64, kernel_size=(2, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0), device=self.device),
            MyConv3d(64, 32, kernel_size=(2, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1), device=self.device),
            MyConv3d(32, 32, kernel_size=(2, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1), device=self.device),
        )

        # stage 4
        # c_in:64->c_out:32 t_in:1->t_out:1  16*16->8*8
        self.stage_4 = nn.Sequential(
            MyConv3d(64, 32, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), device=self.device),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=0),
        )
        # c_in:256->c_out:32 t_in:8->t_out:1  8*8->8*8
        self.stage_4_side = nn.Sequential(
            MyConv3d(256, 64, kernel_size=(2, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0), device=self.device),
            MyConv3d(64, 32, kernel_size=(2, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1), device=self.device),
            MyConv3d(32, 32, kernel_size=(2, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1), device=self.device),
        )

        #######################################
        # side net

        # c_in:64->c_out:64 t_in:8->t_out:8  128*128->1*1
        self.side_net_0 = nn.Sequential(
            MyConv3d(64, 64, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), device=self.device),
            MyConv3d(64, 64, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), device=self.device),
            MyConv3d(64, 64, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), device=self.device),
            MyConv3d(64, 64, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), device=self.device),
            nn.MaxPool3d(kernel_size=(1, 8, 8), stride=1, padding=0),
        )

        # c_in:64->c_out:64 t_in:4->t_out:4  64*64->1*1
        self.side_net_1 = nn.Sequential(
            MyConv3d(64, 64, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), device=self.device),
            MyConv3d(64, 64, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), device=self.device),
            MyConv3d(64, 64, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), device=self.device),
            nn.MaxPool3d(kernel_size=(1, 8, 8), stride=1, padding=0),
        )

        # c_in:64->c_out:64 t_in:2->t_out:2  32*32->1*1
        self.side_net_2 = nn.Sequential(
            MyConv3d(64, 64, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), device=self.device),
            MyConv3d(64, 64, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), device=self.device),
            nn.MaxPool3d(kernel_size=(1, 8, 8), stride=1, padding=0),
        )

        # c_in:64->c_out:128 t_in:1->t_out:1  16*16->1*1
        self.side_net_3 = nn.Sequential(
            MyConv3d(64, 128, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), device=self.device),
            nn.MaxPool3d(kernel_size=(1, 8, 8), stride=1, padding=0),
        )
        
        # c_in:64->c_out:128 t_in:1->t_out:1  8*8->1*1
        self.side_net_4 = nn.Sequential(
            MyConv3d(64, 128, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0), device=self.device),
            nn.MaxPool3d(kernel_size=(1, 8, 8), stride=1, padding=0),
        )

        self.rnn_0 = MyGRU(64, 128, device=self.device, batch_first=True)
        self.rnn_1 = MyGRU(64, 128, device=self.device, batch_first=True)
        self.rnn_2 = MyGRU(64, 128, device=self.device, batch_first=True)

        ####################################
        # fc net 得出q
        self.fc_net_0 = nn.Sequential(
            MyFc(128, 64, 0.6),
            MyFc(64, 1, 0.6),
        )
        self.fc_net_1 = nn.Sequential(
            MyFc(128, 64, 0.6),
            MyFc(64, 1, 0.6),
        )
        self.fc_net_2 = nn.Sequential(
            MyFc(128, 64, 0.6),
            MyFc(64, 1, 0.6),
        )
        self.fc_net_3 = nn.Sequential(
            MyFc(128, 64, 0.6),
            MyFc(64, 1, 0.6),
        )
        self.fc_net_4 = nn.Sequential(
            MyFc(128, 64, 0.6),
            MyFc(64, 1, 0.6),
        )
        
        self.att_net = nn.Sequential(
            nn.Linear(128 * 5, 128 * 5),
            nn.ReLU(inplace=True),
            nn.Linear(128 * 5, 128 * 5),
            nn.Sigmoid(),
        )
        
        self.fc_net_5 = nn.Sequential(
            MyFc(128 * 5, 128, 0.65),
            MyFc(128, 1, 0.6),
        )
        

    def mapout(self, x):
        # x torch.Size([1, 3, 8, 256, 256])
        tmp = x
        x1,x2,x3,x4,x5,x6,x7,x8 = tmp.split(1, 2)	# 在2维进行间隔为1的拆分

        # 去除时间维度
        x1 = x1.squeeze(2)
        x2 = x2.squeeze(2)
        x3 = x3.squeeze(2)
        x4 = x4.squeeze(2)
        x5 = x5.squeeze(2)
        x6 = x6.squeeze(2)
        x7 = x7.squeeze(2)
        x8 = x8.squeeze(2)

        # 预训练模型使用，输入的input尺寸 torch.Size([1, 3, 256, 256])
        x1 = [x1.cuda()]
        res1, map10, map11, map12, map13, map14 = self.premodel(*x1)
        x2 = [x2.cuda()]
        res2, map20, map21, map22, map23, map24 = self.premodel(*x2)
        x3 = [x3.cuda()]
        res3, map30, map31, map32, map33, map34 = self.premodel(*x3)
        x4 = [x4.cuda()]
        res4, map40, map41, map42, map43, map44 = self.premodel(*x4)
        x5 = [x5.cuda()]
        res5, map50, map51, map52, map53, map54 = self.premodel(*x5)
        x6 = [x6.cuda()]
        res6, map60, map61, map62, map63, map64 = self.premodel(*x6)
        x7 = [x7.cuda()]
        res7, map70, map71, map72, map73, map74 = self.premodel(*x7)
        x8 = [x8.cuda()]
        res8, map80, map81, map82, map83, map84 = self.premodel(*x8)

        # 创建第2维，沿着该维度进行堆叠
        res = torch.stack([res1, res2, res3, res4, res5, res6, res7, res8], 2)
        map0 = torch.stack([map10, map20, map30, map40, map50, map60, map70, map80], 2)
        map1 = torch.stack([map11, map21, map31, map41, map51, map61, map71, map81], 2)
        map2 = torch.stack([map12, map22, map32, map42, map52, map62, map72, map82], 2)
        map3 = torch.stack([map13, map23, map33, map43, map53, map63, map73, map83], 2)
        map4 = torch.stack([map14, map24, map34, map44, map54, map64, map74, map84], 2)

        # print('\n res', res.shape) # 残差图
        # print('\n map0', map0.shape) # map0
        # print('\n map1', map1.shape) # map1
        # print('\n map2', map2.shape) # map2
        # print('\n map3', map3.shape) # map3
        # print('\n map4', map4.shape) # map4
        
        return res, map0, map1, map2, map3, map4

    def forward(self, x):

        res, map0, map1, map2, map3, map4 = self.mapout(x)
        x = torch.cat([x, res], 1)

        c0 = self.stage_0(x)
        GANmap0 = self.stage_0_side(map0)
        c0 = torch.cat([c0, GANmap0], 1)

        c1 = self.stage_1(c0)
        GANmap1 = self.stage_1_side(map1)
        c1 = torch.cat([c1, GANmap1], 1)

        c2 = self.stage_2(c1)
        GANmap2 = self.stage_2_side(map2)
        c2 = torch.cat([c2, GANmap2], 1)

        c3 = self.stage_3(c2)
        GANmap3 = self.stage_3_side(map3)
        c3 = torch.cat([c3, GANmap3], 1)
 
        c4 = self.stage_4(c3)
        GANmap4 = self.stage_4_side(map4)
        c4 = torch.cat([c4, GANmap4], 1)

        # for i in (c0, c1, c2, c3, c4):
        #     print(i.size())

        s0 = self.side_net_0(c0)
        p0 = self.rnn_0(s0)
        t0 = torch.flatten(p0, 1)
        q0 = self.fc_net_0(t0).view(-1, 1)


        s1 = self.side_net_1(c1)
        p1 = self.rnn_1(s1)
        t1 = torch.flatten(p1, 1)
        q1 = self.fc_net_1(t1).view(-1, 1)

        s2 = self.side_net_2(c2)
        p2 = self.rnn_2(s2)
        t2 = torch.flatten(p2, 1)
        q2 = self.fc_net_2(t2).view(-1, 1)

        p3 = self.side_net_3(c3)
        t3 = torch.flatten(p3, 1)
        q3 = self.fc_net_3(t3).view(-1, 1)

        p4 = self.side_net_4(c4)
        t4 = torch.flatten(p4, 1)
        q4 = self.fc_net_4(t4).view(-1, 1)

        # for i in (t0, t1, t2, t3, t4):
        #    print(i.size())

        s = torch.cat((t0, t1, t2, t3, t4), 1)
        sw = self.att_net(s)
        t = s.mul(sw)  # 对应元素相乘
        q = self.fc_net_5(t).view(-1, 1) 

        # 返回二维张量 n * 5
        return torch.cat((q0, q1, q2, q3, q4, q), 1)


class MyHuberLoss(torch.nn.Module):
    def __init__(self):
        super(MyHuberLoss, self).__init__()
        self.delta = 2

    # Huber loss: 与平方误差损失相比，鲁棒性高，对数据中的游离点较不敏感
    # Pseudo-Huber loss
    def pseudo_huber_loss(self, target, pred, delta):
        return delta ** 2 * ((1 + ((pred - target) / delta) ** 2) ** 0.5 - 1)
        

    def compute_p(self, input):
        """
        :param input: 一维 [ q1, q2, q3, q4, q, target ]
        :return: 零维 单个样本的loss
        """
        pred = input[0:-1]
        target = input[-1]
        q_n = input.size()[-1] - 1
        alpha = torch.zeros(q_n)

        # 权重 0-1
        alpha[0:-1] = 0.2 * (torch.tanh(pred[0:-1] - target)) ** 2
        alpha[-1] = 1
        loss = [self.pseudo_huber_loss(target, pred[i], self.delta)
                * alpha[i] for i in range(q_n)]
        loss = sum(loss)
        return loss

    def forward(self, output, label):
        """
        :param output: 二维 [ n * [q1, q2, q3, q4, q] ]
        :param label: 二维 [ n * 1 ]
        :return: 零维 n个batch的loss均值
        """
        q0, q1, q2, q3, q4, q = output.split(1, 1)
        video_train_pred_q0 = q0.squeeze(1)
        video_train_pred_q1 = q1.squeeze(1)
        video_train_pred_q2 = q2.squeeze(1)
        video_train_pred_q3 = q3.squeeze(1)
        video_train_pred_q4 = q4.squeeze(1)
        video_train_pred_q = q.squeeze(1)
        video_train_label = label.squeeze(1)
        # print(video_train_pred_q.shape)
        pearson = PearsonCorrelation()
        train_plcc_0 = pearson(video_train_pred_q0, video_train_label)
        train_plcc_1 = pearson(video_train_pred_q1, video_train_label)
        train_plcc_2 = pearson(video_train_pred_q2, video_train_label)
        train_plcc_3 = pearson(video_train_pred_q3, video_train_label)
        train_plcc_4 = pearson(video_train_pred_q4, video_train_label)
        train_plcc_5 = pearson(video_train_pred_q, video_train_label)
        print('\033[1;31m plcc_q0: ','{:.3f}'.format(train_plcc_0.data), '\033[0m')
        print('\033[1;31m plcc_q1: ','{:.3f}'.format(train_plcc_1.data), '\033[0m')
        print('\033[1;31m plcc_q2: ','{:.3f}'.format(train_plcc_2.data), '\033[0m')
        print('\033[1;31m plcc_q3: ','{:.3f}'.format(train_plcc_3.data), '\033[0m')
        print('\033[1;31m plcc_q4: ','{:.3f}'.format(train_plcc_4.data), '\033[0m')
        print('\033[1;31m plcc_final: ','{:.3f}'.format(train_plcc_5.data), '\033[0m')
        q_info = torch.cat((output, label), 1)
        loss = list(map(self.compute_p, q_info))
        loss = torch.stack(loss, 0).float()
        loss = torch.mean(loss)
        print('\033[1;31m loss_old ','{:.3f}'.format(loss.data), '\033[0m')
        sum = 5 - train_plcc_0 - train_plcc_1 - train_plcc_2 - train_plcc_3 - train_plcc_4
        loss = 1*loss + 1*((1-train_plcc_0)**2 + (1-train_plcc_1)**2 + (1-train_plcc_2)**2  
               + (1-train_plcc_3)**2 + (1-train_plcc_4)**2)/sum + 1*(1-train_plcc_5)
        return loss

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=True)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer
    

def squeezeANDconcat(x1,x2,x3,x4,x5,x6,x7,x8):
    x1 = x1.unsqueeze(2)
    x2 = x2.unsqueeze(2)
    x3 = x3.unsqueeze(2)
    x4 = x4.unsqueeze(2)
    x5 = x5.unsqueeze(2)
    x6 = x6.unsqueeze(2)
    x7 = x7.unsqueeze(2)
    x8 = x8.unsqueeze(2)
    return 

class PearsonCorrelation(nn.Module):
    def forward(self,tensor_1,tensor_2):
        x = tensor_1
        y = tensor_2

        vx = x - torch.mean(x)
        vy = y - torch.mean(y)

        cost = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
        return cost