import streamlit as st
from torch import nn
import torch
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
import torch.nn.functional as F
from CSIKit.visualization.plot_scenario import *
import numpy as np
import matplotlib.pyplot as plt
import random


def global_standardization(data):
    # tensor维度为(2250, 3, 400, 30)
    mean = data.mean()  # 计算全局均值
    std = data.std()  # 计算全局标准差
    # 进行标准化
    normalized_data = (data - mean) / std
    return normalized_data

def process_file(csi_file):
    #读取单个csv文件中的所有sci信息
    csi_list = []
    np.seterr(divide='ignore', invalid='ignore')  #解决无效值问题，如除以0
    my_reader =IWLBeamformReader()
    csi_data = my_reader.read_file(csi_file)
    for i in range(len(csi_data.frames)):
        if (csi_data.frames[i].csi_matrix[::, ::, ::].shape[2] > 1) & (len(csi_list)<2800) :
            csi_list.append(csi_data.frames[i].csi_matrix[::, ::, 0:2])
    csi_data = np.array(csi_list)
    csi_data = csi_data[..., 0] / csi_data[..., 1]
    # 振幅+相位
    csi_data_an = np.angle(csi_data)
    csi_data_abs = np.log(np.abs(csi_data))        #abs求振幅，log缓解振幅异常值
    csi_data = np.concatenate((csi_data_abs, csi_data_an), axis=1)
    # 检测并替换 inf 值
    csi_data[np.isinf(csi_data)] = 0
    # 检测并替换 nan 值
    csi_data[np.isnan(csi_data)] = 0
    # csi_data = np.abs(csi_data)      #只算振幅时
    return csi_data

class DeformConv2d(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, bias=None, modulation=False):
        """
        新增modulation 参数： 是DCNv2中引入的调制标量
        """
        super(DeformConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ZeroPad2d(padding)
        self.conv = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)

        self.p_conv = nn.Conv2d(inc, 2 * kernel_size * kernel_size, kernel_size=3, padding=1, stride=stride)
        # 输出通道是2N
        nn.init.constant_(self.p_conv.weight, 0)  # 权重初始化为0
        self.p_conv.register_backward_hook(self._set_lr)

        self.modulation = modulation
        if modulation:  # 如果需要进行调制
            # 输出通道是N
            self.m_conv = nn.Conv2d(inc, kernel_size * kernel_size, kernel_size=3, padding=1, stride=stride)
            nn.init.constant_(self.m_conv.weight, 0)
            self.m_conv.register_backward_hook(self._set_lr)  # 在指定网络层执行完backward（）之后调用钩子函数

    @staticmethod
    def _set_lr(module, grad_input, grad_output):  # 设置学习率的大小
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x):  # x: (b,c,h,w)
        offset = self.p_conv(x)  # (b,2N,h,w) 学习到的偏移量 2N表示在x轴方向的偏移和在y轴方向的偏移
        if self.modulation:  # 如果需要调制
            m = torch.sigmoid(self.m_conv(x))  # (b,N,h,w) 学习到的N个调制标量

        dtype = offset.data.type()
        ks = self.kernel_size
        N = offset.size(1) // 2

        if self.padding:
            x = self.zero_padding(x)

        # (b, 2N, h, w)
        p = self._get_p(offset, dtype)

        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1

        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2) - 1), torch.clamp(q_lt[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2) - 1), torch.clamp(q_rb[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        # clip p
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2) - 1), torch.clamp(p[..., N:], 0, x.size(3) - 1)], dim=-1)

        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # (b, c, h, w, N)
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # (b, c, h, w, N)
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        # 如果需要调制
        if self.modulation:  # m: (b,N,h,w)
            m = m.contiguous().permute(0, 2, 3, 1)  # (b,h,w,N)
            m = m.unsqueeze(dim=1)  # (b,1,h,w,N)
            m = torch.cat([m for _ in range(x_offset.size(1))], dim=1)  # (b,c,h,w,N)
            x_offset *= m  # 为偏移添加调制标量

        x_offset = self._reshape_x_offset(x_offset, ks)
        out = self.conv(x_offset)

        return out

    def _get_p_n(self, N, dtype):
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1),
            torch.arange(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1))
        # (2N, 1)
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        p_n = p_n.view(1, 2 * N, 1, 1).type(dtype)

        return p_n

    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(1, h * self.stride + 1, self.stride),
            torch.arange(1, w * self.stride + 1, self.stride))
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)

        return p_0

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1) // 2, offset.size(2), offset.size(3)

        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)
        index = q[..., :N] * padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s + ks].contiguous().view(b, c, h, w * ks) for s in range(0, N, ks)],
                             dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h * ks, w * ks)

        return x_offset


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=30, kernel_size=3,
                      stride=3),
            nn.Conv2d(in_channels=30, out_channels=60, kernel_size=3,
                      stride=3),
            DeformConv2d(inc=60, outc=90, kernel_size=3, padding=1, stride=1, bias=None, modulation=False),
            DeformConv2d(inc=90, outc=150, kernel_size=3, padding=1, stride=1, bias=None, modulation=False),
            nn.Dropout2d(p=0.5),
            # nn.Dropout2d(p=0.2),
            # DeformConv2d(inc=60, outc=90, kernel_size=3, padding=1, stride=1, bias=None, modulation=False),
            # nn.Dropout2d(p=0.5),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, 150))
        self.position = nn.Parameter(torch.randn(265, 150))

    def forward(self, x):
        x = x.view(-1, 3, 400, 60)
        b, _, _, _ = x.shape
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        x = torch.cat([cls_tokens, x], dim=1)
        x += self.position
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size=10 * 15, num_heads=5, dropout=0.5):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.qkv = nn.Linear(emb_size, emb_size * 3)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x, mask=None):
        qkv = rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)
        queries, keys, values = qkv[0], qkv[1], qkv[2]
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy, dim=-1) / scaling
        att = self.att_drop(att)
        # sum up over the third axis
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class ResidualAdd(nn.Module):       #残差连接
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardBlock(nn.Sequential):         #前馈网络
    def __init__(self, emb_size, expansion=4, drop_p=0.5):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class TransformerEncoderBlock(nn.Sequential):        #编码器块
    def __init__(self,
                 emb_size=10 * 15,
                 drop_p=0.5,
                 forward_expansion=4,
                 forward_drop_p=0.5,
                 **kwargs):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, **kwargs),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth=1, **kwargs):
        super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])


class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size, num_classes):
        super().__init__(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, num_classes),
            nn.Softmax(dim=1))

class DCNv2_VIT_model(nn.Sequential):
    def __init__(self,
                 in_channels=3,
                 emb_size=10 * 15,
                 depth=1,
                 *,
                 num_classes=3,
                 **kwargs):
        super().__init__(
            PatchEmbedding(in_channels),
            TransformerEncoder(depth, emb_size=emb_size, **kwargs),
            ClassificationHead(emb_size, num_classes)
        )


def save_uploaded_file(uploaded_file, path):
    # 检查保存路径是否存在，如果不存在则创建
    if not os.path.exists(path):
        os.makedirs(path)

    # 构建完整的文件保存路径
    file_path = os.path.join(path, uploaded_file.name)

    # 以二进制写入模式打开文件
    with open(file_path, "wb") as f:
        # 将上传的文件内容写入到文件中
        f.write(uploaded_file.getvalue())
    return file_path

# save_path = '/Users/1bn/Desktop/temporary_data'


# 创建一个文本输入框，让用户输入文件路径
user_input = st.text_input("请输入空闲的文件路径，例：/Users/1bn/Desktop/temporary_data", "")
save_path = None  # 在按钮逻辑之外初始化
# 创建一个按钮，当按下时保存路径
if st.button('保存路径'):
    if user_input:  # 检查用户是否输入了路径
        # 检查路径是否有效（可选）
        if os.path.exists(user_input):
            st.markdown("路径已保存")
        else:
            # 路径无效，显示错误信息
            st.error('输入的路径不存在，请输入一个有效的路径。')
    else:
        # 用户没有输入路径，显示警告信息
        st.warning('请输入一个路径。')


# 加载模型
with st.spinner("模型加载中，请稍后..."):
    model = DCNv2_VIT_model().to('cpu')
    model.load_state_dict(torch.load('model_weights.pth', map_location='cpu'))
    model.eval()
# 文件上传器，允许上传.dat文件
upload_file = st.file_uploader("请选择一个.dat文件上传", type=['dat'])
spinal_statuses_labels = {
    0: "normal",
    1: "kyphosis",
    2: "scoliosis",
}
if upload_file is not None:
    with st.spinner("数据加载中，请稍后..."):
        file_path = save_uploaded_file(upload_file, user_input)
        data_list = []
        data_list_show = []
        window_size = 400  # 窗口大小
        step_size = 200  # 滑动步长
        tensor = torch.Tensor(process_file(file_path))
        n_samples = tensor.size(0)  # 数据的总长度
        # 400的滑动窗口，并且每次滑动200步来切割数据
        for start in range(0, n_samples - window_size + 1, step_size):
            end = start + window_size
            slice = tensor[start:end, :, :]
            data_list_show.append(slice)
            slice = slice.permute(2, 0, 1)
            data_list.append(slice)
        data = torch.stack(data_list, dim=0)
        data_show = torch.stack(data_list_show, dim=0)
        normalized_data = global_standardization(data)
        normalized_data_show = global_standardization(data_show)
    st.markdown("### 用户上传数据，显示如下: ")
    fig = plt.figure()
    plt.plot(normalized_data_show[5, 0:400, 20, 0])
    plt.title('The CSI')
    plt.xlabel('time')
    plt.ylabel('CSI')
    st.pyplot(fig)
    # 模型预测
    st.markdown("**请点击按钮开始检测**")
    predict = st.button("脊柱状态检测")
    if predict:
        # random_number = random.randint(0, 12)
        predict = model(normalized_data)
        predict = torch.mean(predict, dim=0)
        pred_labels = torch.argmax(predict)
        pred_labels = pred_labels.item()
        st.title("被测人脊柱状态为: {}".format(spinal_statuses_labels[pred_labels]))


