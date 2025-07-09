import torch
import torch.nn as nn

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if not training or drop_prob == 0.:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # 将随机张量向下取整，生成0或1的掩码
    output = x.div(keep_prob) * random_tensor  # 按照保留比例缩放
    return output

# 对残差分支进行随机置零
class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768, norm_layer=None):
        '''
        img_size: 输入图像大小
        patch_size: 图像块大小
        in_channels: 输入通道数
        embed_dim: 嵌入维度
        norm_layer: 归一化层
        '''
        super(PatchEmbed, self).__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]# patch数量
        # 获得patch
        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        self.norm_layer = norm_layer if norm_layer is not None else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert (H, W) == self.img_size, f"Input image size must be {self.img_size}, but got {(H, W)}"
        
        x = self.proj(x) # (B, embed_dim, grid_size[0], grid_size[1])
        # x.flatten(2)：将每个 patch 的空间位置展平成一个序列，shape 变为 (B, embed_dim, num_patches)
        # .transpose(1, 2)：交换通道和 patch 维度，变为 (B, num_patches, embed_dim)，即每一行是一个 patch 的向量。
        x = x.flatten(2).transpose(1, 2)
        return self.norm_layer(x)
        
class Attention(nn.Module):
    def __init__(self, embed_dim, num_heads, qkv_bias=False, qk_scale=None, attn_dropout=0.0, out_dropout=0.0):
        '''
        embed_dim: 输入维度
        num_heads: 注意力头数
        qkv_bias: 是否使用偏置
        qk_scale: q和k的缩放因子
        attn_dropout: 注意力权重的 dropout 比例
        out_dropout: 输出的 dropout 比例
        '''
        super(Attention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads # 注意力头数
        self.head_dim = embed_dim // num_heads # 每个头的维度

        assert self.head_dim * num_heads == embed_dim, "注意力头数和嵌入维度不匹配"

        self.scaling = qk_scale if qk_scale is not None else self.head_dim ** -0.5 # 对点积结果进行缩放，保证数值稳定
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias) # 线性变换，将输入映射到查询、键、值三个向量
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(embed_dim, embed_dim) # 模型重新融合和变换各个头的信息
        self.out_drop = nn.Dropout(out_dropout)

    def forward(self, x):
        B, N, C = x.shape # [batch_size, num_patches + 1, embed_dim] 加了一个分类 token
        # qkv(): -> [batch_size, num_patches + 1, 3 * embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, head_dim]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, head_dim] 调整维度顺序
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        # qkv[0] 是 Q，qkv[1] 是 K，qkv[2] 是 V，形状都是 (batch_size, num_heads, num_patches + 1, head_dim)。
        q, k, v = qkv[0], qkv[1], qkv[2]

        # k.transpose(-2, -1) 将k的最后两个维度交换，得到 (batch_size, num_heads, head_dim, num_patches + 1)方便计算点积
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn_weights = (q @ k.transpose(-2, -1)) * self.scaling
        attn_weights = attn_weights.softmax(dim=-1)
        attn_weights = self.attn_drop(attn_weights)

        # @: multiply -> [batch_size, num_heads, num_patches + 1, head_dim]
        # transpose: -> [batch_size, num_patches + 1, num_heads, head_dim]
        # reshape: -> [batch_size, num_patches + 1, embed_dim]
        out = (attn_weights @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.out_drop(out)
        return out   
    
class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        '''
        in_features: 输入特征维度
        hidden_features: 隐藏层特征维度
        out_features: 输出特征维度
        '''
        super(MLP, self).__init__()
        if hidden_features is None:
            hidden_features = in_features
        if out_features is None:
            out_features = in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
class Block(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, qkv_bias=False, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        '''
        embed_dim: 嵌入维度
        num_heads: 注意力头数
        mlp_ratio: MLP的隐藏层维度与嵌入维度的比率
        drop_path: 跳过层的比例
        '''
        super(Block, self).__init__()
        self.norm1 = norm_layer(embed_dim)
        self.attn = Attention(embed_dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_dropout=attn_drop, out_dropout=drop)
        self.drop_path = nn.Dropout(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = MLP(in_features=embed_dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x))) # 残差连接
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
    
class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, num_classes=1000, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, representation_size=None, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0, norm_layer=None, act_layer=None):
        '''
        img_size: 输入图像大小
        patch_size: 图像块大小
        in_channels: 输入通道数
        num_classes: 分类数
        embed_dim: 嵌入维度
        depth: Transformer的层数
        num_heads: 注意力头数
        mlp_ratio: MLP的隐藏层维度与嵌入维度的比率
        '''
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        norm_layer = norm_layer if norm_layer is not None else nn.LayerNorm
        self.norm = norm_layer(embed_dim)
        act_layer = act_layer if act_layer is not None else nn.GELU

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_channels=in_channels, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))  # 分类 token
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))  # 位置编码
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # 生成drop path的比例，通常越深的层drop path概率越大

        self.blocks = nn.Sequential(
            *[
                Block(
                    embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer
                ) for i in range(depth)
            ]
        )
        
        # 如果指定了representation_size，则在最后一层添加一个线性层用于输出
        if representation_size is not None:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(
                nn.Linear(embed_dim, representation_size),
                nn.Tanh()  # 或者使用其他激活函数
            )
        else:
            self.has_logits = False
            self.num_features = embed_dim
            self.pre_logits = nn.Identity()

        # 分类头
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        # 初始化参数
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(_init_vit_weights)

    def forward_feature(self, x):
        # [B, C, H, W] -> [B, num_patches, embed_dim]
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1) # 扩展分类 token 的形状为 [B, 1, embed_dim]
        x = torch.cat((cls_token, x), dim=1)  # 在patch前添加分类 token
        
        x = x + self.pos_embed
        x = self.pos_drop(x) # 应用位置编码和dropout

        x = self.blocks(x)  # 通过Transformer块
        x = self.norm(x)  # 最后归一化
        return self.pre_logits(x[:, 0])  # 返回分类 token 的特征
    
    def forward(self, x):
        x = self.forward_feature(x)
        x = self.head(x)  # 分类头
        return x


def _init_vit_weights(m):
    """
    参考github方法https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/blob/master/pytorch_classification/vision_transformer/vit_model.py#L272
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)

def vit_base_patch16_224(num_classes: int = 1000):
    """
    ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    链接: https://pan.baidu.com/s/1zqb08naP0RPqqfSXfkB2EA  密码: eu9f
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=None,
                              num_classes=num_classes)
    return model

if __name__ == '__main__':
    model = vit_base_patch16_224()
    model.load_state_dict(torch.load('vit_base_patch16_224.pth'))
    print(model)
    img = torch.randn(1,3,224,224)
    output = model(img)
    print(output.size())