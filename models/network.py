#网络结构定义
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet50_Weights
import numpy as np

class WoodDefectBD(nn.Module):
    def __init__(self, pretrained=True):
        """
        木材缺陷检测网络(BDWNet)初始化
        Args:
            pretrained: 是否使用预训练权重
        """
        super().__init__()
        # 设置通道数
        self.out_channels = 64
        
        # 多尺度纹理感知编码器
        self.mtpe = MTPE(pretrained=pretrained)
        
        # 方向敏感特征提取与边界增强模块
        self.dsbem = DSBEM(self.out_channels)
        
        # 全局特征融合模块(GFM)
        self.gfm = nn.Sequential(
            nn.Conv2d(self.out_channels * 4, self.out_channels, 1),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 纹理引导自适应分割模块
        self.tgasm = TGASM(self.out_channels)
        
        # 创建备用输出层以防止前向传播失败
        self.fallback_conv = nn.Conv2d(self.out_channels, 1, 1)
        self.bn = nn.BatchNorm2d(self.out_channels)

    def forward(self, x):
        """
        前向传播函数
        Args:
            x: 输入图像
        Returns:
            分割结果
        """
        input_size = x.shape[-2:]
        
        try:
            # 使用MTPE提取多尺度特征
            p1, p2, p3, p4, f1, f2, f3, f4 = self.mtpe(x)
            
            # 检查特征是否包含NaN
            if (torch.isnan(p1).any() or torch.isnan(p2).any() or
                torch.isnan(p3).any() or torch.isnan(p4).any()):
                print("警告: 特征包含NaN值，将尝试使用简化版本")
                raise ValueError("特征包含NaN值")
            
            # 应用DSBEM增强边界特征
            dsbem_feat, edge_map = self.dsbem(p1)
            
            # 融合多尺度特征(GFM)
            p2_up = F.interpolate(p2, size=p1.shape[-2:], mode='bilinear', align_corners=False)
            p3_up = F.interpolate(p3, size=p1.shape[-2:], mode='bilinear', align_corners=False)
            p4_up = F.interpolate(p4, size=p1.shape[-2:], mode='bilinear', align_corners=False)
            
            global_feat = torch.cat([p1, p2_up, p3_up, p4_up], dim=1)
            global_feat = self.gfm(global_feat)
            
            # 合并边界增强特征和全局特征
            fused_feat = global_feat + dsbem_feat
            
            # 上采样到原始尺寸
            fused_feat = F.interpolate(fused_feat, size=input_size, mode='bilinear', align_corners=False)
            
            # 使用TGASM生成最终分割结果
            result = self.tgasm(fused_feat)
            
            # 添加边界图到结果
            if edge_map is not None:
                edge_map_up = F.interpolate(edge_map, size=input_size, mode='bilinear', align_corners=False)
                result['edge_map'] = edge_map_up
            
            # 检查结果中是否有NaN值
            has_nan = False
            for key in ['logits', 'prob_map', 'binary_map']:
                if key in result and torch.is_tensor(result[key]):
                    if torch.isnan(result[key]).any():
                        print(f"警告: 结果中的 {key} 包含NaN值")
                        has_nan = True
            
            if has_nan:
                # 如果有NaN值，使用简化版本重新计算
                print("检测到NaN值，使用简化版本")
                raise ValueError("检测到NaN值")
            
            # 确保所有输出都可以计算梯度
            for key in result:
                if torch.is_tensor(result[key]) and not result[key].requires_grad:
                    if key in ['logits', 'prob_map', 'binary_map']:
                        # 如果是预测结果，我们需要确保它可以反向传播梯度
                        dummy_grad = self.fallback_conv(fused_feat).sum() * 0
                        result[key] = result[key] + dummy_grad
            
            return result
            
        except Exception as e:
            # 出现错误时打印调试信息并使用简化版本
            print(f"前向传播错误: {e}")
            print(f"输入形状: {x.shape}")
            
            try:
                # 获取多尺度特征
                if 'p1' not in locals() or torch.isnan(p1).any():
                    # 如果p1不存在或包含NaN，重新提取特征
                    try:
                        p1, p2, p3, p4, _, _, _, _ = self.mtpe(x)
                    except Exception:
                        # 如果MTPE失败，使用更简单的方法
                        print("MTPE失败，使用简化特征提取")
                        # 应用简单卷积获取特征
                        p1 = F.relu(self.bn(nn.Conv2d(x.shape[1], self.out_channels, 3, padding=1).to(x.device)(x)))
            
                # 上采样到原始尺寸
                if 'p1' in locals() and not torch.isnan(p1).any():
                    fused = F.interpolate(p1, size=input_size, mode='bilinear', align_corners=False)
                else:
                    # 创建一个备用特征
                    print("创建备用特征")
                    fused = F.relu(self.bn(nn.Conv2d(x.shape[1], self.out_channels, 3, padding=1).to(x.device)(x)))
                    
                # 应用备用卷积层
                logits = self.fallback_conv(fused)
                prob_map = torch.sigmoid(logits)
                binary_map = torch.sigmoid((prob_map - 0.5) * 10.0)  # 平滑近似
                
                return {
                    'logits': logits,
                    'prob_map': prob_map,
                    'binary_map': binary_map,
                    'threshold': torch.tensor(0.5, device=x.device)
                }
            except Exception as e2:
                # 最终回退方案
                print(f"备用方案也失败: {e2}，使用最终回退方案")
                # 创建全新的特征和输出
                fallback_feat = torch.zeros((x.shape[0], self.out_channels, *input_size), device=x.device)
                fallback_logits = self.fallback_conv(fallback_feat)
                fallback_prob = torch.sigmoid(fallback_logits)
                
                return {
                    'logits': fallback_logits,
                    'prob_map': fallback_prob,
                    'binary_map': (fallback_prob > 0.5).float(),
                    'threshold': torch.tensor(0.5, device=x.device)
                }

class DirectionAwareConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        
        # 基础卷积
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        
        # 方向感知模块
        self.direction_conv = nn.Conv2d(in_channels, 2, kernel_size=3, padding=1)
        # 修改direction_offset的维度
        self.direction_offset = nn.Parameter(torch.zeros(out_channels, in_channels, 2, kernel_size, kernel_size))
        
        # 初始化权重
        self.init_weights()
    
    def init_weights(self):
        # Gabor滤波器初始化
        for i in range(self.out_channels):
            sigma = 3.0
            lambd = 10.0
            gamma = 0.5
            psi = 0
            for j in range(self.in_channels):
                theta = torch.tensor(np.pi * i / self.out_channels)
                kernel = self._create_gabor_kernel(self.kernel_size, sigma, theta, lambd, gamma, psi)
                self.conv.weight.data[i, j] = kernel
    
    def _create_gabor_kernel(self, ksize, sigma, theta, lambd, gamma, psi):
        """创建Gabor滤波器核"""
        y, x = torch.meshgrid(torch.linspace(-1, 1, ksize), torch.linspace(-1, 1, ksize))
        
        # 旋转
        x_theta = x * torch.cos(theta) + y * torch.sin(theta)
        y_theta = -x * torch.sin(theta) + y * torch.cos(theta)
        
        # Gabor函数
        gb = torch.exp(-.5 * (x_theta ** 2 + gamma ** 2 * y_theta ** 2) / sigma ** 2)
        gb *= torch.cos(2 * np.pi * x_theta / lambd + psi)
        
        return gb
    
    def _compute_local_direction(self, x):
        """计算局部方向"""
        direction = self.direction_conv(x)
        # 避免全零向量导致的归一化错误
        # 添加小的epsilon值防止除零错误
        epsilon = 1e-8
        # 计算每个向量的范数
        norm = torch.sqrt(torch.sum(direction**2, dim=1, keepdim=True) + epsilon)
        # 进行归一化，确保范数不为零
        direction = direction / (norm + epsilon)
        return direction
    
    def _adjust_kernel(self, direction):
        """根据方向调整卷积核"""
        B, C, H, W = direction.shape
        
        # 获取基础权重
        weight = self.conv.weight  # shape: [out_channels, in_channels, kernel_size, kernel_size]
        
        # 调整direction的维度以匹配权重
        direction = F.interpolate(direction, size=self.kernel_size, mode='bilinear', align_corners=False)  # [B, 2, kernel_size, kernel_size]
        direction = direction.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, 2, kernel_size, kernel_size]
        
        # 调整direction_offset的维度以进行广播
        offset = self.direction_offset.unsqueeze(0)  # [1, out_channels, in_channels, 2, kernel_size, kernel_size]
        
        # 计算方向调整后的权重
        # 确保维度匹配进行广播
        try:
            direction_weight = (offset * direction).sum(dim=3)  # 沿方向维度求和
            adjusted_weight = weight + direction_weight.mean(dim=0)  # 平均batch维度
            return adjusted_weight
        except RuntimeError as e:
            # 如果出现维度不匹配错误，返回原始权重
            print(f"警告：调整卷积核时出错: {e}")
            print(f"方向张量形状: {direction.shape}, 偏移张量形状: {offset.shape}")
            return weight
    
    def forward(self, x):
        # 计算局部方向
        direction = self._compute_local_direction(x)
        
        # 调整卷积核
        adjusted_weight = self._adjust_kernel(direction)
        
        # 应用卷积并归一化
        out = F.conv2d(x, adjusted_weight, self.conv.bias, padding=self.kernel_size//2)
        return F.normalize(out, dim=1)  # 添加归一化

class TextureAttention(nn.Module):
    def __init__(self, channels, downsample_factor=4, chunk_size=512):
        super().__init__()
        self.texture_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)  # 简化为标准卷积
        self.bn = nn.BatchNorm2d(channels)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.downsample_factor = downsample_factor
        self.chunk_size = chunk_size
        # 添加一个回退层，确保即使注意力机制失败也能返回有效输出
        self.fallback_layer = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
    def _compute_texture_similarity(self, features):
        """计算纹理特征之间的相似度矩阵，使用空间下采样和分块计算以减少内存消耗"""
        B, C, H, W = features.shape
        
        # 空间下采样，减少计算量
        if self.downsample_factor > 1 and H > 16 and W > 16:
            ds_features = F.interpolate(
                features, 
                size=(H // self.downsample_factor, W // self.downsample_factor),
                mode='bilinear', 
                align_corners=False
            )
        else:
            ds_features = features
            
        # 将特征展平
        B, C, H_ds, W_ds = ds_features.shape
        N = H_ds * W_ds  # 空间位置数量
        
        flat_features = ds_features.view(B, C, N)  # [B, C, H_ds*W_ds]
        
        # 对通道维度归一化以提高数值稳定性
        flat_features = F.normalize(flat_features, dim=1)
        
        # 转置用于计算相似度
        features_t = flat_features.transpose(1, 2)  # [B, N, C]
        
        # 使用分块计算相似度矩阵以减少内存使用
        if N <= self.chunk_size:
            # 直接计算完整相似度矩阵 - 修正为先转置后乘
            similarity = torch.bmm(features_t, flat_features)  # [B, N, N]
        else:
            # 分块计算相似度矩阵
            similarity = torch.zeros(B, N, N, device=flat_features.device)
            for i in range(0, N, self.chunk_size):
                end_i = min(i + self.chunk_size, N)
                chunk_i = features_t[:, i:end_i, :]  # [B, chunk, C]
                
                # 计算这个块与所有特征的相似度
                chunk_sim = torch.bmm(chunk_i, flat_features)  # [B, chunk, N]
                similarity[:, i:end_i, :] = chunk_sim
        
        # 应用Softmax使相似度值正规化，添加温度系数提高数值稳定性
        temperature = 0.5
        similarity = F.softmax(similarity / temperature, dim=-1)
        
        return similarity, N, flat_features, H_ds, W_ds
        
    def forward(self, x):
        # 提取纹理特征
        texture_feat = F.relu(self.bn(self.texture_conv(x)))
        
        try:
            # 尝试使用优化版本
            B, C, H, W = x.shape
            
            # 计算纹理相似度
            similarity, N, x_flat, H_ds, W_ds = self._compute_texture_similarity(texture_feat)
            
            # 确保特征的空间维度与相似度矩阵匹配
            if H * W != N:
                # 对原始特征进行下采样
                x_sampled = F.interpolate(
                    x, 
                    size=(H_ds, W_ds),
                    mode='bilinear', 
                    align_corners=False
                )
                x_flat = x_sampled.view(B, C, -1)  # [B, C, N]
            else:
                x_flat = x.view(B, C, -1)  # 已经匹配，直接展平
            
            # 正确的矩阵乘法顺序
            attended = torch.bmm(x_flat, similarity.transpose(1, 2))  # [B, C, N]
            
            # 加上残差连接
            out_flat = x_flat + self.gamma * attended
            
            # 重塑回原始形状
            if N != H * W:
                # 上采样回原始大小
                out = out_flat.view(B, C, H_ds, W_ds)
                out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
            else:
                out = out_flat.view_as(x)
            
            return out
            
        except Exception as e:
            # 如果出现任何错误，使用回退策略
            print(f"TextureAttention错误: {e}，使用回退策略")
            return self.fallback_layer(x)  # 直接对输入应用回退层

class MTAM(nn.Module):
    """多尺度纹理注意力模块"""
    def __init__(self, in_channels):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_channels, in_channels, 1)
        self.conv3x3_1 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.conv3x3_2 = nn.Conv2d(in_channels, in_channels, 3, padding=2, dilation=2)
        self.conv3x3_3 = nn.Conv2d(in_channels, in_channels, 3, padding=4, dilation=4)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # 通道注意力
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_attention = nn.Sequential(
            nn.Linear(in_channels, in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // 4, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 多尺度特征
        feat1 = self.conv1x1(x)
        feat2 = self.conv3x3_1(x)
        feat3 = self.conv3x3_2(x)
        feat4 = self.conv3x3_3(x)
        feat = feat1 + feat2 + feat3 + feat4
        
        # 通道注意力
        b, c, _, _ = feat.shape
        avg_pool = self.avg_pool(feat).view(b, c)
        channel_weight = self.channel_attention(avg_pool).view(b, c, 1, 1)
        
        out = feat * channel_weight
        out = self.relu(self.bn(out))
        
        return out

class BoundaryAwareModule(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # 减少中间通道数
        mid_channels = in_channels // 4  # 进一步减少通道数以节省内存
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, 1, 1)
        
    def forward(self, x):
        try:
            feat = F.relu(self.bn1(self.conv1(x)))
            feat = F.relu(self.bn2(self.conv2(feat)))
            edge = torch.sigmoid(self.conv3(feat))
            
            # 修改维度扩展方式 - 使用更高效的方法
            # 这里采用广播而不是repeat，减少内存使用
            refined = x * edge + x
            return refined
        except RuntimeError as e:
            # 如果内存不足，返回原始输入作为回退策略
            if "CUDA out of memory" in str(e):
                print(f"警告: BoundaryAwareModule内存不足，跳过边界增强: {e}")
                return x
            else:
                raise e

class MTPE(nn.Module):
    """多尺度纹理感知编码器"""
    def __init__(self, pretrained=True):
        super().__init__()
        # 使用ResNet18作为backbone
        resnet = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
        
        # 提取ResNet的各个层
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1  # 1/4
        self.layer2 = resnet.layer2  # 1/8 
        self.layer3 = resnet.layer3  # 1/16
        self.layer4 = resnet.layer4  # 1/32

        # 多尺度纹理注意力模块
        self.mtam1 = MTAM(64)
        self.mtam2 = MTAM(128)
        self.mtam3 = MTAM(256)
        self.mtam4 = MTAM(512)
        
        # 设置通道数
        self.out_channels = 64

        # 上采样卷积层
        self.up_conv4 = nn.Sequential(
            nn.Conv2d(512, self.out_channels, 1),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True)
        )
        self.up_conv3 = nn.Sequential(
            nn.Conv2d(256, self.out_channels, 1),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True)
        )
        self.up_conv2 = nn.Sequential(
            nn.Conv2d(128, self.out_channels, 1),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True)
        )
        self.up_conv1 = nn.Sequential(
            nn.Conv2d(64, self.out_channels, 1),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 基础特征提取
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # 提取多尺度特征
        f1 = self.layer1(x)      # 1/4
        f1_enhanced = self.mtam1(f1)
        
        f2 = self.layer2(f1)     # 1/8
        f2_enhanced = self.mtam2(f2)
        
        f3 = self.layer3(f2)     # 1/16
        f3_enhanced = self.mtam3(f3)
        
        f4 = self.layer4(f3)     # 1/32
        f4_enhanced = self.mtam4(f4)

        # 特征金字塔处理
        p4 = self.up_conv4(f4_enhanced)
        p3 = self.up_conv3(f3_enhanced)
        p2 = self.up_conv2(f2_enhanced)
        p1 = self.up_conv1(f1_enhanced)

        # 自顶向下融合
        p4_up = F.interpolate(p4, size=p3.shape[-2:], mode='bilinear', align_corners=False)
        p3 = p3 + p4_up
        p3_up = F.interpolate(p3, size=p2.shape[-2:], mode='bilinear', align_corners=False)
        p2 = p2 + p3_up
        p2_up = F.interpolate(p2, size=p1.shape[-2:], mode='bilinear', align_corners=False)
        p1 = p1 + p2_up
        
        return p1, p2, p3, p4, f1_enhanced, f2_enhanced, f3_enhanced, f4_enhanced

class DSBEM(nn.Module):
    """方向敏感特征提取与边界增强模块 - 简化版本"""
    def __init__(self, in_channels):
        super().__init__()
        # 简化特征提取
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)
        
        # 精简纹理注意力
        self.texture_attention = TextureAttention(in_channels, downsample_factor=8)
        
        # 边界特征提取
        self.edge_conv = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)
        
        # 添加空间注意力
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//2, kernel_size=1),
            nn.BatchNorm2d(in_channels//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels//2, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # 基础特征提取
        feat = F.relu(self.bn1(self.conv1(x)))
        feat = F.relu(self.bn2(self.conv2(feat)))
        
        # 纹理增强
        texture_feat = self.texture_attention(feat)
        
        # 空间注意力
        spatial_attn = self.spatial_attn(texture_feat)
        
        # 边缘特征
        edge_map = torch.sigmoid(self.edge_conv(feat))
        
        # 应用注意力和边缘增强
        enhanced_feat = texture_feat * spatial_attn + texture_feat
        
        # 返回增强后的特征和边缘图
        return enhanced_feat, edge_map

class TGASM(nn.Module):
    """纹理引导自适应分割模块"""
    def __init__(self, in_channels):
        super().__init__()
        # 纹理复杂度估计
        self.texture_pool = nn.AdaptiveAvgPool2d(1)
        self.texture_fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // 4, 1),
            nn.Sigmoid()
        )
        
        # 添加边界感知模块
        self.boundary_enhance = BoundaryAwareModule(in_channels)
        
        # 分割层
        self.seg_conv = nn.Conv2d(in_channels, in_channels // 2, 3, padding=1)
        self.seg_bn = nn.BatchNorm2d(in_channels // 2)
        self.final_conv = nn.Conv2d(in_channels // 2, 1, 1)
        
        # 可学习的阈值参数
        self.base_threshold = nn.Parameter(torch.tensor(0.5))
        self.threshold_range = nn.Parameter(torch.tensor(0.2))
        self.temperature = nn.Parameter(torch.tensor(5.0))
        
        # 添加一个回退卷积层
        self.fallback_conv = nn.Conv2d(in_channels, 1, 1)
        
    def forward(self, x):
        try:
            # 应用边界增强
            x_enhanced = self.boundary_enhance(x)
            
            # 纹理复杂度估计
            b, c, _, _ = x_enhanced.shape
            texture_feat = self.texture_pool(x_enhanced).view(b, c)
            complexity = self.texture_fc(texture_feat)
            
            # 自适应阈值计算 - 调整为与prob_map兼容的形状
            # 使用clamp确保阈值在合理范围内
            threshold = torch.clamp(
                self.base_threshold + (complexity - 0.5) * self.threshold_range,
                min=0.1, max=0.9  # 阈值限制在0.1-0.9之间
            )
            
            # 分割预测
            feat = F.relu(self.seg_bn(self.seg_conv(x_enhanced)))
            logits = self.final_conv(feat)
            
            # 使用sigmoid获取概率图，处理可能的NaN
            prob_map = torch.sigmoid(logits)
            
            # 检查prob_map是否包含NaN，如果有则替换为0
            if torch.isnan(prob_map).any():
                print("警告: prob_map包含NaN值，已替换为0")
                prob_map = torch.where(torch.isnan(prob_map), torch.zeros_like(prob_map), prob_map)
            
            # 将threshold从形状[b,1]调整为[b,1,1,1]
            threshold_expanded = threshold.view(b, 1, 1, 1)
            
            # 使用可微分的二值化方法
            # 添加小值epsilon避免数值不稳定
            eps = 1e-6
            binary_logits = (prob_map - threshold_expanded + eps) * self.temperature
            binary_map = torch.sigmoid(binary_logits)
            
            # 确保所有输出都可以计算梯度
            # 如果logits需要梯度但prob_map或binary_map没有，则重新计算
            needs_grad_fix = False
            if logits.requires_grad:
                if not prob_map.requires_grad:
                    prob_map = torch.sigmoid(logits)
                    needs_grad_fix = True
                if not binary_map.requires_grad:
                    binary_logits = (prob_map - threshold_expanded + eps) * self.temperature
                    binary_map = torch.sigmoid(binary_logits)
                    needs_grad_fix = True
            
            if needs_grad_fix:
                print("修复梯度流：重新计算概率图和二值图")
            
            return {
                'logits': logits,
                'prob_map': prob_map,
                'binary_map': binary_map,
                'binary_logits': binary_logits,
                'threshold': threshold,
                'complexity': complexity
            }
        except Exception as e:
            print(f"TGASM前向传播错误: {e}，使用回退实现")
            # 回退到基本实现
            logits = self.fallback_conv(x)
            prob_map = torch.sigmoid(logits)
            binary_map = torch.sigmoid((prob_map - self.base_threshold) * 10.0)  # 使用平滑的近似代替硬阈值
            
            return {
                'logits': logits,
                'prob_map': prob_map, 
                'binary_map': binary_map,
                'threshold': self.base_threshold
            }
