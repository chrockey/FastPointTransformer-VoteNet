import os
import gc
import sys
from omegaconf import DictConfig, OmegaConf
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch_geometric.data import Batch
from MinkowskiEngine import (
    MinkowskiSPMMFunction,
    MinkowskiSPMMAverageFunction,
    MinkowskiDirectMaxPoolingFunction,
    SparseTensor,
    TensorField,
    KernelGenerator
) 

from torch_points3d.applications.modelfactory import ModelFactory
from torch_points3d.modules.MinkowskiEngine.api_modules import *
from torch_points3d.core.base_conv.message_passing import *
from torch_points3d.core.base_conv.partial_dense import *
from torch_points3d.models.base_architectures.unet import UnwrappedUnetBasedModel
from torch_points3d.core.common_modules.base_modules import MLP

from .utils import extract_output_nc
import cuda_sparse_ops


CUR_FILE = os.path.realpath(__file__)
DIR_PATH = os.path.dirname(os.path.realpath(__file__))

log = logging.getLogger(__name__)


# FPT common modules ------------------------------------------

@torch.no_grad()
def downsample_points(points, tensor_map, field_map, size):
    down_points = MinkowskiSPMMAverageFunction().apply(
        tensor_map, field_map, size, points
    ) # rows == inverse_map
    _, count = torch.unique(tensor_map, return_counts=True)
    
    return down_points, count.unsqueeze_(1).type_as(down_points)


@torch.no_grad()
def stride_centroids(points, count, rows, cols, size):
    stride_centroids = MinkowskiSPMMFunction().apply(
        rows, cols, count, size, points
    )
    ones = torch.ones(size[1], dtype=points.dtype, device=points.device)
    stride_count = MinkowskiSPMMFunction().apply(
        rows, cols, ones, size, count
    )
    
    return torch.true_divide(stride_centroids, stride_count), stride_count


def downsample_embeddings(embeddings, inverse_map, size, mode="avg"):
    assert len(embeddings) == size[1]
    
    if mode == "max":
        in_map = torch.arange(size[1], dtype=inverse_map.dtype, device=inverse_map.device)
        down_embeddings = MinkowskiDirectMaxPoolingFunction().apply(
            in_map, inverse_map, embeddings, size[0]
        )
    elif mode == "avg":
        cols = torch.arange(size[1], dtype=inverse_map.dtype, device=inverse_map.device)
        down_embeddings = MinkowskiSPMMAverageFunction().apply(
            inverse_map, cols, size, embeddings
        )
    else:
        raise NotImplementedError
    
    return down_embeddings


# ---------------------------------------------------------------
# FPT cuda functions --------------------------------------------

class DotProduct(Function):
    @staticmethod
    def forward(ctx, query, pos_enc, out_F, kq_map):
        assert (
            query.is_contiguous()
            and pos_enc.is_contiguous()
            and out_F.is_contiguous()
        )
        ctx.m = kq_map.shape[1]
        _, ctx.h, ctx.c = query.shape
        ctx.kkk = pos_enc.shape[0]
        ctx.save_for_backward(query, pos_enc, kq_map)
        cuda_sparse_ops.dot_product_forward(
            ctx.m, ctx.h, ctx.kkk, ctx.c, query, pos_enc, out_F, kq_map
        )
        return out_F

    @staticmethod
    def backward(ctx, grad_out_F):
        query, pos_enc, kq_map = ctx.saved_tensors
        grad_query = torch.zeros_like(query)
        grad_pos = torch.zeros_like(pos_enc)
        cuda_sparse_ops.dot_product_backward(
            ctx.m, ctx.h, ctx.kkk, ctx.c, query, pos_enc, kq_map,
            grad_query, grad_pos, grad_out_F
        )
        return grad_query, grad_pos, None, None

dot_product_cuda = DotProduct.apply

class ScalarAttention(Function):
    @staticmethod
    def forward(ctx, weight, value, out_F, kq_indices):
        assert (
            weight.is_contiguous()
            and value.is_contiguous()
            and out_F.is_contiguous()
        )
        ctx.m = kq_indices.shape[1]
        _, ctx.h, ctx.c = value.shape
        ctx.save_for_backward(weight, value, kq_indices)
        cuda_sparse_ops.scalar_attention_forward(
            ctx.m, ctx.h, ctx.c, weight, value, out_F, kq_indices
        )
        return out_F

    @staticmethod
    def backward(ctx, grad_out_F):
        weight, value, kq_indices = ctx.saved_tensors
        grad_weight = torch.zeros_like(weight)
        grad_value = torch.zeros_like(value)
        cuda_sparse_ops.scalar_attention_backward(
            ctx.m, ctx.h, ctx.c, weight, value, kq_indices,
            grad_weight, grad_value, grad_out_F
        )
        return grad_weight, grad_value, None, None

scalar_attention_cuda = ScalarAttention.apply

# ---------------------------------------------------------------
# FPT building modules ------------------------------------------
class FastPointTransformerLayer(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels=None,
        kernel_size=3,
        stride=1,
        num_heads=8,
        bias=True,
        dimension=3
    ):
        out_channels = in_channels if out_channels is None else out_channels
        assert out_channels % num_heads == 0
        super(FastPointTransformerLayer, self).__init__()
        
        self.out_channels = out_channels
        self.attn_channels = out_channels // num_heads
        self.num_heads = num_heads
        
        self.to_query = nn.Sequential(
            ME.MinkowskiConvolution(in_channels, self.out_channels, kernel_size=1, stride=stride, bias=bias, dimension=dimension),
            ME.MinkowskiToFeature()
        )
        self.to_value = nn.Sequential(
            ME.MinkowskiConvolution(in_channels, self.out_channels, kernel_size=1, bias=bias, dimension=dimension),
            ME.MinkowskiToFeature()
        )
        self.to_out = nn.Linear(out_channels, out_channels, bias=bias)
        
        # just for infomation
        if kernel_size == 3 and stride == 2:
            logging.info("Recommend to use kernel size 5 instead of 3 for stride 2.")
        self.kernel_size = kernel_size
        self.kernel_generator = KernelGenerator(kernel_size=kernel_size, stride=stride, dimension=dimension)
        self.kernel_volume = self.kernel_generator.kernel_volume

        self.inter_pos_enc = nn.Parameter(torch.FloatTensor(self.kernel_volume, self.num_heads, self.attn_channels))
        self.intra_pos_mlp = nn.Sequential(
            nn.Linear(dimension, dimension, bias=False),
            nn.BatchNorm1d(dimension),
            nn.ReLU(inplace=True),
            nn.Linear(dimension, in_channels, bias=False),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels, in_channels)
        )
        nn.init.normal_(self.inter_pos_enc, 0, 1)
    
    @torch.no_grad()
    def key_query_map_from_kernel_map(self, kernel_map):
        kq_map = []
        for kernel_idx, in_out in kernel_map.items():
            in_out[0] = in_out[0] * self.kernel_volume + kernel_idx
            kq_map.append(in_out)
        kq_map = torch.cat(kq_map, -1)
        
        return kq_map
    
    @torch.no_grad()
    def key_query_indices_from_key_query_map(self, kq_map):
        kq_indices = kq_map.clone()
        kq_indices[0] = kq_indices[0] // self.kernel_volume

        return kq_indices
    
    @torch.no_grad()
    def get_kernel_map_and_out_key(self, stensor):
        cm = stensor.coordinate_manager
        in_key = stensor.coordinate_key
        out_key = cm.stride(in_key, self.kernel_generator.kernel_stride)
        region_type, region_offset, _ = self.kernel_generator.get_kernel(stensor.tensor_stride, False)
        kernel_map = cm.kernel_map(
            in_key,
            out_key,
            self.kernel_generator.kernel_stride,
            self.kernel_generator.kernel_size,
            self.kernel_generator.kernel_dilation,
            region_type=region_type,
            region_offset=region_offset,
        )
        
        return kernel_map, out_key

    def forward(self, stensor, points, kq_map=None):
        assert len(stensor) == len(points)
        
        # query and value with intra-voxel relative positional encodings
        intra_pos_enc = self.intra_pos_mlp(points)
        stensor = stensor + intra_pos_enc
        q = self.to_query(stensor)
        v = self.to_value(stensor)
        q = q.view(-1, self.num_heads, self.attn_channels).contiguous()
        v = v.view(-1, self.num_heads, self.attn_channels).contiguous()
        num_queries = len(q)
        
        # kernel map
        if kq_map is None:
            kernel_map, out_key = self.get_kernel_map_and_out_key(stensor)
            kq_map = self.key_query_map_from_kernel_map(kernel_map)
        else:
            cm = stensor.coordinate_manager
            out_key = cm.stride(stensor.coordinate_key, self.kernel_generator.kernel_stride)
        
        # dot-product similarity
        attn = stensor._F.new(kq_map.shape[1], self.num_heads).zero_()
        norm_q = F.normalize(q, p=2, dim=-1)
        norm_pos_enc = F.normalize(self.inter_pos_enc, p=2, dim=-1)
        attn = dot_product_cuda(norm_q, norm_pos_enc, attn, kq_map)
        
        # aggregation & projection
        kq_indices = self.key_query_indices_from_key_query_map(kq_map)
        out_F = stensor._F.new(num_queries, self.num_heads, self.attn_channels).zero_()
        out_F = scalar_attention_cuda(attn, v, out_F, kq_indices)
        out_F = self.to_out(out_F.view(-1, self.out_channels).contiguous())
        
        return ME.SparseTensor(out_F, coordinate_map_key=out_key, coordinate_manager=stensor.coordinate_manager), kq_map


class FastPointTransformerBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels=None, kernel_size=3, dimension=3):
        out_channels = in_channels if out_channels is None else out_channels
        assert out_channels == in_channels
        super(FastPointTransformerBasicBlock, self).__init__()

        self.layer1 = FastPointTransformerLayer(
            in_channels, out_channels, kernel_size=kernel_size, dimension=dimension
        )
        self.norm1 = ME.MinkowskiBatchNorm(out_channels)
        self.layer2 = FastPointTransformerLayer(
            out_channels, kernel_size=kernel_size, dimension=dimension
        )
        self.norm2 = ME.MinkowskiBatchNorm(out_channels)
        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, stensor, points, meta=None):
        out, meta = self.layer1(stensor, points, meta)
        out = self.norm1(out)
        out = self.relu(out)

        out, meta = self.layer2(out, points, meta)
        out = self.norm2(out)

        out += stensor
        out = self.relu(out)

        return out, meta


class StridedMaxPoolLayer(nn.Module):
    
    def __init__(self, kernel_size=2, stride=2, dimension=3):
        assert kernel_size == 2
        assert stride == 2
        super(StridedMaxPoolLayer, self).__init__()

        self.pool = ME.MinkowskiMaxPooling(
            kernel_size=kernel_size, stride=stride, dimension=dimension
        )
        
    def forward(self, stensor, points, count):
        assert len(stensor) == len(points)
        cm = stensor.coordinate_manager
        
        down_stensor = self.pool(stensor)
        cols, rows = cm.stride_map(stensor.coordinate_key, down_stensor.coordinate_key)
        size = torch.Size([len(down_stensor), len(stensor)])
        down_points, down_count = stride_centroids(points, count, rows, cols, size)
        
        return down_stensor, down_points, down_count
# ---------------------------------------------------------------

class FPTUNet(nn.Module):
    H_DIM = 32
    INIT_DIM = 32
    PLANES = (64, 128, 384, 640, 384, 256, 256, 256)

    def __init__(self):
        assert torch.cuda.is_available()
        super(FPTUNet, self).__init__()

        self.input_nc = 3
        self.output_nc = 128
        self.grid_size = 0.05
        self.network_initialization()
        self.weight_initialization()
        self.device = torch.device("cuda:0")

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, ME.MinkowskiLinear):
                nn.init.xavier_normal_(m.linear.weight)
                if m.linear.bias is not None:
                    nn.init.constant_(m.linear.bias, 0)
            elif isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(
                    m.kernel,
                    mode="fan_out",
                    nonlinearity="relu"
                )

    def network_initialization(self):
        self.h_mlp = nn.Sequential(
            nn.Linear(self.input_nc, self.H_DIM, bias=False),
            nn.BatchNorm1d(self.H_DIM),
            nn.Tanh(),
            nn.Linear(self.H_DIM, self.H_DIM, bias=False),
            nn.BatchNorm1d(self.H_DIM),
            nn.Tanh()
        )
        self.attn0p1 = FastPointTransformerLayer(
            self.input_nc + self.H_DIM, self.INIT_DIM
        )
        self.bn0 = ME.MinkowskiBatchNorm(self.INIT_DIM)

        self.attn1p1 = FastPointTransformerLayer(self.INIT_DIM, self.PLANES[0])
        self.bn1 = ME.MinkowskiBatchNorm(self.PLANES[0])
        self.pool1p1s2 = StridedMaxPoolLayer()
        self.block1 = FastPointTransformerBasicBlock(self.PLANES[0])

        self.attn2p2 = FastPointTransformerLayer(self.PLANES[0], self.PLANES[1])
        self.bn2 = ME.MinkowskiBatchNorm(self.PLANES[1])
        self.pool2p2s2 = StridedMaxPoolLayer()
        self.block2 = FastPointTransformerBasicBlock(self.PLANES[1])

        self.attn3p4 = FastPointTransformerLayer(self.PLANES[1], self.PLANES[2])
        self.bn3 = ME.MinkowskiBatchNorm(self.PLANES[2])
        self.pool3p4s2 = StridedMaxPoolLayer()
        self.block3 = FastPointTransformerBasicBlock(self.PLANES[2])

        self.attn4p8 = FastPointTransformerLayer(self.PLANES[2], self.PLANES[3])
        self.bn4 = ME.MinkowskiBatchNorm(self.PLANES[3])
        self.pool4p8s2 = StridedMaxPoolLayer()
        self.block4 = FastPointTransformerBasicBlock(self.PLANES[3])

        self.attn5p8 = FastPointTransformerLayer(
            self.PLANES[3] + self.PLANES[3], self.PLANES[4]
        )
        self.bn5 = ME.MinkowskiBatchNorm(self.PLANES[4])
        self.block5 = FastPointTransformerBasicBlock(self.PLANES[4])

        self.attn6p4 = FastPointTransformerLayer(
            self.PLANES[4] + self.PLANES[2], self.PLANES[5]
        )
        self.bn6 = ME.MinkowskiBatchNorm(self.PLANES[5])
        self.block6 = FastPointTransformerBasicBlock(self.PLANES[5])

        self.attn7p2 = FastPointTransformerLayer(
            self.PLANES[5] + self.PLANES[1], self.PLANES[6]
        )
        self.bn7 = ME.MinkowskiBatchNorm(self.PLANES[6])
        self.block7 = FastPointTransformerBasicBlock(self.PLANES[6])

        self.attn8p1 = FastPointTransformerLayer(
            self.PLANES[6] + self.PLANES[0], self.PLANES[7]
        )
        self.bn8 = ME.MinkowskiBatchNorm(self.PLANES[7])
        self.block8 = FastPointTransformerBasicBlock(self.PLANES[7])

        self.final = nn.Sequential(
            nn.Linear(self.PLANES[7] + self.H_DIM, self.PLANES[7], bias=False),
            nn.BatchNorm1d(self.PLANES[7]),
            nn.ReLU(inplace=True),
            nn.Linear(self.PLANES[7], self.output_nc) # Voting Module dimension
        )
        self.relu = ME.MinkowskiReLU(inplace=True)
        self.pooltr = ME.MinkowskiPoolingTranspose(kernel_size=2, stride=2, dimension=3)

    def voxelize_with_centroids(self, x: ME.TensorField):
        cm = x.coordinate_manager
        points = x.C[:, 1:]
        
        out = x.sparse()
        size = torch.Size([len(out), len(x)])
        tensor_map, field_map = cm.field_to_sparse_map(x.coordinate_key, out.coordinate_key)
        points_p1, count_p1 = downsample_points(points, tensor_map, field_map, size)
        norm_points = self.normalize_points(points, points_p1, tensor_map)
        
        h_embs = self.h_mlp(norm_points)
        down_h_embs = downsample_embeddings(h_embs, tensor_map, size, mode="avg")
        out = ME.SparseTensor(
            features=torch.cat([out.F, down_h_embs], dim=1),
            coordinate_map_key=out.coordinate_key,
            coordinate_manager=cm
        )
        
        norm_points_p1 = self.normalize_centroids(points_p1, out.C, out.tensor_stride[0])
        
        return out, norm_points_p1, points_p1, count_p1, h_embs

    @torch.no_grad()
    def normalize_centroids(self, down_points, coordinates, tensor_stride):
        norm_points = (down_points - coordinates[:, 1:]) / tensor_stride - 0.5

        return norm_points

    @torch.no_grad()
    def normalize_points(self, points, centroids, tensor_map):
        tensor_map = tensor_map if tensor_map.dtype == torch.int64 else tensor_map.long()
        norm_points = points - centroids[tensor_map]
        
        return norm_points

    def devoxelize_with_centroids(self, out: ME.SparseTensor, x: ME.TensorField, h_embs):
        out = self.final(torch.cat([out.slice(x).F, h_embs], dim=1))
        
        return out

    def forward(self, data):
        assert data.pos is not None
        gc.collect()
        torch.cuda.empty_cache()

        x = ME.TensorField(
            features=data.rgb - 0.5,
            coordinates=torch.cat([
                data.batch.unsqueeze(-1), data.pos / self.grid_size
            ], dim=1),
            quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
            device=self.device
        )

        out, norm_points_p1, points_p1, count_p1, h_embs = self.voxelize_with_centroids(x)
        out, kq_map_p1k3 = self.attn0p1(out, norm_points_p1)
        out = self.relu(self.bn0(out))
        out_p1 = self.relu(self.bn1(self.attn1p1(out, norm_points_p1, kq_map_p1k3)[0]))

        out, points_p2, count_p2 = self.pool1p1s2(out_p1, points_p1, count_p1)
        norm_points_p2 = self.normalize_centroids(points_p2, out.C, out.tensor_stride[0])
        out, kq_map_p2k3 = self.block1(out, norm_points_p2)
        out_p2 = self.relu(self.bn2(self.attn2p2(out, norm_points_p2, kq_map_p2k3)[0]))

        out, points_p4, count_p4 = self.pool2p2s2(out_p2, points_p2, count_p2)
        norm_points_p4 = self.normalize_centroids(points_p4, out.C, out.tensor_stride[0])
        out, kq_map_p4k3 = self.block2(out, norm_points_p4)
        out_p4 = self.relu(self.bn3(self.attn3p4(out, norm_points_p4, kq_map_p4k3)[0]))

        out, points_p8, count_p8 = self.pool3p4s2(out_p4, points_p4, count_p4)
        norm_points_p8 = self.normalize_centroids(points_p8, out.C, out.tensor_stride[0])
        out, kq_map_p8k3 = self.block3(out, norm_points_p8)
        out_p8 = self.relu(self.bn4(self.attn4p8(out, norm_points_p8, kq_map_p8k3)[0]))

        out, points_p16, _ = self.pool4p8s2(out_p8, points_p8, count_p8)
        norm_points_p16 = self.normalize_centroids(points_p16, out.C, out.tensor_stride[0])
        out, kq_map_p16k3 = self.block4(out, norm_points_p16)

        out = self.pooltr(out)
        out = ME.cat(out, out_p8)
        out = self.relu(self.bn5(self.attn5p8(out, norm_points_p8, kq_map_p8k3)[0]))
        out = self.block5(out, norm_points_p8, kq_map_p8k3)[0]

        out = self.pooltr(out)
        out = ME.cat(out, out_p4)
        out = self.relu(self.bn6(self.attn6p4(out, norm_points_p4, kq_map_p4k3)[0]))
        out = self.block6(out, norm_points_p4, kq_map_p4k3)[0]

        out = self.pooltr(out)
        out = ME.cat(out, out_p2)
        out = self.relu(self.bn7(self.attn7p2(out, norm_points_p2, kq_map_p2k3)[0]))
        out = self.block7(out, norm_points_p2, kq_map_p2k3)[0]

        out = self.pooltr(out)
        out = ME.cat(out, out_p1)
        out = self.relu(self.bn8(self.attn8p1(out, norm_points_p1, kq_map_p1k3)[0]))
        out = self.block8(out, norm_points_p1, kq_map_p1k3)[0]

        out = self.devoxelize_with_centroids(out, x, h_embs)

        out = Batch(x=out, pos=data.pos.to(self.device), batch=data.batch)

        return out


def FPT(
    architecture: str = None, input_nc: int = None, num_layers: int = None, config: DictConfig = None, *args, **kwargs
):
    assert architecture == "unet"
    return FPTUNet()
