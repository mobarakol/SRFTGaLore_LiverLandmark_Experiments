import math
import warnings
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from models.resnet import ResNet18, ResNet34, ResNet50
from models.context_modules import get_context_module
from models.model_utils import ConvBNAct, Swish, Hswish, SqueezeAndExcitation
from models.decoder_v1 import Decoder
from segment_anything import sam_model_registry


class Edge_Prototypes(nn.Module):
    def __init__(self, num_classes=3, feat_dim=256):
        super(Edge_Prototypes, self).__init__()
        self.class_embeddings = nn.Embedding(num_classes, feat_dim)

    def forward(self):
        return self.class_embeddings.weight


class Attention(nn.Module):

    def __init__(
            self,
            embedding_dim: int,
            num_heads: int,
            downsample_rate: int = 1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)

        # Get output
        out = attn @ v
        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out


class DPE(nn.Module):

    def __init__(self):
        super(DPE, self).__init__()

        self.FFN = nn.Sequential(
            nn.Conv2d(256 * 3, 128 * 2, kernel_size=1),
            nn.BatchNorm2d(128 * 2),
            nn.ReLU(),
            nn.Conv2d(128 * 2, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.key_projections = nn.ModuleList()
        self.edge_semantics = nn.ModuleList()
        self.prompt_feat_projections = nn.ModuleList()
        self.fusion_forwards = nn.ModuleList()

        for i in range(3):
            self.key_projections.append(nn.Linear(256, 256))
            self.fusion_forwards.append(nn.Sequential(
                nn.Conv2d(256 + 256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(),
            ))
            self.edge_semantics.append(nn.Sequential(
                nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1, bias=False),
                nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=False),
                nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1, bias=False)
            ))
            self.prompt_feat_projections.append(nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(128, 256, kernel_size=1)
            ))

    def forward(self, features, sam_feature, prototypes):
        fused_feat = None
        edge_map = None

        features = features.reshape(features.shape[0], -1, features.shape[2] * features.shape[3])  # 2, 128, 256
        prior_prototypes = prototypes.unsqueeze(-1)
        sam_feat = F.interpolate(sam_feature, size=16, mode='bilinear').reshape(sam_feature.shape[0], -1,
                                                                                16 * 16)  # 2, 256, 256

        for i in range(prior_prototypes.shape[0]):
            specific_prototype = prior_prototypes[i]
            specific_prototype = torch.stack([specific_prototype for _ in range(sam_feature.shape[0])], dim=0)
            sim = torch.matmul(sam_feat, specific_prototype).squeeze(-1)
            sim = torch.stack([sim for _ in range(sam_feat.shape[1])], dim=1)
            specific_sam_feat = sam_feat + sam_feat * sim  # 2, 256, (16*16)
            specific_sam_feat = self.prompt_feat_projections[i](
                specific_sam_feat.reshape(specific_sam_feat.shape[0], -1, 16, 16)).reshape(specific_sam_feat.shape[0],
                                                                                           -1, 256)  # 2, 256, (16*16)

            # dot production
            # print('features shape:', features.shape)  # [4, 256, 256]
            # print('specific_sam_feat shape:', specific_sam_feat.shape)  # [4, 256, 256]
            fused_feature = self.fusion_forwards[i](
                torch.cat((features, specific_sam_feat), dim=1).reshape(specific_sam_feat.shape[0], -1, 16, 16))

            edge_attn = 1 - F.sigmoid(fused_feature)
            # print('edge_attn shape:', edge_attn.shape)  # [4, 128, 16, 16]
            # print('features shape:', features.shape)  # [4, 256, 256]
            # print('shape of :', (edge_attn * features.reshape(features.shape[0], -1, 16, 16)).shape)  # [4, 256, 256]
            map = self.edge_semantics[i]((edge_attn * features.reshape(features.shape[0], -1, 16, 16)))
            if fused_feat is None:
                fused_feat = fused_feature
                edge_map = map
            else:
                fused_feat = torch.cat((fused_feature, fused_feat), dim=1)
                edge_map = torch.cat((edge_map, map), dim=1)

        out = self.FFN(fused_feat)

        return out, edge_map, specific_sam_feat.reshape(specific_sam_feat.shape[0], -1, 16, 16)



class SR_CrossAttention(nn.Module):
    def __init__(self, embed_dim1=256, embed_dim2=256, num_heads=2, reduction_ratio=4):
        super().__init__()
        self.reduction = nn.AvgPool2d(kernel_size=reduction_ratio)
        self.proj_mat2 = nn.Linear(embed_dim2, embed_dim1)
        self.cross_attn = nn.MultiheadAttention(embed_dim1, num_heads)

    def forward(self, mat1, mat2):
        B, C1, H, W = mat1.shape
        B, C2, _, _ = mat2.shape
        # Reduce spatial size of mat2
        mat2_reduced = self.reduction(mat2)  # [B, 256, H//r, W//r]
        Hr, Wr = mat2_reduced.shape[2:]
        x1 = mat1.view(B, C1, -1).permute(2, 0, 1)           # [HW, B, 512]
        x2 = mat2_reduced.view(B, C2, -1).permute(2, 0, 1)    # [H'W', B, 256]
        x2_proj = self.proj_mat2(x2)                          # [H'W', B, 512]
        attn_out, _ = self.cross_attn(query=x1, key=x2_proj, value=x2_proj)
        return attn_out.permute(1, 2, 0).contiguous().view(B, C1, H, W)



# Assume sam_encoder is your image encoder, and 'image' is your input tensor

# def get_last_n_block_outputs(sam_encoder, image, n=4):
#     x = sam_encoder.patch_embed(image)  # Patch embedding
#     if sam_encoder.pos_embed is not None:
#         x = x + sam_encoder.pos_embed
#     x = sam_encoder.pos_drop(x)
#     hidden_states = []
#     for i, blk in enumerate(sam_encoder.blocks):
#         x = blk(x)
#         hidden_states.append(x)
#     # hidden_states is a list of tensors: one per block (length: num_blocks)
#     return hidden_states[-n:]  # last n feature maps

def get_last_n_block_outputs(sam_encoder, image, n=4):
    # Step 1: Patch Embedding (SAM: uses patch_embed, always present)
    x = sam_encoder.patch_embed(image)  # [B, num_patches, embed_dim]

    # Step 2: Positional Embedding (may be present)
    if hasattr(sam_encoder, "pos_embed") and sam_encoder.pos_embed is not None:
        x = x + sam_encoder.pos_embed

    # Step 3: (NO pos_drop in SAM! skip...)

    # Step 4: Forward through all transformer blocks, saving outputs
    hidden_states = []
    for blk in sam_encoder.blocks:
        x = blk(x)
        hidden_states.append(x.permute( 0, 3, 1, 2)) 
    out, skip1, skip2, skip3 = sam_encoder.neck(hidden_states[-1]),  sam_encoder.neck(hidden_states[-2]), sam_encoder.neck(hidden_states[-3]), sam_encoder.neck(hidden_states[-4])
    # Step 5: Optionally, handle norm if needed (often SAM does norm after all blocks)
    # You can get pre-norm features from last N blocks.
    return out, skip1, skip2, skip3

# Usage:

# Each element in 'last4' is a tensor: [batch, num_patches, embed_dim]


class D2GPLand(nn.Module):
    def __init__(self,
                 height=256,
                 width=480,
                 num_classes=4,
                 encoder='resnet34',
                 encoder_block='BasicBlock',
                 channels_decoder=None,  # default: [128, 128, 128]
                 pretrained_on_imagenet=True,
                 pretrained_dir='/results_nas/moko3016/'
                                'moko3016-efficient-rgbd-segmentation/'
                                'imagenet_pretraining',
                 activation='relu',
                 input_channels=3,
                 encoder_decoder_fusion='add',
                 context_module='ppm',
                 nr_decoder_blocks=None,  # default: [1, 1, 1]
                 weighting_in_encoder='None',
                 upsampling='bilinear'):
        super(D2GPLand, self).__init__()

        if channels_decoder is None:
            channels_decoder = [256, 256, 256]
        if nr_decoder_blocks is None:
            nr_decoder_blocks = [1, 1, 1]

        self.weighting_in_encoder = weighting_in_encoder

        if activation.lower() == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation.lower() in ['swish', 'silu']:
            self.activation = Swish()
        elif activation.lower() == 'hswish':
            self.activation = Hswish()
        else:
            raise NotImplementedError('Only relu, swish and hswish as '
                                      'activation function are supported so '
                                      'far. Got {}'.format(activation))
        sam = sam_model_registry["vit_b"](
            checkpoint="sam_path/sam_vit_b_01ec64.pth")

        self.sam_encoder = sam.image_encoder
        self.first_conv = ConvBNAct(1, 3, kernel_size=1,
                                    activation=self.activation)
        self.mid_conv = ConvBNAct(512 + 256, 512, kernel_size=1,
                                  activation=self.activation)
        self.mid_img_conv = ConvBNAct(512, 256, kernel_size=1,
                                      activation=self.activation)
        self.depth_encoder = ResNet34(
            block=encoder_block,
            pretrained_on_imagenet=pretrained_on_imagenet,
            pretrained_dir=pretrained_dir,
            activation=self.activation,
            input_channels=1
        )
        # encoder
        if encoder == 'resnet18':
            self.encoder = ResNet18(
                block=encoder_block,
                pretrained_on_imagenet=pretrained_on_imagenet,
                pretrained_dir=pretrained_dir,
                activation=self.activation,
                input_channels=input_channels
            )
        elif encoder == 'resnet34':
            self.encoder = ResNet34(
                block=encoder_block,
                pretrained_on_imagenet=pretrained_on_imagenet,
                pretrained_dir=pretrained_dir,
                activation=self.activation,
                input_channels=input_channels
            )
        elif encoder == 'resnet50':
            self.encoder = ResNet50(
                pretrained_on_imagenet=pretrained_on_imagenet,
                activation=self.activation,
                input_channels=input_channels
            )
        else:
            raise NotImplementedError('Only ResNets as encoder are supported '
                                      'so far. Got {}'.format(activation))

        # self.channels_decoder_in = self.encoder.down_32_channels_out
        self.channels_decoder_in = 256  

        if weighting_in_encoder == 'SE-add':
            self.se_layer0 = SqueezeAndExcitation(
                64, activation=self.activation)
            self.se_layer1 = SqueezeAndExcitation(
                self.encoder.down_4_channels_out,
                activation=self.activation)
            self.se_layer2 = SqueezeAndExcitation(
                self.encoder.down_8_channels_out,
                activation=self.activation)
            self.se_layer3 = SqueezeAndExcitation(
                self.encoder.down_16_channels_out,
                activation=self.activation)
            self.se_layer4 = SqueezeAndExcitation(
                self.encoder.down_32_channels_out,
                activation=self.activation)
        else:
            self.se_layer0 = nn.Identity()
            self.se_layer1 = nn.Identity()
            self.se_layer2 = nn.Identity()
            self.se_layer3 = nn.Identity()
            self.se_layer4 = nn.Identity()

        if encoder_decoder_fusion == 'add':
            layers_skip1 = list()
            if self.encoder.down_4_channels_out != channels_decoder[2]:
                layers_skip1.append(ConvBNAct(
                    self.encoder.down_4_channels_out,
                    channels_decoder[2],
                    kernel_size=1,
                    activation=self.activation))
            self.skip_layer1 = nn.Sequential(*layers_skip1)

            layers_skip2 = list()
            if self.encoder.down_8_channels_out != channels_decoder[1]:
                layers_skip2.append(ConvBNAct(
                    self.encoder.down_8_channels_out,
                    channels_decoder[1],
                    kernel_size=1,
                    activation=self.activation))
            self.skip_layer2 = nn.Sequential(*layers_skip2)

            layers_skip3 = list()
            if self.encoder.down_16_channels_out != channels_decoder[0]:
                layers_skip3.append(ConvBNAct(
                    self.encoder.down_16_channels_out,
                    channels_decoder[0],
                    kernel_size=1,
                    activation=self.activation))
            self.skip_layer3 = nn.Sequential(*layers_skip3)

        # context module
        if 'learned-3x3' in upsampling:
            warnings.warn('for the context module the learned upsampling is '
                          'not possible as the feature maps are not upscaled '
                          'by the factor 2. We will use nearest neighbor '
                          'instead.')
            upsampling_context_module = 'nearest'
        else:
            upsampling_context_module = upsampling
        self.context_module, channels_after_context_module = get_context_module(
            context_module,
            self.channels_decoder_in,
            channels_decoder[0],
            
            input_size=(height // 32, width // 32),
            activation=self.activation,
            upsampling_mode=upsampling_context_module)

        # decoder
        self.decoder = Decoder(
            channels_in=channels_after_context_module,
            channels_decoder=channels_decoder,
            activation=self.activation,
            nr_decoder_blocks=nr_decoder_blocks,
            encoder_decoder_fusion=encoder_decoder_fusion,
            upsampling_mode=upsampling,
            num_classes=num_classes
        )

        self.prompt_fusion_layer = DPE()
        self.fuse_model = SR_CrossAttention()  # try num_head = 2 and 4

    def forward(self, image, depth, prototypes):
        original_depth = F.interpolate(depth, size=(1024, 1024), mode='area') # [1, 1, 1024, 1024]

        original_depth = self.first_conv(original_depth) # [1, 3, 1024, 1024]

        original_depth = self.sam_encoder(original_depth) # [1, 256, 64, 64]
        # out = self.sam_encoder(image) # [1, 256, 64, 64]
        # print('SAM encoder: ', self.sam_encoder)
        out, skip1, skip2, skip3  = get_last_n_block_outputs(self.sam_encoder, image, n=4)# [4, 64, 64, 768]
        skip1 = F.interpolate(skip1, size=(256, 256), mode='bilinear', align_corners=False)
        skip2 = F.interpolate(skip2, size=(128, 128), mode='bilinear', align_corners=False)
        # skip3 = F.interpolate(skip3, size=(256, 256), mode='bilinear', align_corners=False)
        # out = self.encoder.forward_first_conv(image) # [1, 64, 512, 512]  
        # out = self.se_layer0(out) #[1, 64, 512, 512]  
        # out = F.max_pool2d(out, kernel_size=3, stride=2, padding=1) # [1, 64, 256, 256]

        # # block 1
        # out = self.encoder.forward_layer1(out) # [1, 64, 256, 256] 
        # out = self.se_layer1(out) # [1, 64, 256, 256]
        # skip1 = self.skip_layer1(out) # [1, 128, 256, 256]

        # # block 2
        # out = self.encoder.forward_layer2(out) # [1, 128, 128, 128]
        # out = self.se_layer2(out) # [1, 128, 128, 128]
        # skip2 = self.skip_layer2(out) # [1, 128, 128, 128]

        # # block 3
        # out = self.encoder.forward_layer3(out) # [1, 256, 64, 64]
        # out = self.se_layer3(out) # [1, 256, 64, 64]
        # skip3 = self.skip_layer3(out) # [1, 128, 64, 64]

        # # block 4
        # out = self.encoder.forward_layer4(out) # [1, 512, 32, 32]
        # out = self.se_layer4(out) # [1, 512, 32, 32] (R_rgb)

        # depth = F.interpolate(original_depth, size=(32, 32), mode='area') # [1, 256, 32, 32] (F_d)
        # depth = F.interpolate(original_depth, size=out.shape[2:], mode='bilinear', align_corners=False)

        # out = self.mid_conv(torch.cat((out, depth), 1)) # [1, 512, 32, 32]  (F_f)
        # Get fused output
        # print('out shape:', out.shape)  # [4, 256, 64, 64]
        # print('original_depth shape:', original_depth.shape) # [4, 256, 64, 64]
        out = self.fuse_model(out, original_depth)
        # print('fused out shape:', out.shape)  # [4, 256, 64, 64]

        out = self.context_module(out) # [1, 128, 32, 32] old
        # print('context module out shape:', out.shape)  # [4, 256, 64, 64] new

        out = F.interpolate(out, size=(16, 16), mode='area') # [1, 128, 16, 16] old
        # print('interpolated out shape:', out.shape) # [4, 256, 16, 16] new


        # DPE [1, 256, 64, 64] original_depth
        # print('prototype: ', prototypes.shape)  # [3, 256]
        fused_feat, edge_out, geometry_feat = self.prompt_fusion_layer(out, original_depth, prototypes) # fused_feat: [1, 128, 16, 16] (F_s,l,r) ??
        fused_feat = F.interpolate(fused_feat, size=(32, 32), mode='area') # [1, 128, 32, 32] (F_a)? 
        # print('fused_feat shape:', fused_feat.shape)  # 
        # print('skip1 shape:', skip1.shape)  # 
        # print('skip2 shape:', skip2.shape)  # 
        # print('skip3 shape:', skip3.shape)

        outs = [fused_feat, skip3, skip2, skip1]
        
        # outs = [fused_feat]
        outs, out_visual = self.decoder(enc_outs=outs) # outs: [4, 1024, 1024]
        # print('decoder outs shape:', outs.shape)
        outs = F.log_softmax(outs, dim=1) # [1, 1, 1024, 1024]

        return outs, original_depth, edge_out
