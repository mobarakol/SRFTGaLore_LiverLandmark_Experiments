# Replace SAM in v_14 with SAM2
import math
import warnings
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch import nn, Tensor
# from models.resnet import ResNet18, ResNet34, ResNet50
from models.context_modules import get_context_module
from models.model_utils import ConvBNAct, Swish, Hswish, SqueezeAndExcitation
from models.decoder_v15 import Decoder
from segment_anything import sam_model_registry

# DA2 imports
from transformers import AutoImageProcessor
from transformers import DepthAnythingForDepthEstimation

# SAM2 imports
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# DoRA imports
from peft import get_peft_model, LoraConfig, TaskType

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



def get_last_n_block_outputs(sam_encoder, image):
    outputs = sam_encoder.base_model.model(image)

    # 最後一層的 image features
    out = outputs["vision_features"]  # [B, C, H, W]

    # 多尺度特徵圖
    fpn = outputs["backbone_fpn"]  # 是 list，不是 dict


    skip3 = fpn[0]
    skip2 = fpn[1]
    skip1 = fpn[2]

    return out, skip1, skip2, skip3


class CrossAttentionFuse(nn.Module):
    def __init__(self, embed_dim1, embed_dim2, num_heads):
        super().__init__()
        self.embed_dim1 = embed_dim1  # e.g., 512
        self.embed_dim2 = embed_dim2  # e.g., 256
        self.num_heads = num_heads

        # Project mat2 to the same dim as mat1 for attention compatibility
        self.proj_mat2 = nn.Linear(embed_dim2, embed_dim1)
        self.cross_attn = nn.MultiheadAttention(embed_dim1, num_heads)
        
    def forward(self, mat1, mat2):
        # mat1: [B, 512, 32, 32], mat2: [B, 256, 32, 32]
        B, C1, H, W = mat1.shape
        _, C2, _, _ = mat2.shape

        # Flatten spatial dimensions
        x1 = mat1.view(B, C1, -1).permute(2, 0, 1)  # [HW, B, C1]
        x2 = mat2.view(B, C2, -1).permute(2, 0, 1)  # [HW, B, C2]

        # Project x2 to match C1
        x2_proj = self.proj_mat2(x2)  # [HW, B, C1]

        # Cross-attention: Query=x1, Key/Value=x2_proj
        # Output: [HW, B, C1]
        attn_output, _ = self.cross_attn(query=x1, key=x2_proj, value=x2_proj)

        # Add & reshape to original mat1 shape
        fused = attn_output.permute(1, 2, 0).contiguous().view(B, C1, H, W)  # [B, 512, 32, 32]
        return fused



    

def apply_dora_to_sam2_encoder(sam_encoder, r=8, alpha=1.0):
    # print("=== 可用 Linear 模組名 ===")
    # for name, module in sam_encoder.named_modules():
    #     if isinstance(module, nn.Linear):
    #         print(name)
    # target_modules = []
    # for i in range(48):
    #     target_modules += [
    #         f"trunk.blocks.{i}.attn.qkv",
    #         f"trunk.blocks.{i}.attn.proj",
    #         f"trunk.blocks.{i}.mlp.layers.0",
    #         f"trunk.blocks.{i}.mlp.layers.1"
    #     ]
    dora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        # target_modules=target_modules,
        target_modules = [
                "attn.qkv",
                "attn.proj",
                "mlp.layers.0",
                "mlp.layers.1",
                "proj"
            ],
        lora_dropout=0.1,
        use_dora=True,
        bias="none",
        task_type="FEATURE_EXTRACTION"
    )
    sam_encoder = get_peft_model(sam_encoder, dora_config)

    # freeze base parameters, keep adapters trainable
    for name, param in sam_encoder.named_parameters():
        if "lora_" not in name and "dora_" not in name:
            param.requires_grad = False

    return sam_encoder



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
            # channels_decoder = [512, 512, 512]
            channels_decoder=[256, 256, 256]

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
        

        # -------------- Add SAM2 encoder --------------
        # 1. Load configuration and checkpoint
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        checkpoint = "sam2.1_hiera_large.pt" 

        # 2. Build the SAM2 image model
        sam2_model = build_sam2(model_cfg, checkpoint, device="cuda")

        # 3. Get the encoder (backbone) from SAM2
        sam2_encoder = sam2_model.image_encoder

        # 4. Apply DoRA to encoder
        rank = 6
        sam_encoder = apply_dora_to_sam2_encoder(sam2_encoder, r=rank, alpha=16)

        # 5. Assign to self.sam2_encoder
        self.sam2_encoder = sam_encoder
        print("DoRA applied to SAM2 encoder. Rank =", rank)

        # ----------------------------------------------

        self.first_conv = ConvBNAct(1, 3, kernel_size=1,
                                    activation=self.activation)
        # self.mid_conv = ConvBNAct(256 + 256, 512, kernel_size=1,
        #                           activation=self.activation)
        self.mid_img_conv = ConvBNAct(512, 256, kernel_size=1,
                                      activation=self.activation)
        # Remove ResNet

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

        # self.prompt_fusion_layer = DPE()
        self.fuse_model = CrossAttentionFuse(embed_dim1=256, embed_dim2=256, num_heads=2)  # try num_head = 2 and 4
        model_name = "depth-anything/Depth-Anything-V2-Small-hf"

        # 初始化
        self.da2_processor = AutoImageProcessor.from_pretrained(model_name)
        da2_model = DepthAnythingForDepthEstimation.from_pretrained(model_name)
        self.da2_encoder = da2_model.backbone  # 正確引用 encoder
        # freeze the DA2 encoder
        for param in self.da2_encoder.parameters():
            param.requires_grad = False
        self.depth_conv = ConvBNAct(channels_in=384, channels_out=256, kernel_size=1, activation=self.activation)

    def forward(self, image, depth):
        # ----- DA2 encoder for depth ------
        # print('depth shape:', depth.shape)  # [4, 1, 1024, 1024]

        with torch.no_grad():
            inputs = self.da2_processor(images=image, return_tensors="pt", do_rescale=False)
            # features = self.da2_encoder(inputs["pixel_values"])
            device = next(self.da2_encoder.parameters()).device
            pixel_values = inputs["pixel_values"].to(device)
            features = self.da2_encoder(pixel_values)
            # print("DA2 encoder output type:", type(features))

            # features is a BackboneOutput with .feature_maps containing a list of feature tensors
            feat = features.feature_maps[-1]  # [4, 1370, 384]
            # print('DA2 encoder output shape:', feat.shape)
            B, N, C = feat.shape
            H = W = int(N ** 0.5) 
            feat = feat[:, :H*W, :]  # [4, 1369, 384]
            # print('DA2 encoder output shape:', feat.shape)  
            feat = feat.permute(0, 2, 1).reshape(B, C, H, W)  # [4, 384, 37, 37]
            # print('DA2 encoder reshaped output shape:', feat.shape) 
            feat = self.depth_conv(feat)  # [4, 256, 37, 37]
            # print('DA2 encoder output after conv shape:', feat.shape)


        # print('depth_feat shape before interpolation:', depth_feat.shape)
        depth_feat = F.interpolate(feat, size=(64, 64), mode="bilinear", align_corners=False) # [B, 256, 64, 64]
        # print('depth_feat shape after interpolation:', depth_feat.shape) 

        # ----- SAM encoder for RGB ------
        # for name, module in self.sam2_encoder.named_modules():
        #     print(name, ":", module.__class__.__name__)
        out, skip1, skip2, skip3  = get_last_n_block_outputs(self.sam2_encoder, image)# [4, 64, 64, 768]
        # print('SAM encoder output shape:', out.shape). [4, 256, 64, 64]
        # print('SAM skip1 shape:', skip1.shape). [4, 256, 64, 64]
        # print('SAM skip2 shape:', skip2.shape). [4, 256, 128, 128]
        # print('SAM skip3 shape:', skip3.shape). [4, 256, 256, 256]

       
        # ----- Cross Attention Fusion ------
        # Ensure depth_feat and out have same spatial dimensions before fusion
        out_resized = F.interpolate(out, size=depth_feat.shape[-2:], mode='bilinear', align_corners=False)
        fused_feat = self.fuse_model(depth_feat, out_resized) # [4, 256, 64, 64]
        # print('Fused feature shape:', fused_feat.shape)  # [4, 256, 64, 64]
        
        # Resize all features to have consistent spatial dimensions
        # The decoder expects features with progressively smaller spatial sizes
        # Let's make them all the same size as the largest feature for now
        target_size = (256, 256)  # You may need to adjust this based on your input size
        
        
        outs = [fused_feat, skip3, skip2, skip1]

        outs, out_visual = self.decoder(enc_outs=outs) # outs: [4, 1024, 1024]
        # print('decoder outs shape:', outs.shape) # [4, 4, 256, 256]
        outs = F.log_softmax(outs, dim=1) # [1, 1, 1024, 1024]

        return outs, depth_feat
