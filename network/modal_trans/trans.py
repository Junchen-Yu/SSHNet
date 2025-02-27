import torch
import torch.nn as nn
from einops import rearrange
from network.modal_trans.utils import *

class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=3, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.channel_reduction = 16

        self.inc = (DoubleConv(n_channels, 16))
        self.down1 = (Down(16, 32))
        self.down2 = (Down(32, 64))
        self.down3 = (Down(64, 128))
        factor = 2 if bilinear else 1
        self.down4 = (Down(128, 256 // factor))
        self.up1 = (Up(256, 128 // factor, bilinear))
        self.up2 = (Up(128, 64 // factor, bilinear))
        self.up3 = (Up(64, 32 // factor, bilinear))
        self.up4 = (Up(32, 16, bilinear))
        self.outc = (OutConv(16, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

class TransformerUNet(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 input_size=(128, 128),
                 dim=16,
                 mlp_ratio=4.,
                 window_size=8, 
                 bias=False,
                 drop_path_rate=0.,
                 num_blocks=[2,2,2,2,6],
                 num_heads=[2, 4, 8, 16, 32]
                 ):

        super(TransformerUNet, self).__init__()
        self.patch_embed = PatchEmbed(inp_channels, dim)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(num_blocks))]  # stochastic depth decay rule

        self.init_encoder = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=(input_size[0], input_size[1]),
                        num_heads=num_heads[0],
                        window_size = window_size,
                        drop_path=dpr[i],
                        shift_size=0 if (i % 2 == 0) else window_size // 2,
                        mlp_ratio=mlp_ratio)
            for i in range(num_blocks[0])])

        self.down1 = Downsample(dim)
        self.encoder_level1 = nn.ModuleList([
            SwinTransformerBlock(dim=dim * 2, input_resolution=(input_size[0] // 2, input_size[1] // 2),
                                num_heads=num_heads[1],
                                window_size = window_size,
                                drop_path=dpr[i+sum(num_blocks[:1])],
                                shift_size=0 if (i % 2 == 0) else window_size // 2,
                                mlp_ratio=mlp_ratio)
            for i in range(num_blocks[1])])
        
        self.down2 = Downsample(int(dim * 2))
        self.encoder_level2 = nn.ModuleList([
            SwinTransformerBlock(dim=dim * 4, input_resolution=(input_size[0] // 4, input_size[1] // 4),
                                num_heads=num_heads[2],
                                window_size = window_size,
                                drop_path=dpr[i+sum(num_blocks[:2])],
                                shift_size=0 if (i % 2 == 0) else window_size // 2,
                                mlp_ratio=mlp_ratio)
            for i in range(num_blocks[2])])

        self.down3 = Downsample(int(dim * 4))
        self.encoder_level3 = nn.ModuleList([
            SwinTransformerBlock(dim=dim * 8, input_resolution=(input_size[0] // 8, input_size[1] // 8),
                                num_heads=num_heads[3],
                                window_size = window_size,
                                drop_path=dpr[i+sum(num_blocks[:3])],
                                shift_size=0 if (i % 2 == 0) else window_size // 2,
                                mlp_ratio=mlp_ratio)
            for i in range(num_blocks[3])])

        self.down4 = Downsample(int(dim * 8))
        self.encoder_level4 = nn.ModuleList([
            SwinTransformerBlock(dim=dim * 16, input_resolution=(input_size[0] // 16, input_size[1] // 16),
                                num_heads=num_heads[4],
                                window_size = window_size,
                                drop_path=dpr[i+sum(num_blocks[:4])],
                                shift_size=0 if (i % 2 == 0) else window_size // 2,
                                mlp_ratio=mlp_ratio)
            for i in range(num_blocks[4])])

        self.up4 = Upsample(int(dim * 16))
        self.chan_conv4 = nn.Conv2d(int(dim * 16), int(dim * 8), kernel_size=1, bias=bias)
        self.decoder_level4 = nn.ModuleList([
            SwinTransformerBlock(dim=dim * 8, input_resolution=(input_size[0] // 8, input_size[1] // 8),
                                num_heads=num_heads[3],
                                window_size = window_size,
                                drop_path=dpr[i+sum(num_blocks[:3])],
                                shift_size=0 if (i % 2 == 0) else window_size // 2,
                                mlp_ratio=mlp_ratio)
            for i in range(num_blocks[3])])

        self.up3 = Upsample(int(dim * 8))
        self.chan_conv3 = nn.Conv2d(int(dim * 8), int(dim * 4), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.ModuleList([
            SwinTransformerBlock(dim=dim * 4, input_resolution=(input_size[0] // 4, input_size[1] // 4),
                                num_heads=num_heads[2],
                                window_size = window_size,
                                drop_path=dpr[i+sum(num_blocks[:2])],
                                shift_size=0 if (i % 2 == 0) else window_size // 2,
                                mlp_ratio=mlp_ratio)
            for i in range(num_blocks[2])])
        
        self.up2 = Upsample(int(dim * 4))
        self.chan_conv2 = nn.Conv2d(int(dim * 4), int(dim * 2), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.ModuleList([
            SwinTransformerBlock(dim=dim * 2, input_resolution=(input_size[0] // 2, input_size[1] // 2),
                                num_heads=num_heads[1],
                                window_size = window_size,
                                drop_path=dpr[i+sum(num_blocks[:1])],
                                shift_size=0 if (i % 2 == 0) else window_size // 2,
                                mlp_ratio=mlp_ratio)
            for i in range(num_blocks[1])])
        
        self.up1 = Upsample(int(dim * 2))
        self.chan_conv1 = nn.Conv2d(int(dim * 2), dim, kernel_size=1, bias=bias)
        self.decoder_level1 = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=(input_size[0], input_size[1]),
                        num_heads=num_heads[0],
                        window_size = window_size,
                        drop_path=dpr[i],
                        shift_size=0 if (i % 2 == 0) else window_size // 2,
                        mlp_ratio=mlp_ratio)
            for i in range(num_blocks[0])])

        self.output = nn.Conv2d(dim, out_channels, kernel_size=1)

    def forward(self, inp_img):
        _, _, H, W = inp_img.shape
        inp_enc = self.patch_embed(inp_img)  # b,hw,c
        out_enc = inp_enc
        for layer in self.init_encoder:
            out_enc = layer(out_enc)

        inp_enc_level1 = self.down1(out_enc, H, W)  # b, hw//4, 2c
        out_enc_level1 = inp_enc_level1
        for layer in self.encoder_level1:
            out_enc_level1 = layer(out_enc_level1)

        inp_enc_level2 = self.down2(out_enc_level1, H // 2, W // 2)  # b, hw//16, 4c
        out_enc_level2 = inp_enc_level2
        for layer in self.encoder_level2:
            out_enc_level2 = layer(out_enc_level2)

        inp_enc_level3 = self.down3(out_enc_level2, H // 4, W // 4)  # b, hw//64, 8c
        out_enc_level3 = inp_enc_level3
        for layer in self.encoder_level3:
            out_enc_level3 = layer(out_enc_level3)

        inp_enc_level4 = self.down4(out_enc_level3, H // 8, W // 8)  # b, hw//256, 16c
        out_enc_level4 = inp_enc_level4
        for layer in self.encoder_level4:
            out_enc_level4 = layer(out_enc_level4)

        inp_dec_level4 = self.up4(out_enc_level4, H // 16, W // 16)  # b, hw//64, 8c
        inp_dec_level4 = torch.cat([inp_dec_level4, out_enc_level3], 2)
        inp_dec_level4 = rearrange(inp_dec_level4, "b (h w) c -> b c h w", h=H // 8, w=W // 8).contiguous()
        inp_dec_level4 = self.chan_conv4(inp_dec_level4)
        inp_dec_level4 = rearrange(inp_dec_level4, "b c h w -> b (h w) c").contiguous()  # b, hw//64, 8c
        out_dec_level4 = inp_dec_level4
        for layer in self.decoder_level4:
            out_dec_level4 = layer(out_dec_level4)

        inp_dec_level3 = self.up3(out_dec_level4, H // 8, W // 8)  # b, hw//16, 8c
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level2], 2)
        inp_dec_level3 = rearrange(inp_dec_level3, "b (h w) c -> b c h w", h=H // 4, w=W // 4).contiguous()
        inp_dec_level3 = self.chan_conv3(inp_dec_level3)
        inp_dec_level3 = rearrange(inp_dec_level3, "b c h w -> b (h w) c").contiguous()  # b, hw//16, 4c
        out_dec_level3 = inp_dec_level3
        for layer in self.decoder_level3:
            out_dec_level3 = layer(out_dec_level3)
    
        inp_dec_level2 = self.up2(out_dec_level3, H // 4, W // 4)  # b, hw//4, 2c
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level1], 2)
        inp_dec_level2 = rearrange(inp_dec_level2, "b (h w) c -> b c h w", h=H // 2, w=W // 2).contiguous()
        inp_dec_level2 = self.chan_conv2(inp_dec_level2)
        inp_dec_level2 = rearrange(inp_dec_level2, "b c h w -> b (h w) c").contiguous()  # b, hw//4, 2c
        out_dec_level2 = inp_dec_level2
        for layer in self.decoder_level2:
            out_dec_level2 = layer(out_dec_level2)

        inp_dec_level1 = self.up1(out_dec_level2, H // 2, W // 2)  # b, hw, c
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc], 2)
        inp_dec_level1 = rearrange(inp_dec_level1, "b (h w) c -> b c h w", h=H, w=W).contiguous()
        inp_dec_level1 = self.chan_conv1(inp_dec_level1)
        inp_dec_level1 = rearrange(inp_dec_level1, "b c h w -> b (h w) c").contiguous()  # b, hw, c
        out_dec_level1 = inp_dec_level1
        for layer in self.decoder_level1:
            out_dec_level1 = layer(out_dec_level1)

        out_dec_level1 = rearrange(out_dec_level1, "b (h w) c -> b c h w", h=H, w=W).contiguous()
        out_dec = self.output(out_dec_level1)

        return out_dec