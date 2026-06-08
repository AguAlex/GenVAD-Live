import torch
from einops import rearrange
from torch import nn
from model.cvt import ConvEmbed, Block
from util.morphology import Erosion2d, Dilation2d

class MaskedAutoencoderCvT(nn.Module):
    def __init__(self, img_size=(320, 640), patch_size=16, in_chans=3, out_chans=3,
                 embed_dim=128, depth=2, num_heads=2,
                 decoder_embed_dim=64, decoder_depth=1, decoder_num_heads=2,
                 mlp_ratio=4., norm_layer=nn.LayerNorm,
                 use_only_masked_tokens_ab=False, masking_method="random_masking"):
        super().__init__()
        
        self.use_only_masked_tokens_ab = use_only_masked_tokens_ab
        self.masking = getattr(self, masking_method)
        
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = ConvEmbed(
            patch_size=patch_size, in_chans=in_chans, stride=patch_size,
            padding=0, embed_dim=embed_dim, norm_layer=norm_layer
        )
        self.patch_size = patch_size
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.blocks = nn.ModuleList([
            Block(embed_dim, embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(depth)])
        self.norm = norm_layer(embed_dim)
        
        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * out_chans, bias=True)
        self.out_chans = out_chans

        # Utilities pentru post-procesare (opțional pentru inferență)
        self.erosion = Erosion2d(1, 1, 2, soft_max=False)
        self.dilation = Dilation2d(1, 1, 3, soft_max=False)

    def patchify(self, imgs):
        p = self.patch_embed.patch_size[0]
        h = imgs.shape[2] // p
        w = imgs.shape[3] // p
        x = imgs.reshape(shape=(imgs.shape[0], self.out_chans, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * self.out_chans))
        return x

    def random_masking(self, x, mask_ratio):
        N, D, H, W = x.shape
        L = H * W
        x = rearrange(x, 'b c h w -> b (h w) c')
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        self.masked_H = H
        self.masked_W = int(W * (1. - mask_ratio))
        self.H, self.W = H, W
        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        x = self.patch_embed(x)
        x, mask, ids_restore = self.masking(x, mask_ratio)
        
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        for blk in self.blocks:
            x = blk(x, self.masked_H, self.masked_W)
        x = self.norm(x)
        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        x = self.decoder_embed(x)
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        x = torch.cat([x[:, :1, :], x_], dim=1)

        for blk in self.decoder_blocks:
            x = blk(x, self.H, self.W)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        x = x[:, 1:, :] # scoatem cls token
        return x

    def forward_loss(self, imgs, pred, mask):
        target = self.patchify(imgs)
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  
        loss = (loss * mask).sum() / mask.sum()  # Calculăm pierderea DOAR pe patch-urile mascate
        return loss

    def forward(self, imgs, mask_ratio=0.75):
        # Așteptăm ca imgs să fie imaginea curată (sau imaginea cu anomalii, depinde cum setezi in dataset)
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)

        if self.training:
            return loss, pred, mask
        else:
            return loss, pred, mask, self.abnormal_score(imgs, pred, mask)

    def abnormal_score(self, imgs, pred, mask):
        imgs = self.patchify(imgs)
        if self.use_only_masked_tokens_ab:
            mask = mask.bool()
            selected_pred, selected_lbl = [], []
            for i in range(0, imgs.shape[0]):
                selected_pred.append(pred[i][mask[i]])
                selected_lbl.append(imgs[i][mask[i]])
            pred, imgs = torch.stack(selected_pred), torch.stack(selected_lbl)
        return ((imgs - pred) ** 2).mean((1, 2))