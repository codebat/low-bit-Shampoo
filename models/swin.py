from timm.models import SwinTransformer


def swin_tiny(img_size=32, num_classes=100, drop_path_rate=0.1):
    window_size = 4 if img_size == 32 else 7
    model = SwinTransformer(img_size=img_size,
                            patch_size=img_size // 16,
                            in_chans=3,
                            num_classes=num_classes,
                            embed_dim=96,
                            depths=(2, 2, 6, 2),
                            num_heads=(3, 6, 12, 24),
                            drop_path_rate=drop_path_rate,
                            window_size=window_size)
    return model
