from timm.models import VisionTransformer, create_model


def vit_small(img_size=32, num_classes=100, drop_path_rate=0.1):
    model = VisionTransformer(img_size=img_size,
                              patch_size=img_size // 8,
                              in_chans=3,
                              num_classes=num_classes,
                              embed_dim=384,
                              depth=12,
                              num_heads=6,
                              drop_path_rate=drop_path_rate)
    return model

def vit_base_32_224(pretrained=False, num_classes=1000, drop_path_rate=0.1):
    model = create_model("vit_base_patch32_224",
                         pretrained=pretrained, 
                         num_classes=num_classes,
                         drop_path_rate=drop_path_rate)
    return model
