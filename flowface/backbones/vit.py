from flowvision.models import VisionTransformer

def vit_t():
    return VisionTransformer(
        img_size=112, patch_size=9, num_classes=512, embed_dim=256, depth=12,
        num_heads=8, drop_path_rate=0.1, norm_layer="ln")
    

if __name__ == "__main__":
    vit_t()