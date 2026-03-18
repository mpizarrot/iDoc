# iDoc

For a glimpse at the full documentation of iBOT pre-training, please run:
```bash
python main_ibot.py --help
   ```
To start the iBOT pre-training with LoRA, simply run the following command:
```bash
python main_ibot.py \
--arch vit_lora \
--data_path /../your_dataset.pkl \
--old_path /../old_path \
--new_path /../new_path \
--ckpt_path_student /../student.pth \
--ckpt_path_teacher /../teacher.pth \
--epochs 50 \
--batch_size_per_gpu 32 \
--output_dir ./output_dir
   ```

## Use a pretrained encoder on a single image

You can download the pretrained checkpoint from [this link](http://201.238.213.114:2280/sketchapp/get_files/idoc_pretrained.pth) and then use it to extract the embedding of a single image by loading the encoder in evaluation mode.

### 1. Load the pretrained model

```python
import torch
from models.vision_transformer_lora import vit_lora

ckpt_path = "idoc_pretrained.pth"
checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
state_dict = checkpoint.get("state_dict", checkpoint)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = vit_lora().to(device)
model.load_state_dict(state_dict)
model.eval()

print("Model loaded and set to eval mode.")
```

### 2. Preprocess the input image

The encoder expects an RGB image resized to `224x224` and normalized with ImageNet statistics.

```python
from PIL import Image, ImageOps
from torchvision import transforms

img = Image.open("1349.jpg").convert("RGB")
img = ImageOps.pad(img, size=(224, 224))

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

img_tensor = transform(img).unsqueeze(0).to(device)
```

### 3. Run inference and obtain the embedding

```python
with torch.no_grad():
    img_embedding = model(img_tensor)

print(img_embedding.shape)
```
