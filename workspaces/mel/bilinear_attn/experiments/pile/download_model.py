from huggingface_hub import snapshot_download
import torch
import os

path = snapshot_download(
    repo_id="Elriggs/bilinear_atnn_only_2L_Pile",
    revision="e185f7e72a20ccf0022a15b5a6f31b6c7b2d66b0",
)

# find your .pt file
checkpoint_path = os.path.join(path, "checkpoints")
pt_path = [f for f in os.listdir(checkpoint_path) if f.endswith(".pt")][0]
pt_path = os.path.join(checkpoint_path, pt_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
state = torch.load(pt_path, map_location=device)
print(state.keys())