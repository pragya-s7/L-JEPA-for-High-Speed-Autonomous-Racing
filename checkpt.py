import torch

# Load the checkpoint file
checkpoint = torch.load('/home/jarvis615/roboracer_ws/src/f1tenth_gym/L-JEPA-for-High-Speed-Autonomous-Racing/checkpoints/pretrain/best.pt', map_location=torch.device('cpu'))


print(checkpoint.keys())

# If 'loss' is a key, you can print its value
if 'loss' in checkpoint:
    print(f"Loss value: {checkpoint['loss']}")
else:
    print("Loss value was not saved in this .pt file.")