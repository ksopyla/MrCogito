import torch

if torch.cuda.is_available():
    device = torch.device("cuda")          # a CUDA device object
    print(f'Cuda version: {torch.version.cuda}, CuDNN:{torch.backends.cudnn.version()}')
    print(torch.cuda.get_device_properties(0))
   
    # allocate tensor on gpu
    x = torch.ones(5, 3).cuda()
    print(x)
else:
    print('GPU not enabled')