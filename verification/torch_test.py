"""Smoke-test the PyTorch install: report CUDA, MPS, or CPU and run a tiny op."""

import torch


def main() -> None:
    print(f"PyTorch: {torch.__version__}")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(
            f"CUDA: {torch.version.cuda}, "
            f"cuDNN: {torch.backends.cudnn.version()}"
        )
        print(torch.cuda.get_device_properties(0))
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        built = torch.backends.mps.is_built()
        print(f"MPS: available (built={built})")
    else:
        device = torch.device("cpu")
        print("No GPU/MPS detected — falling back to CPU.")

    x = torch.ones(5, 3, device=device)
    y = (x @ x.T).sum()
    print(f"Device: {device}, sample tensor sum: {y.item():.1f}")


if __name__ == "__main__":
    main()
