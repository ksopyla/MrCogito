# MrCogito




## Installing dependencies

Most of the dependencies are installed via poetry. 
For pytorch I have added a "suplemental" Poetry source for CUDA 12.1

```toml
[[tool.poetry.source]]
name = "torch"
url = "https://download.pytorch.org/whl/cu121"
priority = "supplemental"
```

And then I have added the `torch` dependency as follows:

```bash
poetry add torch==2.2.1+cu121 --source torch
poetry add triton --source torch
```

