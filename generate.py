import torch

from minitransformer import (
    GPTLanguageModel,
    decode,
    device
)

model = GPTLanguageModel().to(device)

model.load_state_dict(
    torch.load(
        "checkpoint_2000.pth",
        map_location=device
    )
)

model.eval()

context = torch.zeros(
    (1,1),
    dtype=torch.long,
    device=device
)

generated = model.generate(
    context,
    max_new_tokens=1000
)

print(
    decode(
        generated[0].tolist()
    )
)