#!/usr/bin/env python
"""CPU sanity check: validates the GPTNeoX API + our init/param-group/forward paths
against the installed transformers version, without needing a GPU or data."""

import torch
import train_control as T


def main():
    model, cfg = T.build_model("31M")
    n = sum(p.numel() for p in model.parameters())
    print(f"params = {n / 1e6:.2f}M  (expect ~31M)")

    names = dict(model.named_parameters())
    assert "embed_out.weight" in names, (
        f"no embed_out.weight; heads: {[k for k in names if 'out' in k][:5]}"
    )
    wu = names["embed_out.weight"]
    print(f"W_U (embed_out.weight) shape = {tuple(wu.shape)}  (expect (50304, 256))")

    groups = T.make_param_groups(model, 0.1, 4.0, 1e-3)
    sizes = {g["name"]: sum(p.numel() for p in g["params"]) for g in groups}
    lrs = {g["name"]: g["lr"] for g in groups}
    print(f"param-group sizes = {sizes}")
    print(f"param-group lrs   = {lrs}  (readout should be 4e-3)")
    assert sizes["readout"] == wu.numel(), "readout group must be exactly W_U"
    assert abs(lrs["readout"] - 4e-3) < 1e-9

    # forward + backward on random tokens
    torch.manual_seed(0)
    x = torch.randint(0, T.VOCAB, (2, T.SEQLEN))
    out = model(input_ids=x, labels=x)
    print(f"forward loss = {out.loss.item():.4f}  (expect ~ln(50304) ≈ 10.83 at init)")
    assert torch.isfinite(out.loss), "non-finite loss"
    out.loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    print(f"grad norm = {gn:.3f}  (finite={torch.isfinite(gn).item()})")
    print(f"W_U geometry = {T.geom_stats(model)}")
    print("SELFCHECK_OK")


if __name__ == "__main__":
    main()
