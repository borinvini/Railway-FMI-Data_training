import torch

from src.fcll_model import FCLLNet


def _batch(n=8, z=5, h=5, n_of=9, n_nf=4):
    return dict(
        z_seq=torch.randn(n, z, n_of), h_seq=torch.randn(n, h, n_of),
        z_mask=torch.ones(n, z), h_mask=torch.ones(n, h), nf=torch.randn(n, n_nf),
    )


def test_forward_returns_one_logit_per_sample():
    model = FCLLNet(n_of=9, n_nf=4)
    out = model(**_batch())
    assert out.shape == (8,)


def test_all_three_branches_receive_gradient():
    """A branch wired but never reaching the loss trains silently to nothing.
    The paper's ablation is the whole argument for having three."""
    model = FCLLNet(n_of=9, n_nf=4)
    model(**_batch()).sum().backward()
    named = dict(model.named_parameters())
    for branch in ("fcnn", "lstm_trains", "lstm_stations"):
        grads = [p.grad for n, p in named.items() if n.startswith(branch) and p.grad is not None]
        assert grads, f"{branch} received no gradient"
        assert any(g.abs().sum() > 0 for g in grads), f"{branch} gradient is all zero"


def test_masked_timesteps_do_not_change_the_output():
    """Padding is zeros; if the LSTM reads it, a short route and a long one
    with the same real history give different answers."""
    torch.manual_seed(42)
    model = FCLLNet(n_of=9, n_nf=4).eval()
    b = _batch(n=1)
    b["z_mask"][:, :2] = 0.0
    with torch.no_grad():
        first = model(**b)
        b["z_seq"][:, :2] = 999.0          # garbage in the masked slots
        second = model(**b)
    assert torch.allclose(first, second, atol=1e-5)


def test_the_model_is_deterministic_under_a_fixed_seed():
    torch.manual_seed(42)
    a = FCLLNet(n_of=9, n_nf=4).eval()
    torch.manual_seed(42)
    b = FCLLNet(n_of=9, n_nf=4).eval()
    batch = _batch()
    with torch.no_grad():
        assert torch.allclose(a(**batch), b(**batch))


def test_the_output_reads_the_last_real_timestep_not_a_padded_one():
    """Sequences are PRE-padded: the real steps sit at the end. Indexing by
    length instead of by mask position lands inside the padding, so the model
    would read a zero row and ignore the station nearest the target — the most
    informative one in the sequence."""
    torch.manual_seed(42)
    model = FCLLNet(n_of=9, n_nf=4).eval()

    b = _batch(n=1)
    b["z_seq"].zero_()
    b["z_mask"].zero_()
    b["z_mask"][:, 3:] = 1.0            # two real steps, pre-padded into slots 3,4
    b["z_seq"][:, 3:] = 0.0

    with torch.no_grad():
        baseline = model(**b)
        b["z_seq"][:, 4] = 5.0          # change the LAST real step
        changed = model(**b)

    assert not torch.allclose(baseline, changed, atol=1e-6), (
        "the final real timestep had no effect on the output"
    )


def test_a_fully_padded_sequence_contributes_nothing():
    """An originating train has no upstream stations at all. That must read as
    'no information', not as whatever the zero row happens to encode."""
    torch.manual_seed(42)
    model = FCLLNet(n_of=9, n_nf=4).eval()
    b = _batch(n=1)
    b["z_mask"].zero_()
    with torch.no_grad():
        first = model(**b)
        b["z_seq"].fill_(123.0)
        second = model(**b)
    assert torch.allclose(first, second, atol=1e-5)
