import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from outis.detection import deberta_attention  # noqa: E402


def tiny_deberta(share_att_key=True, position_buckets=16):
    from transformers import DebertaV2Config, DebertaV2ForTokenClassification

    config = DebertaV2Config(
        vocab_size=120,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=512,
        relative_attention=True,
        position_buckets=position_buckets,
        max_relative_positions=-1,
        pos_att_type=["p2c", "c2p"],
        share_att_key=share_att_key,
        norm_rel_ebd="layer_norm",
        position_biased_input=False,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        num_labels=5,
    )
    torch.manual_seed(0)
    return DebertaV2ForTokenClassification(config).eval()


def run(model, ids, mask, labels):
    model.zero_grad(set_to_none=True)
    out = model(input_ids=ids, attention_mask=mask, labels=labels)
    out.loss.backward()
    grads = {
        n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None
    }
    return out.logits.detach(), grads


@pytest.mark.parametrize("share_att_key", [True, False])
@pytest.mark.parametrize("length", [1, 2, 9, 40, 130])
def test_gather_free_bias_matches_stock_outputs_and_gradients(share_att_key, length):
    # 16 buckets make lengths past 8 use DeBERTa's log-bucketed positions.
    stock = tiny_deberta(share_att_key=share_att_key)
    fast = tiny_deberta(share_att_key=share_att_key)
    fast.load_state_dict(stock.state_dict())
    assert deberta_attention.apply_gather_free_attention(fast)
    generator = torch.Generator().manual_seed(length)
    ids = torch.randint(4, 120, (3, length), generator=generator)
    mask = torch.ones(3, length, dtype=torch.long)
    mask[1, length // 2 + 1 :] = 0
    labels = torch.randint(0, 5, (3, length), generator=generator).masked_fill(
        mask == 0, -100
    )

    stock_logits, stock_grads = run(stock, ids, mask, labels)
    fast_logits, fast_grads = run(fast, ids, mask, labels)

    torch.testing.assert_close(fast_logits, stock_logits, rtol=0, atol=1e-5)
    assert fast_grads.keys() == stock_grads.keys()
    for name in stock_grads:
        torch.testing.assert_close(
            fast_grads[name], stock_grads[name], rtol=1e-4, atol=1e-6
        )


def test_bias_is_bound_per_instance_not_globally():
    from transformers.models.deberta_v2 import modeling_deberta_v2

    stock_method = (
        modeling_deberta_v2.DisentangledSelfAttention.disentangled_attention_bias
    )
    fast = tiny_deberta()
    other = tiny_deberta()
    deberta_attention.apply_gather_free_attention(fast)
    attention = fast.deberta.encoder.layer[0].attention.self
    untouched = other.deberta.encoder.layer[0].attention.self
    assert (
        attention.disentangled_attention_bias.__func__
        is deberta_attention.gather_free_attention_bias
    )
    assert untouched.disentangled_attention_bias.__func__ is stock_method
    assert (
        modeling_deberta_v2.DisentangledSelfAttention.disentangled_attention_bias
        is stock_method
    )


def test_other_model_types_are_left_unchanged():
    from transformers import RobertaConfig, RobertaForTokenClassification

    model = RobertaForTokenClassification(
        RobertaConfig(
            vocab_size=50,
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
        )
    )
    assert deberta_attention.apply_gather_free_attention(model) is False


def test_unchecked_transformers_code_keeps_stock(monkeypatch):
    monkeypatch.setitem(
        deberta_attention.CHECKED_SOURCES, "build_relative_position", "0" * 64
    )
    model = tiny_deberta()
    with pytest.warns(UserWarning, match="checked version"):
        assert deberta_attention.apply_gather_free_attention(model) is False
    attention = model.deberta.encoder.layer[0].attention.self
    assert "disentangled_attention_bias" not in vars(attention)


def test_skew_aligns_offsets_into_query_key_grid():
    length = 4
    offsets = torch.arange(2 * length - 1, dtype=torch.float32).repeat(length, 1)[None]
    grid = deberta_attention._skew(offsets)[0]
    for i in range(length):
        for j in range(length):
            assert grid[i, j] == j - i + length - 1
