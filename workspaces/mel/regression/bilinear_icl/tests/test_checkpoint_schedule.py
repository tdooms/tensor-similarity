from bilinear_icl.train.checkpoint import build_schedule


def test_checkpoint_schedule_properties():
    max_steps = 500_000
    n_log = 100
    n_lin = 100
    s1 = build_schedule(max_steps, n_log=n_log, n_lin=n_lin)
    s2 = build_schedule(max_steps, n_log=n_log, n_lin=n_lin)
    assert s1 == s2
    assert 0 in s1
    assert max_steps in s1
    assert all(a < b for a, b in zip(s1, s1[1:]))
    assert len(set(s1)) == len(s1)
    assert all(0 <= x <= max_steps for x in s1)
    assert len(s1) <= n_log + n_lin + 2
