"""visionjev_healthcheck.analyse on synthetic and real logs."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import visionjev_healthcheck as H

LINE = "step %d | %.1f min | loss %s (avg %s) | reward %s | lr 4.23e-05 | w_ce 1.00 | sigma 0.30 | grad %s | data wait %d%%"


def _log(losses, wait=0, grads=None, evals=(), spikes=0):
    lines = ["[eval step 0, untrained head] all n=2480 acc=%.3f ece=0.050 nll=0.495 mae=0.3 xent=0.7 soft_xent=0.8" % (evals[0] if evals else 0.78)]
    for i, l in enumerate(losses):
        g = grads[i] if grads else 60.0
        lines.append(LINE % (i * 50, i * 6.0, l, l, -0.1 + i * 0.01, g, wait))
    lines += ["loss spike step %d: 9.0 (recent avg 1.0, grad norm 200) batch: {}" % i for i in range(spikes)]
    for j, a in enumerate(evals[1:], 1):
        lines.append("[eval step %d] all n=2480 acc=%.3f ece=0.02 nll=0.44" % (j * 400, a))
    return "\n".join(lines)


def test_healthy_and_flat_and_diverging():
    ok = H.analyse(_log([2.0, 1.6, 1.3, 1.1, 1.0, 0.9]))
    assert ok["verdict"] == "healthy" and ok["steps"] == 250 and ok["s_per_step"] == 7.2 and not ok["fatal"]
    flat = H.analyse(_log([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))
    assert flat["verdict"] == "healthy" and any("flat" in w for w in flat["warn"])
    bad = H.analyse(_log([1.0, 1.2, 1.8, 2.4, 3.0, 3.5]))
    assert bad["verdict"] == "unhealthy" and any("diverging" in f for f in bad["fatal"])


def test_fatal_conditions():
    assert "non-finite" in " ".join(H.analyse(_log([1.0, 0.9, "nan", 0.8]))["fatal"])
    assert "data-loader" in " ".join(H.analyse(_log([1.0, 0.9, 0.8, 0.7], wait=55))["fatal"])
    assert "spikes" in " ".join(H.analyse(_log([1.0, 0.9, 0.8, 0.7], spikes=80))["fatal"])
    assert "baseline eval" in " ".join(H.analyse(_log([1.0, 0.9], evals=(0.1,)))["fatal"])
    assert "slower" in " ".join(H.analyse(_log([1.0, 0.9, 0.8]), max_s_per_step=5.0)["fatal"])
    assert "error in log" in " ".join(H.analyse(_log([1.0]) + "\nTraceback (most recent call last):")["fatal"])


def test_early_and_stalled():
    early = H.analyse(_log([]), now=1000.0, mtime=900.0)
    assert early["verdict"] == "early"
    stalled = H.analyse(_log([]), now=10000.0, mtime=0.0, min_wait=8)
    assert stalled["verdict"] == "unhealthy" and "no training step" in stalled["fatal"][0]
    assert any("dropping" in w for w in H.analyse(_log([1.0, 0.9, 0.8, 0.7], evals=(0.7, 0.8, 0.75, 0.70)))["warn"])


def test_real_b2_log_is_healthy():
    p = "/home/kneepolean/shubhj/VisionJev/data/logs/pod/b2/b2-b2.log"
    if not os.path.exists(p):
        return
    res = H.analyse(open(p, errors="replace").read())
    assert res["verdict"] == "healthy" and res["steps"] == 1200 and 6 < res["s_per_step"] < 9 and res["evals"][0][0] == 0
