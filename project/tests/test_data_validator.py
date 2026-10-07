"""Tests for data_validator.py. Run: python3 -m unittest discover -s tests -v"""
import contextlib
import io
import json
import math
import os
import struct
import sys
import tempfile
import unittest
import zlib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import data_validator as dv  # noqa: E402


def png_bytes(seed: int) -> bytes:
    """A valid 1x1 PNG whose pixel depends on seed (distinct seeds -> distinct hashes)."""
    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    raw = b"\x00" + bytes([seed % 256, (seed // 256) % 256, 0])
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw))
            + chunk(b"IEND", b""))


DOC_TYPES = ["letter", "invoice", "form", "memo"]
COMMERCIAL = {"annotations": "CC BY 4.0", "images": "own", "commercial": True}


def choice_row(i, image=None):
    row = {"id": f"c{i}", "state_text": None,
           "question": {"type": "choice", "instructions": "What kind of document is this?", "criteria": list(DOC_TYPES)},
           "label": i % 4}  # gold cycles through positions -> uniform histogram
    if image:
        row["image"] = image
    return row


def noul_row(i, image=None):
    row = {"id": f"n{i}", "question": {"type": "noul", "instructions": "Is there a person in the image?",
                                       "criteria": None},
           "label": i % 2}
    if image:
        row["image"] = image
    return row


def score_row(i):
    return {"id": f"s{i}", "question": {"type": "score", "instructions": "Rate the image quality.",
                                        "criteria": ["bad", "poor", "fair", "good", "excellent"]},
            "label": i % 5}


class ValidatorTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self._next_img = 0

    def tearDown(self):
        self.tmp.cleanup()

    def dataset(self, name="ds", license=COMMERCIAL):
        d = os.path.join(self.tmp.name, name)
        os.makedirs(os.path.join(d, "images"), exist_ok=True)
        if license is not None:
            with open(os.path.join(d, "manifest.json"), "w") as f:
                json.dump({"license": license}, f)
        return d

    def img(self, d, data=None):
        self._next_img += 1
        rel = f"images/{self._next_img:06d}.png"
        with open(os.path.join(d, rel), "wb") as f:
            f.write(png_bytes(self._next_img) if data is None else data)
        return rel

    def write(self, d, split, rows):
        path = os.path.join(d, f"{split}.jsonl")
        with open(path, "w") as f:
            for r in rows:
                f.write((r if isinstance(r, str) else json.dumps(r)) + "\n")
        return path

    def good_rows(self, d, n=100):
        rows = []
        for i in range(n):
            rows += [choice_row(i, self.img(d)), noul_row(i, self.img(d)), score_row(i)]
        return rows

    def run_v(self, *paths, **kw):
        rep, _ = dv.run(list(paths), quiet=True, **kw)
        return rep

    def assertFailsWith(self, rep, check):
        self.assertIn(check, rep.errors, f"expected a {check!r} violation; got {dict(rep.errors)}")

    # --- happy path ------------------------------------------------------
    def test_valid_dataset_dir_passes(self):
        d = self.dataset()
        self.write(d, "train", self.good_rows(d))
        rep = self.run_v(d)
        self.assertEqual(rep.n_errors, 0, dict(rep.errors))

    def test_dict_criteria_and_soft_targets_pass(self):
        d = self.dataset()
        rows = self.good_rows(d)
        rows[0]["question"]["criteria"] = {c: None for c in DOC_TYPES}
        rows[0]["target"] = [0.7, 0.1, 0.1, 0.1]     # label 0
        rows[1]["target"] = [0.2, 0.8]               # noul: [P(false), P(true)], so the label must be 1
        rows[1]["label"] = 1
        rows[1]["value"] = 0.5
        self.write(d, "train", rows)
        rep = self.run_v(d)
        self.assertEqual(rep.n_errors, 0, dict(rep.errors))

    # --- deliberately broken rows ---------------------------------------
    def broken(self, mutate, check, **kw):
        d = self.dataset()
        rows = self.good_rows(d)
        mutate(rows, d)
        self.write(d, "train", rows)
        rep = self.run_v(d, **kw)
        self.assertFailsWith(rep, check)
        return rep

    def test_label_out_of_range(self):
        self.broken(lambda r, d: r[0].__setitem__("label", 4), "label")

    def test_label_string_rejected(self):
        self.broken(lambda r, d: r[0].__setitem__("label", "letter"), "label")

    def test_label_bool_rejected(self):
        self.broken(lambda r, d: r[1].__setitem__("label", True), "label")

    def test_noul_label_two(self):
        self.broken(lambda r, d: r[1].__setitem__("label", 2), "label")

    def test_missing_required_key(self):
        self.broken(lambda r, d: r[0].pop("label"), "schema")

    def test_unknown_key_schema_drift(self):
        self.broken(lambda r, d: r[0].__setitem__("labels", r[0].pop("label")), "schema")

    def test_old_multi_question_schema_rejected(self):
        def m(rows, d):
            rows[0] = {"questions": {"q": rows[0]["question"]}, "labels": {"q": "letter"}}
        self.broken(m, "schema")

    def test_invalid_json(self):
        self.broken(lambda r, d: r.__setitem__(0, '{"question": '), "schema")

    def test_bad_question_type(self):
        self.broken(lambda r, d: r[0]["question"].__setitem__("type", "multi"), "schema")

    def test_noul_with_criteria(self):
        self.broken(lambda r, d: r[1]["question"].__setitem__("criteria", ["yes", "no"]), "schema")

    def test_noul_outcome_descriptions_allowed(self):
        d = self.dataset()
        rows = self.good_rows(d)
        rows[1]["question"]["criteria"] = {"false": "No human attention is warranted.", "true": "A human should look."}
        self.write(d, "train", rows)
        self.assertEqual(self.run_v(d).n_errors, 0)

    def test_duplicate_options(self):
        self.broken(lambda r, d: r[0]["question"].__setitem__("criteria", ["letter", "letter", "memo", "form"]),
                    "options")

    def test_single_option(self):
        self.broken(lambda r, d: r[0]["question"].__setitem__("criteria", ["letter"]), "schema")

    def test_duplicate_id(self):
        self.broken(lambda r, d: r[1].__setitem__("id", r[0]["id"]), "schema")

    def test_soft_argmax_disagrees_with_label(self):
        self.broken(lambda r, d: r[0].__setitem__("target", [0.1, 0.7, 0.1, 0.1]), "soft")

    def test_soft_wrong_length(self):
        self.broken(lambda r, d: r[0].__setitem__("target", [0.5, 0.5]), "soft")

    def test_soft_not_distribution(self):
        self.broken(lambda r, d: r[0].__setitem__("target", [0.9, 0.9, 0.1, 0.1]), "soft")

    def test_value_out_of_range(self):
        self.broken(lambda r, d: r[0].__setitem__("value", 1.5), "schema")

    def test_missing_image(self):
        self.broken(lambda r, d: r[0].__setitem__("image", "images/does_not_exist.jpg"), "image")

    def test_corrupt_image(self):
        self.broken(lambda r, d: r[0].__setitem__("image", self.img(d, b"this is not an image")), "image")

    def test_multi_image_row_checked(self):
        self.broken(lambda r, d: r[0].__setitem__("images", [self.img(d), "images/nope.png"]) or r[0].pop("image"),
                    "image")

    # --- licenses ---------------------------------------------------------
    def test_missing_manifest_fails(self):
        d = self.dataset(license=None)
        self.write(d, "train", self.good_rows(d))
        self.assertFailsWith(self.run_v(d), "license")

    def test_noncommercial_dataset_fails_unless_allowed(self):
        d = self.dataset(license={"annotations": "CC BY-NC-SA 4.0", "commercial": False})
        self.write(d, "train", self.good_rows(d))
        self.assertFailsWith(self.run_v(d), "license")
        self.assertNotIn("license", self.run_v(d, allow_noncommercial=True).errors)

    def test_coco_filtered_rows_need_commercial_image_license(self):
        d = self.dataset(license={**COMMERCIAL, "image_filter": "coco"})
        rows = self.good_rows(d)
        for r in rows:
            r["image_license"] = "CC BY 2.0"
        rows[0]["image_license"] = "CC BY-NC 2.0"
        rows[1].pop("image_license")
        self.write(d, "train", rows)
        rep = self.run_v(d)
        self.assertFailsWith(rep, "license")
        self.assertEqual(len(rep.errors["license"]), 2, rep.errors["license"])  # the NC row and the unlabelled row

    # --- file-level checks ----------------------------------------------
    def test_noul_imbalance(self):
        d = self.dataset()
        rows = [noul_row(0, self.img(d)) | {"id": f"a{i}"} for i in range(90)] + \
               [noul_row(1, self.img(d)) | {"id": f"b{i}"} for i in range(10)]
        self.write(d, "train", rows)
        self.assertFailsWith(self.run_v(d), "noul_balance")

    def test_noul_imbalance_below_min_count_not_judged(self):
        d = self.dataset()
        self.write(d, "train", [noul_row(0, self.img(d)) | {"id": f"a{i}"} for i in range(10)])
        rep = self.run_v(d)
        self.assertEqual(rep.n_errors, 0, dict(rep.errors))

    def test_position_bias(self):
        d = self.dataset()
        self.write(d, "train", [choice_row(0, self.img(d)) | {"id": f"a{i}"} for i in range(200)])
        self.assertFailsWith(self.run_v(d), "position_bias")

    # --- cross-file leakage ---------------------------------------------
    def test_leakage_across_splits(self):
        d = self.dataset()
        train = self.good_rows(d, 40)
        self.write(d, "train", train)
        self.write(d, "val", [choice_row(900 + i, self.img(d)) for i in range(3)] + [choice_row(999, train[0]["image"])])
        self.assertFailsWith(self.run_v(d), "leakage")

    def test_leakage_by_content_across_datasets(self):
        a, b = self.dataset("a"), self.dataset("b")
        train = self.good_rows(a, 40)
        self.write(a, "train", train)
        with open(os.path.join(a, train[0]["image"]), "rb") as f:
            copy = self.img(b, f.read())                    # same bytes, other dataset, other path
        self.write(b, "val", [choice_row(0, copy)])
        self.assertFailsWith(self.run_v(a, b), "leakage")

    @unittest.skipIf(dv.Image is None, "perceptual hashing needs Pillow")
    def test_leakage_same_photo_encoded_differently(self):
        """The A-OKVQA / VQAv2 case: one COCO photo, re-encoded by each source, in one set's train and another's val."""
        a, b = self.dataset("a"), self.dataset("b")
        photo = dv.Image.new("RGB", (64, 48))
        wave = [int(128 + 100 * math.sin(x / 5) * math.cos(y / 4)) for y in range(48) for x in range(64)]
        photo.putdata([(v, 255 - v, v // 2) for v in wave])
        buf_png, buf_jpg = io.BytesIO(), io.BytesIO()
        photo.save(buf_png, "PNG")
        photo.resize((60, 45)).save(buf_jpg, "JPEG", quality=70)
        self.write(a, "train", self.good_rows(a, 40) + [choice_row(500, self.img(a, buf_png.getvalue()))])
        self.write(b, "val", [choice_row(0, self.img(b, buf_jpg.getvalue()))])
        rep = self.run_v(a, b)
        self.assertFailsWith(rep, "leakage")
        self.assertIn("(content)", rep.errors["leakage"][0])

    @unittest.skipIf(dv.Image is None, "perceptual hashing needs Pillow")
    def test_different_photos_are_not_leakage(self):
        """Two unrelated photos must not be flagged even if their 64-bit hashes happen to be close."""
        a, b = self.dataset("a"), self.dataset("b")
        def photo(fx, fy):
            im = dv.Image.new("L", (64, 48))
            im.putdata([int(128 + 100 * math.sin(x / fx) * math.cos(y / fy)) for y in range(48) for x in range(64)])
            buf = io.BytesIO()
            im.save(buf, "PNG")
            return buf.getvalue()
        self.write(a, "train", self.good_rows(a, 40) + [choice_row(500, self.img(a, photo(5, 4)))])
        self.write(b, "val", [choice_row(0, self.img(b, photo(2, 9)))])
        self.assertNotIn("leakage", self.run_v(a, b).errors)

    def test_same_split_may_share_images(self):
        d = self.dataset()
        rows = self.good_rows(d, 40)
        rows += [choice_row(500 + i, rows[0]["image"]) for i in range(4)]
        self.write(d, "train", rows)
        self.assertNotIn("leakage", self.run_v(d).errors)

    # --- CLI --------------------------------------------------------------
    def test_cli_exit_codes(self):
        good = self.dataset("good")
        self.write(good, "train", self.good_rows(good))
        bad = self.dataset("bad")
        rows = self.good_rows(bad)
        rows[0]["label"] = 9
        self.write(bad, "train", rows)
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(dv.main([good]), 0)
        self.assertIn("RESULT: PASS", out.getvalue())
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(dv.main([bad]), 1)
        self.assertIn("RESULT: FAIL", out.getvalue())
        self.assertIn("[label]", out.getvalue())


if __name__ == "__main__":
    unittest.main()
