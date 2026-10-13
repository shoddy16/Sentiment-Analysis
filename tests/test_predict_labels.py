import numpy as np
import pytest

from src.predict import SentimentPredictor


@pytest.mark.parametrize(
    ("label", "classes", "expected"),
    [
        (0, [0, 1], "negative"),
        (1, [0, 1], "positive"),
        (0, [0, 1, 2], "negative"),
        (1, [0, 1, 2], "neutral"),
        (2, [0, 1, 2], "positive"),
        (0, [0, 2, 4], "negative"),
        (2, [0, 2, 4], "neutral"),
        (4, [0, 2, 4], "positive"),
        (-1, [-1, 0, 1], "negative"),
        (0, [-1, 0, 1], "neutral"),
        (1, [-1, 0, 1], "positive"),
    ],
)
def test_readable_numeric_labels(label, classes, expected):
    assert SentimentPredictor._readable_label(label, classes) == expected


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("pos", "positive"),
        ("POSITIVE", "positive"),
        ("neg", "negative"),
        ("NEU", "neutral"),
        ("neutral", "neutral"),
    ],
)
def test_readable_string_labels(label, expected):
    assert SentimentPredictor._readable_label(label) == expected


def test_unknown_numeric_classes_are_preserved():
    assert SentimentPredictor._readable_label(7, [3, 7]) == "7"


def test_confidences_preserve_all_three_classes():
    classes = np.array([0, 1, 2])
    probabilities = np.array([0.2, 0.6, 0.2])

    result = SentimentPredictor._readable_confidences(classes, probabilities)

    assert result == {
        "negative": pytest.approx(0.2),
        "neutral": pytest.approx(0.6),
        "positive": pytest.approx(0.2),
    }


def test_csv_output_preserves_neutral_confidence(tmp_path):
    predictor = SentimentPredictor.__new__(SentimentPredictor)
    predictor.is_loaded = True
    predictor.predict_batch = lambda texts, return_confidence=False: [
        {"negative": 0.1, "neutral": 0.7, "positive": 0.2},
        {"negative": 0.8, "neutral": 0.1, "positive": 0.1},
    ]

    csv_path = tmp_path / "reviews.csv"
    csv_path.write_text(
        "text\nThis is okay\nThis is bad\n",
        encoding="utf-8",
    )

    result = predictor.predict_from_csv(
        str(csv_path), "text", return_confidence=True
    )

    assert result["sentiment"].tolist() == ["neutral", "negative"]
    assert result["confidence_neutral"].tolist() == pytest.approx([0.7, 0.1])
    assert result["confidence"].tolist() == pytest.approx([0.7, 0.8])


def test_csv_output_supports_models_without_probabilities(tmp_path):
    predictor = SentimentPredictor.__new__(SentimentPredictor)
    predictor.is_loaded = True
    predictor.predict_batch = lambda texts, return_confidence=False: [
        "neutral", "positive"
    ]

    csv_path = tmp_path / "reviews.csv"
    csv_path.write_text(
        "text\nThis is okay\nThis is great\n",
        encoding="utf-8",
    )

    result = predictor.predict_from_csv(
        str(csv_path), "text", return_confidence=True
    )

    assert result["sentiment"].tolist() == ["neutral", "positive"]
    assert "confidence" not in result.columns
