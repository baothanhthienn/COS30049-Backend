import pytest
from api.ml.stage_b import predict_family

_FAMILIES = [
    "encoding_obfuscation",
    "instruction_override",
    "role_play_jailbreak",
    "context_exfiltration",
    "delimiter_escape",
    "indirect_injection",
]


def zero_features() -> list[float]:
    return [0.0] * 21

class TestOutputContract:
    def test_returns_tuple_of_family_and_scores(self):
        family, scores = predict_family(zero_features())
        assert isinstance(family, str)
        assert isinstance(scores, dict)

    def test_scores_has_all_six_families(self):
        _, scores = predict_family(zero_features())
        assert set(scores.keys()) == set(_FAMILIES)

    def test_scores_sum_to_one(self):
        _, scores = predict_family(zero_features())
        assert abs(sum(scores.values()) - 1.0) < 1e-3

    def test_scores_all_between_zero_and_one(self):
        _, scores = predict_family(zero_features())
        for name, val in scores.items():
            assert 0.0 <= val <= 1.0, f"{name}={val} out of range"

    def test_returned_family_is_one_of_six(self):
        family, _ = predict_family(zero_features())
        assert family in _FAMILIES

    def test_returned_family_has_highest_score(self):
        f = zero_features()
        f[0] = 3.0   
        family, scores = predict_family(f, spans=[{"label": "instruction_override", "start": 0, "end": 5}])
        assert scores[family] == max(scores.values()), \
            f"Winner '{family}' (score {scores[family]}) is not the max: {scores}"


class TestEncodingPriority:
    def test_base64_blob_flag(self):
        f = zero_features()
        f[4] = 1.0   
        family, _ = predict_family(f)
        assert family == "encoding_obfuscation"

    def test_encoding_anomaly_above_threshold(self):
        f = zero_features()
        f[9] = 0.3  
        family, _ = predict_family(f)
        assert family == "encoding_obfuscation"

    def test_encoding_anomaly_below_threshold_does_not_trigger(self):
        f = zero_features()
        f[9] = 0.1   
        family, _ = predict_family(f)
        assert family != "encoding_obfuscation"

    def test_unicode_lookalike_two_or_more(self):
        f = zero_features()
        f[5] = 2.0  
        family, _ = predict_family(f)
        assert family == "encoding_obfuscation"

    def test_unicode_lookalike_one_wins_via_p3_not_p1(self):
        f = zero_features()
        f[5] = 1.0
        family, _ = predict_family(f)
        assert family == "encoding_obfuscation"

    def test_zero_width_char_triggers(self):
        f = zero_features()
        f[6] = 1.0  
        family, _ = predict_family(f)
        assert family == "encoding_obfuscation"

    def test_leetspeak_above_threshold(self):
        f = zero_features()
        f[7] = 0.4  
        family, _ = predict_family(f)
        assert family == "encoding_obfuscation"

    def test_encoding_beats_span(self):
        f = zero_features()
        f[4] = 1.0   # encoding flag
        spans = [{"label": "instruction_override", "start": 0, "end": 5}]
        family, _ = predict_family(f, spans=spans)
        assert family == "encoding_obfuscation"

class TestSpanPriority:
    def test_instruction_override_span(self):
        spans = [{"label": "instruction_override", "start": 0, "end": 6}]
        family, _ = predict_family(zero_features(), spans=spans)
        assert family == "instruction_override"

    def test_role_swap_span(self):
        spans = [{"label": "role_swap", "start": 0, "end": 7}]
        family, _ = predict_family(zero_features(), spans=spans)
        assert family == "role_play_jailbreak"

    def test_narrative_frame_span(self):
        spans = [{"label": "narrative_frame", "start": 0, "end": 10}]
        family, _ = predict_family(zero_features(), spans=spans)
        assert family == "role_play_jailbreak"

    def test_data_exfiltration_span(self):
        spans = [{"label": "data_exfiltration", "start": 0, "end": 8}]
        family, _ = predict_family(zero_features(), spans=spans)
        assert family == "context_exfiltration"

    def test_filter_bypass_span(self):
        spans = [{"label": "filter_bypass", "start": 0, "end": 5}]
        family, _ = predict_family(zero_features(), spans=spans)
        assert family == "delimiter_escape"

    def test_unknown_span_label_ignored(self):
        spans = [{"label": "unknown_attack_type", "start": 0, "end": 5}]
        family, _ = predict_family(zero_features(), spans=spans)
        assert family in _FAMILIES   # any valid family just no crash

    def test_span_beats_feature_threshold(self):
        f = zero_features()
        f[0] = 3.0   
        spans = [{"label": "role_swap", "start": 0, "end": 5}]
        family, _ = predict_family(f, spans=spans)
        assert family == "role_play_jailbreak"

    def test_first_matching_span_wins(self):
        spans = [
            {"label": "role_swap", "start": 0, "end": 5},
            {"label": "instruction_override", "start": 6, "end": 12},
        ]
        family, _ = predict_family(zero_features(), spans=spans)
        assert family == "instruction_override"

    def test_none_spans_handled(self):
        family, scores = predict_family(zero_features(), spans=None)
        assert family in _FAMILIES
        assert len(scores) == 6

class TestFeatureThreshold:
    def test_override_keywords_win_instruction_override(self):
        f = zero_features()
        f[0] = 3.0   
        family, _ = predict_family(f)
        assert family == "instruction_override"

    def test_role_swap_keywords_win_role_play_jailbreak(self):
        f = zero_features()
        f[1] = 3.0  
        family, _ = predict_family(f)
        assert family == "role_play_jailbreak"

    def test_data_exfil_keywords_win_context_exfiltration(self):
        f = zero_features()
        f[2] = 3.0   
        family, _ = predict_family(f)
        assert family == "context_exfiltration"

    def test_filter_bypass_wins_delimiter_escape(self):
        f = zero_features()
        f[3] = 3.0  
        family, _ = predict_family(f)
        assert family == "delimiter_escape"

    def test_weak_signal_below_threshold_falls_through(self):
        f = zero_features()
        f[17] = 1.0 
        f2 = zero_features()
        f2[11] = 0.1 
        family, _ = predict_family(f2)
        assert family == "indirect_injection"


class TestDelimiterFallback:
    def test_high_special_char_ratio_gives_delimiter_escape(self):
        f = zero_features()
        f[15] = 0.15  
        family, _ = predict_family(f)
        assert family == "delimiter_escape"

    def test_special_char_ratio_at_boundary_does_not_trigger(self):
        f = zero_features()
        f[15] = 0.10 
        family, _ = predict_family(f)
        assert family != "delimiter_escape"


class TestDefaultFallback:
    def test_all_zeros_returns_indirect_injection(self):
        family, _ = predict_family(zero_features())
        assert family == "indirect_injection"

    def test_scores_still_valid_on_all_zeros(self):
        _, scores = predict_family(zero_features())
        assert abs(sum(scores.values()) - 1.0) < 1e-3
        assert all(0.0 <= v <= 1.0 for v in scores.values())
        
class TestScoreDistribution:
    def test_winner_score_is_highest(self):
        f = zero_features()
        f[0] = 2.0   
        family, scores = predict_family(f)
        assert scores[family] == max(scores.values())

    def test_encoding_score_dominates_when_base64_present(self):
        f = zero_features()
        f[4] = 1.0
        family, scores = predict_family(f)
        assert family == "encoding_obfuscation"
        assert scores["encoding_obfuscation"] > 0.5

    def test_multiple_signals_still_sum_to_one(self):
        f = zero_features()
        f[0] = 2.0   # instruction_override
        f[1] = 2.0   # role_play_jailbreak
        f[2] = 1.0   # context_exfiltration
        _, scores = predict_family(f)
        assert abs(sum(scores.values()) - 1.0) < 1e-3
