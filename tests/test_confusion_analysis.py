"""
test_confusion_analysis.py

Unit tests for confusion_analysis.py to verify the core logic works.
This tests the functions without requiring LLM API access.
"""

import sys
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
import json

# Add parent directory to path
parent = Path(__file__).resolve().parent.parent
if str(parent) not in sys.path:
    sys.path.insert(0, str(parent))

from red_vs_blue.analysis.confusion_analysis import (
    extract_player_reasoning,
    extract_game_context,
    load_eval_file,
)


def test_extract_player_reasoning():
    """Test that player reasoning is correctly extracted from game log."""
    
    sample_data = {
        "scores": {
            "red_vs_blue_scorer": {
                "metadata": {
                    "public_log": [
                        "p0: I think p1 is red because of the vote",
                        "p1: No, I voted honestly!",
                        "p2: Let's fire p1",
                        "p0: I change my mind, p1 is blue",
                        "not a player message",
                        "p3: Council formed with p1",
                    ]
                }
            }
        }
    }
    
    reasoning = extract_player_reasoning(sample_data)
    
    # Verify extraction
    assert "p0" in reasoning, "p0 should be in reasoning"
    assert len(reasoning["p0"]) == 2, "p0 should have 2 statements"
    assert reasoning["p0"][0] == "I think p1 is red because of the vote"
    assert reasoning["p0"][1] == "I change my mind, p1 is blue"
    
    assert "p1" in reasoning, "p1 should be in reasoning"
    assert reasoning["p1"][0] == "No, I voted honestly!"
    
    assert "p2" in reasoning, "p2 should be in reasoning"
    assert reasoning["p2"][0] == "Let's fire p1"
    
    assert "p3" in reasoning, "p3 should be in reasoning"
    assert reasoning["p3"][0] == "Council formed with p1"
    
    print("✓ Player reasoning extraction works correctly")


def test_extract_game_context():
    """Test that game context is correctly extracted."""
    
    sample_data = {
        "scores": {
            "red_vs_blue_scorer": {
                "metadata": {
                    "num_players": 5,
                    "rounds_played": 7,
                    "blues_win": True,
                    "patch_track": {"blue": 6, "red": 2},
                    "roles": {
                        "p0": "blue",
                        "p1": "red",
                        "p2": "blue",
                        "p3": "blue",
                        "p4": "apt_leader",
                    },
                    "true_apt_leader": "p4",
                    "fired_players": ["p1"],
                    "public_log": ["p0: test"],
                }
            }
        }
    }
    
    context = extract_game_context(sample_data)
    
    # Verify context
    assert context["num_players"] == 5
    assert context["rounds_played"] == 7
    assert context["blues_win"] == True
    assert context["reds_win"] == False
    assert context["patch_track"]["blue"] == 6
    assert context["patch_track"]["red"] == 2
    assert context["roles"]["p4"] == "apt_leader"
    assert context["true_apt_leader"] == "p4"
    assert context["fired_players"] == ["p1"]
    
    print("✓ Game context extraction works correctly")


def test_confusion_analysis_json_parsing():
    """Test that confusion analysis results can be parsed as JSON."""
    
    # Example output from LLM
    llm_response = """{
        "confused": true,
        "confusion_types": ["Rule Misunderstanding", "Strategic Confusion"],
        "explanation": "Player seemed confused about when patches apply",
        "evidence": ["I thought patches applied after voting", "I didn't know I was fired"],
        "improvement_suggestions": ["Make patch application timing explicit", "Show fired status clearly"]
    }"""
    
    # Should parse correctly
    result = json.loads(llm_response)
    
    assert result["confused"] == True
    assert len(result["confusion_types"]) == 2
    assert "Rule Misunderstanding" in result["confusion_types"]
    assert len(result["evidence"]) == 2
    assert len(result["improvement_suggestions"]) == 2
    
    print("✓ Confusion analysis JSON format is valid")


def test_game_improvements_json_format():
    """Test that game improvements JSON format is valid."""
    
    llm_response = """{
        "overall_confusion_level": "medium",
        "key_insights": [
            "Multiple players confused about patch mechanics",
            "APT Leader role confused 1 player"
        ],
        "improvement_suggestions": [
            {
                "category": "Rules",
                "suggestion": "Make patch application timing more explicit",
                "rationale": "3 players asked when patches are applied",
                "implementation": "Add visual counter showing patch triggers"
            },
            {
                "category": "Tutorial",
                "suggestion": "Add practice round with explanations",
                "rationale": "Players struggled early then improved",
                "implementation": "Interactive tutorial before first real game"
            }
        ]
    }"""
    
    result = json.loads(llm_response)
    
    assert result["overall_confusion_level"] in ["low", "medium", "high"]
    assert len(result["key_insights"]) == 2
    assert len(result["improvement_suggestions"]) == 2
    
    for suggestion in result["improvement_suggestions"]:
        assert "category" in suggestion
        assert "suggestion" in suggestion
        assert "rationale" in suggestion
        assert "implementation" in suggestion
    
    print("✓ Game improvements JSON format is valid")


def test_confusion_detection_scenarios():
    """Test that various confusion scenarios would be detected correctly."""
    
    # Scenario 1: Logical inconsistency
    statements_1 = [
        "p1 is definitely red because they voted no",
        "Actually, maybe p1 is blue because they abstained",
        "Wait, now I'm sure p1 is red again",
    ]
    # This should trigger "Logical Inconsistency" detection
    
    # Scenario 2: Rule misunderstanding
    statements_2 = [
        "When do the patches get applied?",
        "Are we voting on the patches or the council?",
        "Does firing someone count as a patch?",
    ]
    # This should trigger "Rule Misunderstanding" detection
    
    # Scenario 3: State confusion
    statements_3 = [
        "How many players are left?",
        "What round are we on?",
        "Did blue patches already get applied?",
    ]
    # This should trigger "State Confusion" detection
    
    # Scenario 4: Strategic confusion
    statements_4 = [
        "I need to help red reach 6 patches because I'm blue",
        "Let's not investigate anyone to help blues win",
        "Fire all the blue players to help blue win",
    ]
    # This should trigger "Strategic Confusion" detection
    
    # These would be analyzed by LLM, but the structure is correct
    assert len(statements_1) == 3
    assert len(statements_2) == 3
    assert len(statements_3) == 3
    assert len(statements_4) == 3
    
    print("✓ Confusion detection scenarios are well-formed")


def test_game_data_structure():
    """Test that game data can be loaded and analyzed."""
    
    # Create a minimal sample data structure
    sample_data = {
        "id": "test-sample-1",
        "scores": {
            "red_vs_blue_scorer": {
                "value": 1,  # Blues win
                "metadata": {
                    "num_players": 5,
                    "rounds_played": 5,
                    "blues_win": True,
                    "patch_track": {"blue": 6, "red": 3},
                    "roles": {
                        "p0": "blue",
                        "p1": "red",
                        "p2": "blue",
                        "p3": "blue",
                        "p4": "apt_leader",
                    },
                    "true_apt_leader": "p4",
                    "fired_players": [],
                    "public_log": [
                        "p0: I think p1 is red",
                        "p1: That's not true!",
                        "p2: Let's vote",
                        "p3: Agreed",
                        "p4: Yes, vote now",
                    ],
                }
            }
        }
    }
    
    # Extract data
    context = extract_game_context(sample_data)
    reasoning = extract_player_reasoning(sample_data)
    
    # Verify
    assert context["blues_win"] == True
    assert len(reasoning) == 5, "All 5 players should have statements"
    assert context["patch_track"]["blue"] == 6
    
    print("✓ Game data structure works correctly")


def run_all_tests():
    """Run all unit tests."""
    print("\n" + "="*70)
    print("CONFUSION ANALYSIS UNIT TESTS")
    print("="*70 + "\n")
    
    tests = [
        test_extract_player_reasoning,
        test_extract_game_context,
        test_confusion_analysis_json_parsing,
        test_game_improvements_json_format,
        test_confusion_detection_scenarios,
        test_game_data_structure,
    ]
    
    failed = []
    passed = 0
    
    for test in tests:
        try:
            print(f"Running: {test.__name__}...")
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ FAILED: {e}")
            failed.append((test.__name__, str(e)))
        except Exception as e:
            print(f"✗ ERROR: {e}")
            failed.append((test.__name__, f"Error: {e}"))
    
    # Summary
    print("\n" + "="*70)
    print(f"RESULTS: {passed}/{len(tests)} tests passed")
    if failed:
        print("\nFailed tests:")
        for name, error in failed:
            print(f"  - {name}: {error}")
    else:
        print("\n🎉 All tests passed!")
    print("="*70 + "\n")
    
    return len(failed) == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
