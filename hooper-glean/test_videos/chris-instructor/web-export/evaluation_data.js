// Evaluation results
const evaluationData = {
  "ground_truth": [
    {
      "timestamp": 110,
      "type": "arm-triangle choke",
      "notes": "1:50 - LIGHT GI PURPLE BELT applies arm-triangle choke from mount, completes at 1:55"
    },
    {
      "timestamp": 306,
      "type": "kimura",
      "notes": "5:06 - LIGHT GI PURPLE BELT secures kimura on left arm from side control, releases at 5:10"
    }
  ],
  "detected": [
    {
      "timestamp": 110,
      "type": "Arm-Triangle Choke",
      "attacker": "LIGHT GI PURPLE BELT",
      "defender": "DARK GI WHITE BELT"
    },
    {
      "timestamp": 301,
      "type": "Kimura",
      "attacker": "LIGHT GI PURPLE BELT",
      "defender": "DARK GI WHITE BELT"
    }
  ],
  "matches": [
    {
      "detected": {
        "timestamp": 110,
        "type": "Arm-Triangle Choke",
        "attacker": "LIGHT GI PURPLE BELT",
        "defender": "DARK GI WHITE BELT"
      },
      "ground_truth": {
        "timestamp": 110,
        "type": "arm-triangle choke",
        "notes": "1:50 - LIGHT GI PURPLE BELT applies arm-triangle choke from mount, completes at 1:55"
      },
      "time_diff": 0
    },
    {
      "detected": {
        "timestamp": 301,
        "type": "Kimura",
        "attacker": "LIGHT GI PURPLE BELT",
        "defender": "DARK GI WHITE BELT"
      },
      "ground_truth": {
        "timestamp": 306,
        "type": "kimura",
        "notes": "5:06 - LIGHT GI PURPLE BELT secures kimura on left arm from side control, releases at 5:10"
      },
      "time_diff": -5
    }
  ],
  "false_positives": [],
  "false_negatives": [],
  "metrics": {
    "true_positives": 2,
    "false_positives": 0,
    "false_negatives": 0,
    "precision": 1.0,
    "recall": 1.0,
    "f1_score": 1.0
  }
};
