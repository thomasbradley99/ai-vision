// AI-detected BJJ events
const aiEventsData = [
  {
    "timestamp": 110,
    "type": "SUBMISSION",
    "title": "Arm-Triangle Choke",
    "description": "LIGHT GI PURPLE BELT submits DARK GI WHITE BELT with Arm-Triangle Choke",
    "attacker": "LIGHT GI PURPLE BELT",
    "defender": "DARK GI WHITE BELT",
    "submission": true,
    "attempt": false,
    "confidence": "high",
    "method": "break-based detection",
    "time_to_break": 5,
    "perspectives": {
      "LIGHT GI PURPLE BELT": {
        "quality": "excellent",
        "score": 100,
        "points": 10,
        "analysis": "Successfully executed Arm-Triangle Choke, demonstrating technical proficiency and control.",
        "betterMove": null,
        "whyBad": null
      },
      "DARK GI WHITE BELT": {
        "quality": "blunder",
        "score": 10,
        "points": 0,
        "analysis": "Was submitted by Arm-Triangle Choke, failing to defend effectively.",
        "betterMove": "Should have defended the Arm-Triangle Choke more actively or escaped the position earlier.",
        "whyBad": "Allowed the submission to be completed, resulting in a tap."
      }
    }
  },
  {
    "timestamp": 301,
    "type": "SUBMISSION",
    "title": "Kimura",
    "description": "LIGHT GI PURPLE BELT submits DARK GI WHITE BELT with Kimura",
    "attacker": "LIGHT GI PURPLE BELT",
    "defender": "DARK GI WHITE BELT",
    "submission": true,
    "attempt": false,
    "confidence": "high",
    "method": "break-based detection",
    "time_to_break": 9,
    "perspectives": {
      "LIGHT GI PURPLE BELT": {
        "quality": "excellent",
        "score": 100,
        "points": 10,
        "analysis": "Successfully executed Kimura, demonstrating technical proficiency and control.",
        "betterMove": null,
        "whyBad": null
      },
      "DARK GI WHITE BELT": {
        "quality": "blunder",
        "score": 10,
        "points": 0,
        "analysis": "Was submitted by Kimura, failing to defend effectively.",
        "betterMove": "Should have defended the Kimura more actively or escaped the position earlier.",
        "whyBad": "Allowed the submission to be completed, resulting in a tap."
      }
    }
  }
];

// Fighter profiles
const fighterProfiles = {
  "fighter1": {
    "short_name": "LIGHT GI PURPLE BELT",
    "belt_rank": "Purple belt (with occasional black stripe/tip indicating a degree), signifying advanced BJJ proficiency.",
    "skill_level": "Advanced/Instructor. Demonstrates methodical, controlled, and stable grappling. Often takes a dominant top position, applies pressure, and maintains control. Sometimes described in an instructional role, or calmly countering/defending.",
    "clothing_detailed": "Predominantly light-colored gi (off-white, light grey, cream, light heather grey, muted grey-green/sage green). Sometimes features purple trim on the collar/lapel. Gi fit is standard, neither excessively baggy nor tight, showing natural wear. Often has a prominent circular patch on the left shoulder (e.g., yellow/gold background with a dark green/black design, or dark background with lighter emblem). Rashguard is highly variable; sometimes a bright orange long-sleeve rashguard is visible, or bright red. Often, no rashguard is visible. Distinctive bright orange wristband on the right wrist.",
    "physical_features": "Age: Late 40s to early 60s. Body Type: Average to stocky, solid, sometimes described as a 'dad-bod' physique, but athletic for his age. Possesses a strong core and broad shoulders. Hair: Short, salt-and-pepper, predominantly grey, often wavy, with visible thinning or receding hairline. Darker eyebrows. Facial Hair: Mostly clean-shaven; occasionally has a short grey beard or heavy stubble. Skin Tone: Fair to light-medium olive, sometimes described as Mediterranean, Hispanic, or Caucasian. Distinguishing Marks: Prominent nose. No tattoos or scars visible. Demeanor: Calm, relaxed, attentive, focused, often with a friendly smile or serious, composed expression, sometimes taking on an instructional role.",
    "fighting_style": "Methodical and controlled, preferring dominant top positions or stable defensive guards. Emphasizes pressure and established technique over explosive movements. Shows high situational awareness, consistent with an instructor or highly experienced practitioner."
  },
  "fighter2": {
    "short_name": "DARK GI WHITE BELT",
    "belt_rank": "White belt, indicating a beginner or foundational level practitioner. No stripes are consistently visible.",
    "skill_level": "Beginner to early-stage practitioner. Exhibits active and engaged participation, showing determination and focus. Often works from defensive or transitional positions, or attempts guard passes/sweeps, sometimes relying on athletic ability. Shows potential for dynamic and explosive movements.",
    "clothing_detailed": "Consistently dark-colored gi (deep navy blue, dark charcoal grey, appears almost black in some lighting). Gi fit is typically standard to athletic, sometimes slightly loose or baggy, and appears in good condition. Gi pants in one instance have a lighter grey/white vertical panel/trim on the outside seam. Occasionally a small rectangular red fabric tag/patch on the lower right side of the gi jacket. Rashguard is variable: often a dark-colored (black, dark blue, or dark red) rashguard is visible. Sometimes a light grey or white rashguard. Sometimes no rashguard, or appears bare-chested under gi.",
    "physical_features": "Age: Consistently younger, ranging from late teens to mid-30s. Body Type: Athletic and muscular build, often described as lean to solid. Possesses visible upper body strength, broad shoulders, and a strong jawline. Hair: Short, dark brown or black hair, often neatly trimmed. Can be light brown/dark blonde, sometimes styled upward, tousled or messy. No balding. Facial Hair: Predominantly clean-shaven; occasionally has short dark beard or heavy stubble. Skin Tone: Light to medium, often described as Caucasian. Distinguishing Marks: No visible tattoos, scars, or glasses. Demeanor: Focused, determined, active, engaged, energetic.",
    "fighting_style": "Active, dynamic, and often aggressive in exchanges. Uses strength and athleticism to attempt guard passes, sweeps, or escapes. Shows strong engagement from both top and bottom positions, typical of a motivated beginner in a learning environment."
  },
  "format": "Gi training"
};

// Position timeline
const positionTimeline = {};

// Fighter stats
const fighterStats = {
  "LIGHT GI PURPLE BELT": {
    "submissions": 2,
    "total_points": 20,
    "blunders": 0,
    "mistakes": 0,
    "inaccuracies": 0
  },
  "DARK GI WHITE BELT": {
    "submissions": 0,
    "total_points": 0,
    "blunders": 2,
    "mistakes": 0,
    "inaccuracies": 0
  }
};

// Match summary
const matchSummary = {
  "final_scores": {
    "LIGHT GI PURPLE BELT": 20,
    "DARK GI WHITE BELT": 0
  },
  "submissions": {
    "LIGHT GI PURPLE BELT": 2,
    "DARK GI WHITE BELT": 0
  },
  "method": "break-based detection"
};
