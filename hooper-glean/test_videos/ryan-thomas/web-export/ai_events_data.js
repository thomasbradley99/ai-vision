const aiEventsData = [
  {
    "timestamp": 0,
    "type": "POSITION",
    "title": "Match starts with fighters standing",
    "description": "The BJJ training session begins with a handshake between DARK RASHGUARD and STRIPED RASHGUARD, both standing in a neutral position.",
    "attacker": null,
    "defender": null,
    "submission": false,
    "attempt": false,
    "perspectives": {
      "DARK RASHGUARD": {
        "quality": "good",
        "score": 75,
        "points": 0,
        "analysis": "Started in a neutral standing position, ready for the engagement.",
        "betterMove": null,
        "whyBad": null
      },
      "STRIPED RASHGUARD": {
        "quality": "good",
        "score": 75,
        "points": 0,
        "analysis": "Began the session respectfully, ready to initiate.",
        "betterMove": null,
        "whyBad": null
      }
    }
  },
  {
    "timestamp": 2,
    "type": "POSITION",
    "title": "STRIPED RASHGUARD pulls closed guard",
    "description": "STRIPED RASHGUARD initiates by pulling guard, securing a closed guard around DARK RASHGUARD.",
    "attacker": "STRIPED RASHGUARD",
    "defender": "DARK RASHGUARD",
    "submission": false,
    "attempt": false,
    "perspectives": {
      "DARK RASHGUARD": {
        "quality": "inaccuracy",
        "score": 65,
        "points": 0,
        "analysis": "Allowed STRIPED RASHGUARD to pull guard without significant resistance, ending up on the bottom.",
        "betterMove": "Prevent the guard pull by establishing grips, creating distance, or immediately attempting a pass as STRIPED RASHGUARD goes to the ground.",
        "whyBad": "Conceded the bottom position and lost the initiative early in the engagement."
      },
      "STRIPED RASHGUARD": {
        "quality": "excellent",
        "score": 85,
        "points": 0,
        "analysis": "Successfully initiated by pulling a tight closed guard, establishing a strong offensive position from the bottom.",
        "betterMove": null,
        "whyBad": null
      }
    }
  },
  {
    "timestamp": 30,
    "type": "POSITION",
    "title": "DARK RASHGUARD passes guard to side control",
    "description": "DARK RASHGUARD works to break the guard and pass, eventually stepping over STRIPED RASHGUARD's legs and establishing side control.",
    "attacker": "DARK RASHGUARD",
    "defender": "STRIPED RASHGUARD",
    "submission": false,
    "attempt": false,
    "perspectives": {
      "DARK RASHGUARD": {
        "quality": "excellent",
        "score": 90,
        "points": 3,
        "analysis": "Executed a successful guard pass, demonstrating good technique and pressure to achieve side control.",
        "betterMove": null,
        "whyBad": null
      },
      "STRIPED RASHGUARD": {
        "quality": "mistake",
        "score": 40,
        "points": 0,
        "analysis": "Failed to maintain guard retention and allowed DARK RASHGUARD to pass directly into side control.",
        "betterMove": "Utilize active frames, hip escapes, and guard recovery techniques to prevent the pass, or attempt a sweep during the pass attempt.",
        "whyBad": "Conceded a dominant position and points to the opponent due to insufficient guard defense."
      }
    }
  },
  {
    "timestamp": 38,
    "type": "POSITION",
    "title": "DARK RASHGUARD transitions to mount",
    "description": "From side control, DARK RASHGUARD then transitions to mount.",
    "attacker": "DARK RASHGUARD",
    "defender": "STRIPED RASHGUARD",
    "submission": false,
    "attempt": false,
    "perspectives": {
      "DARK RASHGUARD": {
        "quality": "excellent",
        "score": 95,
        "points": 4,
        "analysis": "Smoothly transitioned from side control to mount, increasing pressure and gaining significant positional advantage.",
        "betterMove": null,
        "whyBad": null
      },
      "STRIPED RASHGUARD": {
        "quality": "mistake",
        "score": 35,
        "points": 0,
        "analysis": "Allowed DARK RASHGUARD to transition from side control to mount without sufficient defense or escape attempts.",
        "betterMove": "Prioritize frames and hip escapes to prevent the mount transition, or initiate a scramble as the transition begins.",
        "whyBad": "Conceded a highly dominant position, making escape and defense much more difficult."
      }
    }
  },
  {
    "timestamp": 58,
    "type": "POSITION",
    "title": "DARK RASHGUARD dismounts to side control",
    "description": "After maintaining mount, DARK RASHGUARD dismounts to side control, setting up an arm-triangle choke.",
    "attacker": "DARK RASHGUARD",
    "defender": "STRIPED RASHGUARD",
    "submission": false,
    "attempt": false,
    "perspectives": {
      "DARK RASHGUARD": {
        "quality": "excellent",
        "score": 90,
        "points": 0,
        "analysis": "Demonstrated strategic thinking by dismounting from mount to side control to set up a specific submission.",
        "betterMove": null,
        "whyBad": null
      },
      "STRIPED RASHGUARD": {
        "quality": "inaccuracy",
        "score": 60,
        "points": 0,
        "analysis": "Did not capitalize on the dismount to create space or initiate an escape, remaining in a vulnerable position.",
        "betterMove": "Exploit the momentary shift in pressure during the dismount to create an opening for an escape or guard recovery.",
        "whyBad": "Remained passive, allowing DARK RASHGUARD to easily transition to another dominant position and set up a submission."
      }
    }
  },
  {
    "timestamp": 64,
    "type": "SUBMISSION",
    "title": "DARK RASHGUARD submits STRIPED RASHGUARD with arm-triangle choke",
    "description": "DARK RASHGUARD successfully applies the arm-triangle choke, and STRIPED RASHGUARD yields.",
    "attacker": "DARK RASHGUARD",
    "defender": "STRIPED RASHGUARD",
    "submission": true,
    "attempt": true,
    "perspectives": {
      "DARK RASHGUARD": {
        "quality": "excellent",
        "score": 100,
        "points": 10,
        "analysis": "Executed a flawless arm-triangle choke from side control, securing the submission.",
        "betterMove": null,
        "whyBad": null
      },
      "STRIPED RASHGUARD": {
        "quality": "blunder",
        "score": 10,
        "points": 0,
        "analysis": "Failed to defend against the arm-triangle choke and was forced to yield. Did not recognize the setup or apply sufficient defensive measures.",
        "betterMove": "Prioritize head and neck defense, create space, or escape before the choke is fully locked in.",
        "whyBad": "Conceded the submission, indicating a significant lapse in defensive technique and awareness."
      }
    }
  },
  {
    "timestamp": 70,
    "type": "POSITION",
    "title": "STRIPED RASHGUARD pulls closed guard (Round 2)",
    "description": "The athletes shake hands again, and STRIPED RASHGUARD pulls guard, establishing a closed guard position on DARK RASHGUARD.",
    "attacker": "STRIPED RASHGUARD",
    "defender": "DARK RASHGUARD",
    "submission": false,
    "attempt": false,
    "perspectives": {
      "DARK RASHGUARD": {
        "quality": "inaccuracy",
        "score": 65,
        "points": 0,
        "analysis": "Allowed STRIPED RASHGUARD to pull guard for the second time, giving up the initiative.",
        "betterMove": "Engage more actively from standing, either securing a dominant grip or initiating a takedown to avoid being pulled into guard.",
        "whyBad": "Repeatedly conceded the bottom position, which could allow STRIPED RASHGUARD to initiate offense."
      },
      "STRIPED RASHGUARD": {
        "quality": "excellent",
        "score": 85,
        "points": 0,
        "analysis": "Successfully re-established a strong closed guard, maintaining his preferred offensive starting position.",
        "betterMove": null,
        "whyBad": null
      }
    }
  },
  {
    "timestamp": 91,
    "type": "SWEEP",
    "title": "STRIPED RASHGUARD sweeps DARK RASHGUARD",
    "description": "STRIPED RASHGUARD uses his legs to execute a sweep, off-balancing and bringing DARK RASHGUARD to the mat, ending up on top.",
    "attacker": "STRIPED RASHGUARD",
    "defender": "DARK RASHGUARD",
    "submission": false,
    "attempt": true,
    "perspectives": {
      "DARK RASHGUARD": {
        "quality": "mistake",
        "score": 45,
        "points": 0,
        "analysis": "Failed to maintain balance and posture while attempting to break guard, resulting in a successful sweep by STRIPED RASHGUARD.",
        "betterMove": "Maintain strong posture, base, and balance, using hand placement and hip movement to defend against sweep attempts.",
        "whyBad": "Conceded a sweep and lost the top position, giving STRIPED RASHGUARD the advantage."
      },
      "STRIPED RASHGUARD": {
        "quality": "excellent",
        "score": 90,
        "points": 2,
        "analysis": "Executed an effective sweep from closed guard, demonstrating good timing and technique to reverse the position.",
        "betterMove": null,
        "whyBad": null
      }
    }
  },
  {
    "timestamp": 160,
    "type": "POSITION",
    "title": "Fighters standing, preparing for drill",
    "description": "Both athletes, DARK RASHGUARD and STRIPED RASHGUARD, stand facing each other in the training area, preparing for the drill.",
    "attacker": null,
    "defender": null,
    "submission": false,
    "attempt": false,
    "perspectives": {
      "DARK RASHGUARD": {
        "quality": "good",
        "score": 70,
        "points": 0,
        "analysis": "Stood ready and focused for the next sequence.",
        "betterMove": null,
        "whyBad": null
      },
      "STRIPED RASHGUARD": {
        "quality": "good",
        "score": 70,
        "points": 0,
        "analysis": "Prepared to initiate the next drilling sequence.",
        "betterMove": null,
        "whyBad": null
      }
    }
  },
  {
    "timestamp": 163,
    "type": "POSITION",
    "title": "STRIPED RASHGUARD establishes open guard",
    "description": "STRIPED RASHGUARD initiates the action by sitting down and pulling DARK RASHGUARD's leg to establish an open guard. DARK RASHGUARD remains standing.",
    "attacker": "STRIPED RASHGUARD",
    "defender": "DARK RASHGUARD",
    "submission": false,
    "attempt": false,
    "perspectives": {
      "DARK RASHGUARD": {
        "quality": "inaccuracy",
        "score": 60,
        "points": 0,
        "analysis": "Allowed STRIPED RASHGUARD to easily establish open guard from a seated position, indicating a reactive rather than proactive approach.",
        "betterMove": "Control STRIPED RASHGUARD's hips or legs immediately upon sitting to prevent a strong open guard setup.",
        "whyBad": "Conceded the initial offensive position to STRIPED RASHGUARD, allowing him to set the pace."
      },
      "STRIPED RASHGUARD": {
        "quality": "good",
        "score": 80,
        "points": 0,
        "analysis": "Effectively transitioned to an open guard, securing a grip and maintaining distance for offense.",
        "betterMove": null,
        "whyBad": null
      }
    }
  },
  {
    "timestamp": 169,
    "type": "SWEEP",
    "title": "STRIPED RASHGUARD sweeps and mounts DARK RASHGUARD",
    "description": "STRIPED RASHGUARD executes a sweep by elevating DARK RASHGUARD with their legs, causing DARK RASHGUARD to fall onto their back. STRIPED RASHGUARD then moves to a dominant mount position.",
    "attacker": "STRIPED RASHGUARD",
    "defender": "DARK RASHGUARD",
    "submission": false,
    "attempt": true,
    "perspectives": {
      "DARK RASHGUARD": {
        "quality": "mistake",
        "score": 40,
        "points": 0,
        "analysis": "Failed to defend against the sweep from open guard, leading to losing top control and immediately conceding mount.",
        "betterMove": "Maintain strong base and posture against the open guard, actively clearing hooks and preventing elevation.",
        "whyBad": "Conceded both a sweep and a dominant mount position in quick succession, resulting in significant points for STRIPED RASHGUARD."
      },
      "STRIPED RASHGUARD": {
        "quality": "excellent",
        "score": 95,
        "points": 6,
        "analysis": "Executed a highly effective sweep from open guard, demonstrating strength and technique, and immediately transitioned to a strong mount.",
        "betterMove": null,
        "whyBad": null
      }
    }
  },
  {
    "timestamp": 184,
    "type": "ESCAPE",
    "title": "DARK RASHGUARD escapes mount",
    "description": "From mount, DARK RASHGUARD successfully defends and escapes the mount position by turning and rolling out, regaining a brief top position before standing.",
    "attacker": "DARK RASHGUARD",
    "defender": "STRIPED RASHGUARD",
    "submission": false,
    "attempt": false,
    "perspectives": {
      "DARK RASHGUARD": {
        "quality": "good",
        "score": 80,
        "points": 0,
        "analysis": "Successfully escaped a dangerous mount position, demonstrating resilience and effective defensive technique.",
        "betterMove": null,
        "whyBad": null
      },
      "STRIPED RASHGUARD": {
        "quality": "inaccuracy",
        "score": 60,
        "points": 0,
        "analysis": "Failed to maintain mount control, allowing DARK RASHGUARD to escape and neutralize the position.",
        "betterMove": "Maintain tighter hip control, secure strong grips, and prevent bridging and rolling attempts from the bottom.",
        "whyBad": "Lost a dominant mount position, missing an opportunity to score or set up a submission."
      }
    }
  },
  {
    "timestamp": 187,
    "type": "POSITION",
    "title": "Fighters reset to standing",
    "description": "Following the mount escape, both athletes quickly stand up, acknowledge each other with a fist bump, and return to their neutral starting positions.",
    "attacker": null,
    "defender": null,
    "submission": false,
    "attempt": false,
    "perspectives": {
      "DARK RASHGUARD": {
        "quality": "good",
        "score": 75,
        "points": 0,
        "analysis": "Successfully reset after escaping a bad position.",
        "betterMove": null,
        "whyBad": null
      },
      "STRIPED RASHGUARD": {
        "quality": "good",
        "score": 75,
        "points": 0,
        "analysis": "Participated in the reset after losing mount.",
        "betterMove": null,
        "whyBad": null
      }
    }
  },
  {
    "timestamp": 193,
    "type": "POSITION",
    "title": "STRIPED RASHGUARD sits to open guard (Round 2)",
    "description": "STRIPED RASHGUARD again sits to guard, immediately securing a grip on DARK RASHGUARD's leg.",
    "attacker": "STRIPED RASHGUARD",
    "defender": "DARK RASHGUARD",
    "submission": false,
    "attempt": false,
    "perspectives": {
      "DARK RASHGUARD": {
        "quality": "inaccuracy",
        "score": 60,
        "points": 0,
        "analysis": "Allowed STRIPED RASHGUARD to establish open guard easily again, failing to learn from the previous exchange.",
        "betterMove": "Be more assertive in breaking grips and disengaging or initiating a pass as STRIPED RASHGUARD sits.",
        "whyBad": "Conceded the bottom offensive position to STRIPED RASHGUARD for the second time in this segment."
      },
      "STRIPED RASHGUARD": {
        "quality": "good",
        "score": 80,
        "points": 0,
        "analysis": "Successfully re-established open guard with a clear grip, preparing for offense.",
        "betterMove": null,
        "whyBad": null
      }
    }
  },
  {
    "timestamp": 201,
    "type": "SWEEP",
    "title": "STRIPED RASHGUARD sweeps and mounts DARK RASHGUARD (Round 2)",
    "description": "STRIPED RASHGUARD performs another successful sweep, taking DARK RASHGUARD to the mat and quickly transitioning to a mount position.",
    "attacker": "STRIPED RASHGUARD",
    "defender": "DARK RASHGUARD",
    "submission": false,
    "attempt": true,
    "perspectives": {
      "DARK RASHGUARD": {
        "quality": "mistake",
        "score": 40,
        "points": 0,
        "analysis": "Repeated the mistake of getting swept from open guard and immediately mounted, indicating a consistent defensive vulnerability.",
        "betterMove": "Address the fundamental issues in guard pass defense and sweep prevention from open guard.",
        "whyBad": "Conceded another sweep and mount, demonstrating a pattern of defensive failures in this scenario."
      },
      "STRIPED RASHGUARD": {
        "quality": "excellent",
        "score": 95,
        "points": 6,
        "analysis": "Executed another clean sweep from open guard, demonstrating consistency and effectiveness in his offensive game, and transitioning directly to mount.",
        "betterMove": null,
        "whyBad": null
      }
    }
  },
  {
    "timestamp": 224,
    "type": "ESCAPE",
    "title": "DARK RASHGUARD escapes mount (Round 2)",
    "description": "DARK RASHGUARD continues their escape efforts, successfully turning and maneuvering to dislodge STRIPED RASHGUARD, eventually ending up on top or standing.",
    "attacker": "DARK RASHGUARD",
    "defender": "STRIPED RASHGUARD",
    "submission": false,
    "attempt": false,
    "perspectives": {
      "DARK RASHGUARD": {
        "quality": "good",
        "score": 80,
        "points": 0,
        "analysis": "Successfully escaped mount for the second time, showing persistent defensive effort.",
        "betterMove": null,
        "whyBad": null
      },
      "STRIPED RASHGUARD": {
        "quality": "inaccuracy",
        "score": 60,
        "points": 0,
        "analysis": "Failed to maintain mount control, allowing DARK RASHGUARD to escape, missing another opportunity to consolidate the dominant position.",
        "betterMove": "Improve mount retention techniques, anticipating and shutting down escape attempts more effectively.",
        "whyBad": "Lost control of mount again, indicating a need to refine his mount stability."
      }
    }
  },
  {
    "timestamp": 234,
    "type": "POSITION",
    "title": "Fighters reset to standing",
    "description": "After DARK RASHGUARD's successful mount escape, both athletes stand up, acknowledge each other, and return to their neutral positions.",
    "attacker": null,
    "defender": null,
    "submission": false,
    "attempt": false,
    "perspectives": {
      "DARK RASHGUARD": {
        "quality": "good",
        "score": 75,
        "points": 0,
        "analysis": "Successfully reset after another mount escape.",
        "betterMove": null,
        "whyBad": null
      },
      "STRIPED RASHGUARD": {
        "quality": "good",
        "score": 75,
        "points": 0,
        "analysis": "Participated in the reset after mount was lost.",
        "betterMove": null,
        "whyBad": null
      }
    }
  },
  {
    "timestamp": 240,
    "type": "POSITION",
    "title": "STRIPED RASHGUARD in open guard, DARK RASHGUARD attempts pass",
    "description": "The round begins with STRIPED RASHGUARD lying on his back in an open guard position, with DARK RASHGUARD attempting to pass his guard.",
    "attacker": "DARK RASHGUARD",
    "defender": "STRIPED RASHGUARD",
    "submission": false,
    "attempt": true,
    "perspectives": {
      "DARK RASHGUARD": {
        "quality": "good",
        "score": 75,
        "points": 0,
        "analysis": "Started in an advantageous position, immediately working to pass STRIPED RASHGUARD's open guard.",
        "betterMove": null,
        "whyBad": null
      },
      "STRIPED RASHGUARD": {
        "quality": "good",
        "score": 75,
        "points": 0,
        "analysis": "Initiated from open guard, actively defending and setting up attacks against DARK RASHGUARD's pass attempts.",
        "betterMove": null,
        "whyBad": null
      }
    }
  },
  {
    "timestamp": 260,
    "type": "POSITION",
    "title": "DARK RASHGUARD establishes knee-on-belly",
    "description": "DARK RASHGUARD moves into a dominant position, eventually establishing a knee-on-belly.",
    "attacker": "DARK RASHGUARD",
    "defender": "STRIPED RASHGUARD",
    "submission": false,
    "attempt": false,
    "perspectives": {
      "DARK RASHGUARD": {
        "quality": "excellent",
        "score": 85,
        "points": 2,
        "analysis": "Successfully passed the guard and established a strong knee-on-belly position, demonstrating effective pressure and control.",
        "betterMove": null,
        "whyBad": null
      },
      "STRIPED RASHGUARD": {
        "quality": "mistake",
        "score": 40,
        "points": 0,
        "analysis": "Allowed DARK RASHGUARD to establish knee-on-belly, indicating a lapse in guard retention and framing.",
        "betterMove": "Focus on hip escapes and framing to prevent the knee from settling on the belly, or counter-attack during the transition.",
        "whyBad": "Conceded a dominant position and points, making escape more difficult and vulnerable to further attacks."
      }
    }
  },
  {
    "timestamp": 269,
    "type": "POSITION",
    "title": "DARK RASHGUARD transitions to side control",
    "description": "DARK RASHGUARD transitions smoothly from knee-on-belly to securing side control over STRIPED RASHGUARD.",
    "attacker": "DARK RASHGUARD",
    "defender": "STRIPED RASHGUARD",
    "submission": false,
    "attempt": false,
    "perspectives": {
      "DARK RASHGUARD": {
        "quality": "good",
        "score": 80,
        "points": 0,
        "analysis": "Maintained dominant pressure by smoothly transitioning from knee-on-belly to side control.",
        "betterMove": null,
        "whyBad": null
      },
      "STRIPED RASHGUARD": {
        "quality": "inaccuracy",
        "score": 60,
        "points": 0,
        "analysis": "Failed to create an escape opportunity during the transition from knee-on-belly to side control.",
        "betterMove": "Exploit the momentary shift in weight during the transition to bridge and escape or recover guard.",
        "whyBad": "Remained in a defensive, non-scoring position, allowing DARK RASHGUARD to maintain control."
      }
    }
  },
  {
    "timestamp": 271,
    "type": "POSITION",
    "title": "Fighters reset to standing",
    "description": "Both athletes immediately release the position, sit up, stand, and acknowledge each other before preparing for the next round.",
    "attacker": null,
    "defender": null,
    "submission": false,
    "attempt": false,
    "perspectives": {
      "DARK RASHGUARD": {
        "quality": "good",
        "score": 75,
        "points": 0,
        "analysis": "Successfully completed the drill and reset.",
        "betterMove": null,
        "whyBad": null
      },
      "STRIPED RASHGUARD": {
        "quality": "good",
        "score": 75,
        "points": 0,
        "analysis": "Participated in the reset.",
        "betterMove": null,
        "whyBad": null
      }
    }
  },
  {
    "timestamp": 274,
    "type": "POSITION",
    "title": "STRIPED RASHGUARD in open guard, DARK RASHGUARD attempts pass (Round 2)",
    "description": "The next round starts with STRIPED RASHGUARD on his back in open guard, and DARK RASHGUARD positioned over him, immediately working for a guard pass.",
    "attacker": "DARK RASHGUARD",
    "defender": "STRIPED RASHGUARD",
    "submission": false,
    "attempt": true,
    "perspectives": {
      "DARK RASHGUARD": {
        "quality": "good",
        "score": 75,
        "points": 0,
        "analysis": "Initiated the round from a top position, immediately engaging in guard passing.",
        "betterMove": null,
        "whyBad": null
      },
      "STRIPED RASHGUARD": {
        "quality": "good",
        "score": 75,
        "points": 0,
        "analysis": "Started in open guard, actively defending and preparing for offensive actions.",
        "betterMove": null,
        "whyBad": null
      }
    }
  },
  {
    "timestamp": 307,
    "type": "POSITION",
    "title": "DARK RASHGUARD passes guard to mount",
    "description": "From leg control, DARK RASHGUARD efficiently transitions and secures a dominant mount position over STRIPED RASHGUARD.",
    "attacker": "DARK RASHGUARD",
    "defender": "STRIPED RASHGUARD",
    "submission": false,
    "attempt": false,
    "perspectives": {
      "DARK RASHGUARD": {
        "quality": "excellent",
        "score": 95,
        "points": 7,
        "analysis": "Executed a successful guard pass and immediate transition to mount, demonstrating efficient and effective offensive strategy.",
        "betterMove": null,
        "whyBad": null
      },
      "STRIPED RASHGUARD": {
        "quality": "mistake",
        "score": 35,
        "points": 0,
        "analysis": "Failed to prevent the guard pass and mount, indicating a breakdown in guard retention and mount defense.",
        "betterMove": "Focus on strong frames, hip movement, and guard recovery to prevent the pass and subsequent mount.",
        "whyBad": "Conceded both a guard pass and mount, giving DARK RASHGUARD significant points and a dominant position."
      }
    }
  },
  {
    "timestamp": 309,
    "type": "POSITION",
    "title": "Fighters reset to standing",
    "description": "The athletes release the position, sit up, stand, and perform a fist bump to acknowledge the completion of the technique.",
    "attacker": null,
    "defender": null,
    "submission": false,
    "attempt": false,
    "perspectives": {
      "DARK RASHGUARD": {
        "quality": "good",
        "score": 75,
        "points": 0,
        "analysis": "Successfully completed the drill and reset.",
        "betterMove": null,
        "whyBad": null
      },
      "STRIPED RASHGUARD": {
        "quality": "good",
        "score": 75,
        "points": 0,
        "analysis": "Participated in the reset.",
        "betterMove": null,
        "whyBad": null
      }
    }
  },
  {
    "timestamp": 314,
    "type": "POSITION",
    "title": "STRIPED RASHGUARD in open guard, DARK RASHGUARD attempts pass (Round 3)",
    "description": "In the final round, STRIPED RASHGUARD is again on his back with legs up in open guard, and DARK RASHGUARD is kneeling over him, initiating another guard passing sequence.",
    "attacker": "DARK RASHGUARD",
    "defender": "STRIPED RASHGUARD",
    "submission": false,
    "attempt": true,
    "perspectives": {
      "DARK RASHGUARD": {
        "quality": "good",
        "score": 75,
        "points": 0,
        "analysis": "Initiated the round from a top position, immediately working for a guard pass.",
        "betterMove": null,
        "whyBad": null
      },
      "STRIPED RASHGUARD": {
        "quality": "good",
        "score": 75,
        "points": 0,
        "analysis": "Started in open guard, actively defending and preparing for offensive actions.",
        "betterMove": null,
        "whyBad": null
      }
    }
  },
  {
    "timestamp": 318,
    "type": "POSITION",
    "title": "DARK RASHGUARD passes guard to side control (Round 3)",
    "description": "DARK RASHGUARD successfully passes STRIPED RASHGUARD's guard, establishing side control with significant pressure.",
    "attacker": "DARK RASHGUARD",
    "defender": "STRIPED RASHGUARD",
    "submission": false,
    "attempt": false,
    "perspectives": {
      "DARK RASHGUARD": {
        "quality": "excellent",
        "score": 90,
        "points": 3,
        "analysis": "Executed a successful guard pass to side control, maintaining consistent pressure and control.",
        "betterMove": null,
        "whyBad": null
      },
      "STRIPED RASHGUARD": {
        "quality": "mistake",
        "score": 40,
        "points": 0,
        "analysis": "Failed to prevent the guard pass, allowing DARK RASHGUARD to establish side control.",
        "betterMove": "Improve guard retention and framing to prevent DARK RASHGUARD from getting past the legs.",
        "whyBad": "Conceded another guard pass and dominant position, indicating a recurring vulnerability."
      }
    }
  },
  {
    "timestamp": 320,
    "type": "POSITION",
    "title": "STRIPED RASHGUARD controls DARK RASHGUARD's leg from standing",
    "description": "The clip begins with DARK RASHGUARD lying on his back. STRIPED RASHGUARD is standing over him, leaning down and actively controlling DARK RASHGUARD's right leg with his left hand.",
    "attacker": "STRIPED RASHGUARD",
    "defender": "DARK RASHGUARD",
    "submission": false,
    "attempt": true,
    "perspectives": {
      "DARK RASHGUARD": {
        "quality": "inaccuracy",
        "score": 60,
        "points": 0,
        "analysis": "Started from a compromised position, on his back with his leg being controlled by STRIPED RASHGUARD.",
        "betterMove": "Prevent falling into this controlled bottom position, or immediately work to break the grip and recover guard.",
        "whyBad": "Allowed STRIPED RASHGUARD to establish control from the outset of this segment."
      },
      "STRIPED RASHGUARD": {
        "quality": "good",
        "score": 80,
        "points": 0,
        "analysis": "Started in a dominant standing position over DARK RASHGUARD, actively controlling his leg, indicating potential for a pass or submission setup.",
        "betterMove": null,
        "whyBad": null
      }
    }
  },
  {
    "timestamp": 323,
    "type": "POSITION",
    "title": "Fighters reset to standing",
    "description": "Immediately after, STRIPED RASHGUARD releases his grip, and DARK RASHGUARD quickly sits up. Both athletes then touch hands, performing a standard reset to conclude the round.",
    "attacker": null,
    "defender": null,
    "submission": false,
    "attempt": false,
    "perspectives": {
      "DARK RASHGUARD": {
        "quality": "good",
        "score": 75,
        "points": 0,
        "analysis": "Successfully reset after a brief period of being controlled.",
        "betterMove": null,
        "whyBad": null
      },
      "STRIPED RASHGUARD": {
        "quality": "good",
        "score": 75,
        "points": 0,
        "analysis": "Participated in the reset.",
        "betterMove": null,
        "whyBad": null
      }
    }
  },
  {
    "timestamp": 324,
    "type": "POSITION",
    "title": "Fighters standing and talking",
    "description": "Following the reset, DARK RASHGUARD sits on the mat with his legs bent and turned to his left, while STRIPED RASHGUARD stands over him. They engage in a conversation, with STRIPED RASHGUARD gesturing and seemingly providing instruction or feedback to DARK RASHGUARD.",
    "attacker": null,
    "defender": null,
    "submission": false,
    "attempt": false,
    "perspectives": {
      "DARK RASHGUARD": {
        "quality": "good",
        "score": 70,
        "points": 0,
        "analysis": "Engaged in conversation and feedback, demonstrating receptiveness to learning.",
        "betterMove": null,
        "whyBad": null
      },
      "STRIPED RASHGUARD": {
        "quality": "good",
        "score": 70,
        "points": 0,
        "analysis": "Provided instruction or feedback, indicating a coaching or mentoring role.",
        "betterMove": null,
        "whyBad": null
      }
    }
  }
];
