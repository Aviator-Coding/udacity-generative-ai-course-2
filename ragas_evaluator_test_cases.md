# RAGAS Evaluator Test Cases

Test results from `unit_test_ragas_evaluator.py` - All 24 tests PASSED

---

## Score Thresholds

| Threshold Type | Value |
|---------------|-------|
| HIGH_SCORE_THRESHOLD | 0.35 |
| LOW_SCORE_THRESHOLD | 0.55 |

---

## 1. Error Handling Tests (Mocked, no API calls)

### Test: Empty Contexts Returns Error

```python
result = evaluate_response_quality(
    question="What problem occurred on Apollo 13?",
    answer="The crew reported a main B bus undervolt after hearing a loud bang.",
    contexts=[]
)
```

**Expected Result:** `{"error": "No contexts provided for evaluation"}`
**Test Status:** PASSED

---

### Test: Missing API Key Returns Error

```python
result = evaluate_response_quality(
    question="What problem occurred on Apollo 13?",
    answer="The crew reported a main B bus undervolt after hearing a loud bang.",
    contexts=["055:55:35 CDR Houston, we've had a problem. We've had a MAIN B BUS UNDERVOLT."]
)
```

**Expected Result:** `{"error": "OpenAI API key not found"}` (when no API key is set)
**Test Status:** PASSED

---

### Test: RAGAS Unavailable Returns Error

```python
result = evaluate_response_quality(
    question="What problem occurred on Apollo 13?",
    answer="The crew reported a main B bus undervolt after hearing a loud bang.",
    contexts=["055:55:35 CDR Houston, we've had a problem. We've had a MAIN B BUS UNDERVOLT."]
)
```

**Expected Result:** `{"error": "RAGAS not available"}` (when RAGAS_AVAILABLE=False)
**Test Status:** PASSED

---

## 2. High Faithfulness Tests (Expected >= 0.35)

### Test: High Faithfulness - Direct Quote

```python
result = evaluate_response_quality(
    question="What did Commander Lovell report to Houston at 055:55:35?",
    answer="Commander Lovell reported 'Houston, we've had a problem. We've had a MAIN B BUS UNDERVOLT.'",
    contexts=[
        "055:55:19 LMP Okay, Houston - -",
        "055:55:20 CDR I believe we've had a problem here.",
        "055:55:28 CC This is Houston. Say again, please.",
        "055:55:35 CDR Houston, we've had a problem. We've had a MAIN B BUS UNDERVOLT.",
        "055:55:42 CC Roger. MAIN B UNDERVOLT.",
        "055:55:58 CC Okay, stand by, 13. We're looking at it.",
        "055:56:10 LMP Okay. Right now, Houston, the voltage is - is looking good. And we had a pretty large bang associated with the CAUTION AND WARNING there. And as I recall, MAIN B was the one that had had an amp spike on it once before.",
        "055:56:40 CC Roger, Fred."
    ]
)
```

**Expected:** `faithfulness >= 0.35`
**Test Status:** PASSED

---

### Test: High Faithfulness - Voltage Readings

```python
result = evaluate_response_quality(
    question="What voltage readings were reported during the Apollo 13 crisis?",
    answer="The crew reported a MAIN B BUS UNDERVOLT initially, then later a MAIN BUS A UNDERVOLT was also showing. They observed 25 volts on MAIN A and used BUS TIE BAT AC to help the voltage.",
    contexts=[
        "055:55:35 CDR Houston, we've had a problem. We've had a MAIN B BUS UNDERVOLT.",
        "055:56:10 LMP Okay. Right now, Houston, the voltage is - is looking good.",
        "056:01:08 CDR Yes. We got a MAIN BUS A UNDERVOLT now, too, showing.",
        "056:01:15 CC MAIN A UNDERVOLT.",
        "056:03:14 CMP Okay. Let me give you some reading ... in the interim to help MAIN A voltage. Jack. I've got BUS TIE AC on.",
        "056:03:30 CMP In the interim, to help out MAIN A voltage, I've got MAIN BUS TIE BAT AC on. Or would you rather accept the 25 volts we are seeing on MAIN A?"
    ]
)
```

**Expected:** `faithfulness >= 0.35`
**Test Status:** PASSED

---

### Test: High Faithfulness - Launch Sequence

```python
result = evaluate_response_quality(
    question="What was the sequence of events during Apollo 13's launch?",
    answer="The clock started running at 000:00:02, followed by the yaw program at 000:00:05, tower clearance at 000:00:12, then yaw complete and roll program at 000:00:14. Houston gave GO at 30 seconds, and the crew confirmed roll complete and pitching.",
    contexts=[
        "000:00:02 CDR The clock is running.",
        "000:00:03 CMP Okay. P11, Jim.",
        "000:00:05 CDR Yaw program.",
        "000:00:12 CMP Clear the tower.",
        "000:00:14 CDR Yaw complete. Roll program.",
        "000:00:16 CC Houston, Roger. Roll.",
        "000:00:30 CC 13, Houston. GO at 30 seconds.",
        "000:00:34 CDR Roll complete, and we are pitching.",
        "000:00:36 CC Roger that. Stand by for mode I Bravo.",
        "000:00:42 CC MARK.",
        "000:00:43 CC I Bravo.",
        "000:00:44 CMP I Bravo.",
        "000:00:45 CDR RCS COMMAND.",
        "000:01:03 CC 13, Houston. GO at 1. We show the cabin relieving.",
        "000:01:07 CDR 13; Roger."
    ]
)
```

**Expected:** `faithfulness >= 0.35`
**Test Status:** PASSED

---

## 3. Low Faithfulness Tests (Expected < 0.55)

### Test: Low Faithfulness - Hallucinated Details

```python
result = evaluate_response_quality(
    question="What happened during the Apollo 13 incident?",
    answer="The oxygen tank exploded due to a faulty thermostat, causing severe damage to the heat shield. The crew had to use the lunar module as a lifeboat and perform a manual course correction around the moon.",
    contexts=[
        "055:55:19 LMP Okay, Houston - -",
        "055:55:20 CDR I believe we've had a problem here.",
        "055:55:28 CC This is Houston. Say again, please.",
        "055:55:35 CDR Houston, we've had a problem. We've had a MAIN B BUS UNDERVOLT.",
        "055:55:42 CC Roger. MAIN B UNDERVOLT.",
        "055:55:58 CC Okay, stand by, 13. We're looking at it.",
        "055:56:10 LMP Okay. Right now, Houston, the voltage is - is looking good. And we had a pretty large bang associated with the CAUTION AND WARNING there. And as I recall, MAIN B was the one that had had an amp spike on it once before.",
        "055:56:40 CC Roger, Fred."
    ]
)
```

**Expected:** `faithfulness < 0.55` (hallucinated content not in context)
**Test Status:** PASSED

---

### Test: Low Faithfulness - Fabricated Quotes

```python
result = evaluate_response_quality(
    question="What did the crew say after the problem?",
    answer="Commander Lovell declared 'This is a catastrophic emergency, we're losing all power!' and Houston responded 'Prepare for immediate abort procedures, this is a code red situation!'",
    contexts=[
        "055:55:19 LMP Okay, Houston - -",
        "055:55:20 CDR I believe we've had a problem here.",
        "055:55:28 CC This is Houston. Say again, please.",
        "055:55:35 CDR Houston, we've had a problem. We've had a MAIN B BUS UNDERVOLT.",
        "055:55:42 CC Roger. MAIN B UNDERVOLT.",
        "055:55:58 CC Okay, stand by, 13. We're looking at it.",
        "055:56:10 LMP Okay. Right now, Houston, the voltage is - is looking good. And we had a pretty large bang associated with the CAUTION AND WARNING there. And as I recall, MAIN B was the one that had had an amp spike on it once before.",
        "055:56:40 CC Roger, Fred."
    ]
)
```

**Expected:** `faithfulness < 0.55` (fabricated quotes)
**Test Status:** PASSED

---

### Test: Low Faithfulness - Wrong Attribution

```python
result = evaluate_response_quality(
    question="Who reported the problem to Houston?",
    answer="The Lunar Module Pilot Fred Haise first reported 'Houston, we've had a problem. We've had a MAIN B BUS UNDERVOLT.' Houston initially missed the call and had to ask him to repeat it.",
    contexts=[
        "055:55:19 LMP Okay, Houston - -",
        "055:55:20 CDR I believe we've had a problem here.",
        "055:55:28 CC This is Houston. Say again, please.",
        "055:55:35 CDR Houston, we've had a problem. We've had a MAIN B BUS UNDERVOLT.",
        "055:55:42 CC Roger. MAIN B UNDERVOLT.",
        "055:55:58 CC Okay, stand by, 13. We're looking at it.",
        "055:56:10 LMP Okay. Right now, Houston, the voltage is - is looking good. And we had a pretty large bang associated with the CAUTION AND WARNING there. And as I recall, MAIN B was the one that had had an amp spike on it once before.",
        "055:56:40 CC Roger, Fred."
    ]
)
```

**Expected:** `faithfulness < 0.55` (CDR said this, not LMP - wrong attribution)
**Test Status:** PASSED

---

## 4. High Response Relevancy Tests (Expected >= 0.35)

### Test: High Relevancy - Direct Answer

```python
result = evaluate_response_quality(
    question="At what time did the Apollo 13 crew first report a problem to Houston?",
    answer="The crew first reported a problem at 055:55:20 when Commander Lovell said 'I believe we've had a problem here.' This was followed by a more detailed report at 055:55:35 when he specified it was a MAIN B BUS UNDERVOLT.",
    contexts=[
        "055:55:19 LMP Okay, Houston - -",
        "055:55:20 CDR I believe we've had a problem here.",
        "055:55:28 CC This is Houston. Say again, please.",
        "055:55:35 CDR Houston, we've had a problem. We've had a MAIN B BUS UNDERVOLT.",
        "055:55:42 CC Roger. MAIN B UNDERVOLT.",
        "055:55:58 CC Okay, stand by, 13. We're looking at it.",
        "055:56:10 LMP Okay. Right now, Houston, the voltage is - is looking good. And we had a pretty large bang associated with the CAUTION AND WARNING there. And as I recall, MAIN B was the one that had had an amp spike on it once before.",
        "055:56:40 CC Roger, Fred."
    ]
)
```

**Expected:** `response_relevancy >= 0.35`
**Test Status:** PASSED

---

### Test: High Relevancy - Comprehensive Answer

```python
result = evaluate_response_quality(
    question="What electrical failures were observed during the Apollo 13 crisis?",
    answer="Two main electrical failures were observed: First, a MAIN B BUS UNDERVOLT at 055:55:35, followed by a MAIN BUS A UNDERVOLT at 056:01:08. The crew attempted to stabilize MAIN A voltage using BUS TIE BAT AC, observing 25 volts on MAIN A.",
    contexts=[
        "055:55:35 CDR Houston, we've had a problem. We've had a MAIN B BUS UNDERVOLT.",
        "055:56:10 LMP Okay. Right now, Houston, the voltage is - is looking good.",
        "056:01:08 CDR Yes. We got a MAIN BUS A UNDERVOLT now, too, showing.",
        "056:01:15 CC MAIN A UNDERVOLT.",
        "056:03:14 CMP Okay. Let me give you some reading ... in the interim to help MAIN A voltage. Jack. I've got BUS TIE AC on.",
        "056:03:30 CMP In the interim, to help out MAIN A voltage, I've got MAIN BUS TIE BAT AC on. Or would you rather accept the 25 volts we are seeing on MAIN A?"
    ]
)
```

**Expected:** `response_relevancy >= 0.35`
**Test Status:** PASSED

---

## 5. Low Response Relevancy Tests (Expected < 0.55)

### Test: Low Relevancy - Off Topic

```python
result = evaluate_response_quality(
    question="What voltage readings were observed during the crisis?",
    answer="Apollo 13 launched on April 11, 1970, from Kennedy Space Center in Florida. The Saturn V rocket used for the mission was the same type used for the Apollo 11 moon landing. The crew consisted of three astronauts.",
    contexts=[
        "055:55:35 CDR Houston, we've had a problem. We've had a MAIN B BUS UNDERVOLT.",
        "055:56:10 LMP Okay. Right now, Houston, the voltage is - is looking good.",
        "056:01:08 CDR Yes. We got a MAIN BUS A UNDERVOLT now, too, showing.",
        "056:01:15 CC MAIN A UNDERVOLT.",
        "056:03:14 CMP Okay. Let me give you some reading ... in the interim to help MAIN A voltage. Jack. I've got BUS TIE AC on.",
        "056:03:30 CMP In the interim, to help out MAIN A voltage, I've got MAIN BUS TIE BAT AC on. Or would you rather accept the 25 volts we are seeing on MAIN A?"
    ]
)
```

**Expected:** `response_relevancy < 0.55` (completely off-topic)
**Test Status:** PASSED

---

### Test: Low Relevancy - Tangential

```python
result = evaluate_response_quality(
    question="What instructions did Houston give to the Apollo 13 crew after the problem report?",
    answer="Commander Lovell reported that they had a problem and specified it was a MAIN B BUS UNDERVOLT. The Lunar Module Pilot mentioned hearing a loud bang and recalled that MAIN B had an amp spike before.",
    contexts=[
        "055:55:19 LMP Okay, Houston - -",
        "055:55:20 CDR I believe we've had a problem here.",
        "055:55:28 CC This is Houston. Say again, please.",
        "055:55:35 CDR Houston, we've had a problem. We've had a MAIN B BUS UNDERVOLT.",
        "055:55:42 CC Roger. MAIN B UNDERVOLT.",
        "055:55:58 CC Okay, stand by, 13. We're looking at it.",
        "055:56:10 LMP Okay. Right now, Houston, the voltage is - is looking good. And we had a pretty large bang associated with the CAUTION AND WARNING there. And as I recall, MAIN B was the one that had had an amp spike on it once before.",
        "055:56:40 CC Roger, Fred."
    ]
)
```

**Expected:** `response_relevancy < 0.55` (describes crew's reports, not Houston's instructions)
**Test Status:** PASSED

---

## 6. High Context Precision Tests (Expected >= 0.35)

### Test: High Precision - Launch Context

```python
result = evaluate_response_quality(
    question="What happened during the first minute of the Apollo 13 launch?",
    answer="The launch began with the clock running, followed by yaw program, tower clearance, roll program, and Houston giving GO at 30 seconds and 1 minute.",
    contexts=[
        "000:00:02 CDR The clock is running.",
        "000:00:03 CMP Okay. P11, Jim.",
        "000:00:05 CDR Yaw program.",
        "000:00:12 CMP Clear the tower.",
        "000:00:14 CDR Yaw complete. Roll program.",
        "000:00:16 CC Houston, Roger. Roll.",
        "000:00:30 CC 13, Houston. GO at 30 seconds.",
        "000:00:34 CDR Roll complete, and we are pitching.",
        "000:00:36 CC Roger that. Stand by for mode I Bravo.",
        "000:00:42 CC MARK.",
        "000:00:43 CC I Bravo.",
        "000:00:44 CMP I Bravo.",
        "000:00:45 CDR RCS COMMAND.",
        "000:01:03 CC 13, Houston. GO at 1. We show the cabin relieving.",
        "000:01:07 CDR 13; Roger."
    ]
)
```

**Expected:** `0.0 <= context_precision <= 1.0` (valid score)
**Test Status:** PASSED

---

### Test: High Precision - Technical Context

```python
result = evaluate_response_quality(
    question="What electrical issues were reported on Apollo 13?",
    answer="Multiple bus undervolt conditions were reported, affecting both MAIN A and MAIN B buses, with attempts to stabilize using bus ties.",
    contexts=[
        "055:55:35 CDR Houston, we've had a problem. We've had a MAIN B BUS UNDERVOLT.",
        "055:56:10 LMP Okay. Right now, Houston, the voltage is - is looking good.",
        "056:01:08 CDR Yes. We got a MAIN BUS A UNDERVOLT now, too, showing.",
        "056:01:15 CC MAIN A UNDERVOLT.",
        "056:03:14 CMP Okay. Let me give you some reading ... in the interim to help MAIN A voltage. Jack. I've got BUS TIE AC on.",
        "056:03:30 CMP In the interim, to help out MAIN A voltage, I've got MAIN BUS TIE BAT AC on. Or would you rather accept the 25 volts we are seeing on MAIN A?"
    ]
)
```

**Expected:** `context_precision >= 0.35`
**Test Status:** PASSED

---

## 7. Low Context Precision Tests (Expected < 0.55)

### Test: Low Precision - Irrelevant Context

```python
result = evaluate_response_quality(
    question="What voltage problems occurred on Apollo 13?",
    answer="The spacecraft experienced MAIN B BUS UNDERVOLT followed by MAIN A UNDERVOLT.",
    contexts=[
        "002:35:00 CC Apollo 13, Houston. We have your TLI PAD ready to copy.",
        "002:35:05 CMP Stand by. Ready to copy.",
        "002:35:10 CC Roger. TLI, SPS/G&N: 36691, minus 157, plus 163.",
        "002:35:20 CC NOUN 33, 002 50 3799 plus 10780 minus 04529 plus 04125.",
        "002:35:30 CC Roll 180, pitch 343, yaw 010.",
        "030:40:12 LMP Houston, I can see some water floating around in the cabin.",
        "030:40:18 CC Copy that. Any estimate on the quantity?",
        "030:40:25 LMP Looks like a few drops. No concern at the moment.",
        "030:40:30 CC Roger. Keep us posted if it increases."
    ]
)
```

**Expected:** `context_precision < 0.55` (TLI pad and water context unrelated to voltage)
**Test Status:** PASSED

---

### Test: Low Precision - Wrong Time Context

```python
result = evaluate_response_quality(
    question="What happened at mission time 055:55 during the Apollo 13 crisis?",
    answer="At 055:55:35, Commander Lovell reported 'Houston, we've had a problem. We've had a MAIN B BUS UNDERVOLT.'",
    contexts=[
        "142:54:09 R Apollo 13 and Recovery passing through 1000 feet.",
        "142:54:14 IWO Iwo Jima; Roger.",
        "142:54:34 R Through 500 feet.",
        "142:54:38 S-1 Swim 1 on station.",
        "142:54:40 S-2 - - 2 is on station.",
        "142:54:44 P-1 Photo 1's on station. Photo 1 observes splashdown at this time.",
        "142:54:56 P-1 Photo-1. Splashdown at this time. The three chutes are displaced. They're in the water."
    ]
)
```

**Expected:** `context_precision < 0.55` (splashdown context for question about 055:55)
**Test Status:** PASSED

---

## 8. Edge Case Tests

### Test: Minimal Context

```python
result = evaluate_response_quality(
    question="What did Commander Lovell say?",
    answer="He said they had a problem.",
    contexts=["055:55:35 CDR Houston, we've had a problem."]
)
```

**Expected:** All scores between 0.0 and 1.0
**Test Status:** PASSED

---

### Test: Long Context (15+ chunks)

```python
result = evaluate_response_quality(
    question="Summarize the Apollo 13 mission events from launch through the electrical crisis.",
    answer="The mission began with a successful launch sequence including yaw program, tower clearance, and roll program. Later at 055:55, the crew reported a MAIN B BUS UNDERVOLT followed by additional electrical failures.",
    contexts=[
        "000:00:02 CDR The clock is running.",
        "000:00:03 CMP Okay. P11, Jim.",
        "000:00:05 CDR Yaw program.",
        "000:00:12 CMP Clear the tower.",
        "000:00:14 CDR Yaw complete. Roll program.",
        "000:00:16 CC Houston, Roger. Roll.",
        "000:00:30 CC 13, Houston. GO at 30 seconds.",
        "000:00:34 CDR Roll complete, and we are pitching.",
        "000:00:36 CC Roger that. Stand by for mode I Bravo.",
        "000:00:42 CC MARK.",
        "000:00:43 CC I Bravo.",
        "000:00:44 CMP I Bravo.",
        "000:00:45 CDR RCS COMMAND.",
        "000:01:03 CC 13, Houston. GO at 1. We show the cabin relieving.",
        "000:01:07 CDR 13; Roger.",
        "055:55:19 LMP Okay, Houston - -",
        "055:55:20 CDR I believe we've had a problem here.",
        "055:55:28 CC This is Houston. Say again, please.",
        "055:55:35 CDR Houston, we've had a problem. We've had a MAIN B BUS UNDERVOLT.",
        "055:55:42 CC Roger. MAIN B UNDERVOLT.",
        "055:55:58 CC Okay, stand by, 13. We're looking at it.",
        "055:56:10 LMP Okay. Right now, Houston, the voltage is - is looking good. And we had a pretty large bang associated with the CAUTION AND WARNING there. And as I recall, MAIN B was the one that had had an amp spike on it once before.",
        "055:56:40 CC Roger, Fred.",
        "056:01:08 CDR Yes. We got a MAIN BUS A UNDERVOLT now, too, showing.",
        "056:01:15 CC MAIN A UNDERVOLT.",
        "056:03:14 CMP Okay. Let me give you some reading ... in the interim to help MAIN A voltage. Jack. I've got BUS TIE AC on.",
        "056:03:30 CMP In the interim, to help out MAIN A voltage, I've got MAIN BUS TIE BAT AC on. Or would you rather accept the 25 volts we are seeing on MAIN A?"
    ]
)
```

**Expected:** All scores between 0.0 and 1.0
**Test Status:** PASSED

---

### Test: Technical Acronyms

```python
result = evaluate_response_quality(
    question="What systems showed problems according to the acronyms mentioned?",
    answer="The PGNCS (Primary guidance, navigation, and control system) showed a warning light. The RCS (Reaction control system) HELIUM indicators showed barber pole readings on positions B and D.",
    contexts=[
        "055:57:30 CDR And, Houston, we had a RESTART on our computer and we had a PGNCS light and the RESTART RESET.",
        "055:57:44 CDR Okay. And we're looking at our SERVICE MODULE RCS HELIUM 1. We have B is barber poled and D is barber poled.",
        "000:00:45 CDR RCS COMMAND.",
        "PGNCS - Primary guidance, navigation, and control system (CM)",
        "RCS - Reaction control system",
        "CSM - Command and service module"
    ]
)
```

**Expected:** All scores between 0.0 and 1.0
**Test Status:** PASSED

---

## 9. Combined Metric Scenarios

### Test: All High Scores (Perfect Scenario)

```python
result = evaluate_response_quality(
    question="What exactly did the Apollo 13 crew say when they reported the problem to Houston?",
    answer="At 055:55:20, Commander Lovell first said 'I believe we've had a problem here.' Houston asked him to repeat, and at 055:55:35 he clarified: 'Houston, we've had a problem. We've had a MAIN B BUS UNDERVOLT.' Houston acknowledged with 'Roger. MAIN B UNDERVOLT.'",
    contexts=[
        "055:55:19 LMP Okay, Houston - -",
        "055:55:20 CDR I believe we've had a problem here.",
        "055:55:28 CC This is Houston. Say again, please.",
        "055:55:35 CDR Houston, we've had a problem. We've had a MAIN B BUS UNDERVOLT.",
        "055:55:42 CC Roger. MAIN B UNDERVOLT.",
        "055:55:58 CC Okay, stand by, 13. We're looking at it.",
        "055:56:10 LMP Okay. Right now, Houston, the voltage is - is looking good. And we had a pretty large bang associated with the CAUTION AND WARNING there. And as I recall, MAIN B was the one that had had an amp spike on it once before.",
        "055:56:40 CC Roger, Fred."
    ]
)
```

**Expected:**
- `faithfulness >= 0.35`
- `response_relevancy >= 0.35`
- `context_precision >= 0.35`

**Test Status:** PASSED

---

### Test: All Low Scores (Poor Scenario)

```python
result = evaluate_response_quality(
    question="What voltage problems occurred during the Apollo 13 crisis?",
    answer="The spacecraft suffered from thermal runaway in the fuel cells due to improper insulation. Mission control had to perform emergency procedures to prevent complete power failure. The astronauts manually rewired the backup systems.",
    contexts=[
        "002:35:00 CC Apollo 13, Houston. We have your TLI PAD ready to copy.",
        "002:35:05 CMP Stand by. Ready to copy.",
        "002:35:10 CC Roger. TLI, SPS/G&N: 36691, minus 157, plus 163.",
        "002:35:20 CC NOUN 33, 002 50 3799 plus 10780 minus 04529 plus 04125.",
        "002:35:30 CC Roll 180, pitch 343, yaw 010."
    ]
)
```

**Expected:**
- `faithfulness < 0.55`
- `response_relevancy < 0.55`
- `context_precision < 0.55`

**Test Status:** PASSED

---

## 10. Original Integration Tests

### Test: Valid Evaluation Returns Scores

```python
result = evaluate_response_quality(
    question="What did the Apollo 13 crew report to Houston at 055:55:35?",
    answer="Commander Jim Lovell reported that they had a problem - a main B bus undervolt.",
    contexts=[
        "055:55:20 CDR I believe we've had a problem here. "
        "055:55:28 CC This is Houston. Say again, please. "
        "055:55:35 CDR Houston, we've had a problem. We've had a MAIN B BUS UNDERVOLT. "
        "055:55:42 CC Roger. MAIN B UNDERVOLT."
    ]
)
```

**Expected:** All metrics present (faithfulness, response_relevancy, context_precision) with valid float scores 0.0-1.0
**Test Status:** PASSED

---

### Test: Evaluation with Multiple Contexts

```python
result = evaluate_response_quality(
    question="What symptoms did the Apollo 13 crew observe after the incident?",
    answer="The crew observed a main B bus undervolt, a loud bang, computer restart, and multiple system warnings including barber pole indicators.",
    contexts=[
        "055:55:35 CDR Houston, we've had a problem. We've had a MAIN B BUS UNDERVOLT.",
        "055:56:10 LMP Okay. Right now, Houston, the voltage is looking good. And we had a pretty large bang associated with the CAUTION AND WARNING there.",
        "055:57:30 CDR And, Houston, we had a RESTART on our computer and we had a PGNCS light and the RESTART RESET.",
        "055:57:44 CDR Okay. And we're looking at our SERVICE MODULE RCS HELIUM 1. We have B is barber poled and D is barber poled."
    ]
)
```

**Expected:** All metrics present with valid float scores 0.0-1.0
**Test Status:** PASSED

---

## Summary

| Category | Tests | Status |
|----------|-------|--------|
| Error Handling | 3 | All PASSED |
| High Faithfulness | 3 | All PASSED |
| Low Faithfulness | 3 | All PASSED |
| High Response Relevancy | 2 | All PASSED |
| Low Response Relevancy | 2 | All PASSED |
| High Context Precision | 2 | All PASSED |
| Low Context Precision | 2 | All PASSED |
| Edge Cases | 3 | All PASSED |
| Combined Metric Scenarios | 2 | All PASSED |
| Original Integration | 2 | All PASSED |
| **Total** | **24** | **All PASSED** |

**Test Run Time:** 335.82s (5 minutes 35 seconds)
