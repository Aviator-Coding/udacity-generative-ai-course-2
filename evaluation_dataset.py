"""
RAGAS Evaluation Demo - Prints actual scores for all test scenarios
Shows question, answer, contexts, and scores for full transparency
"""

from ragas_evaluator import evaluate_response_quality


def print_evaluation(name, question, answer, contexts):
    """Run evaluation and print all inputs and scores."""
    print(f"\n{'='*80}")
    print(f"TEST: {name}")
    print(f"{'='*80}")

    print(f"\nQUESTION:")
    print(f"  {question}")

    print(f"\nANSWER:")
    print(f"  {answer}")

    print(f"\nCONTEXTS ({len(contexts)} chunks):")
    for i, ctx in enumerate(contexts, 1):
        print(f"  [{i}] {ctx}")

    result = evaluate_response_quality(
        question=question,
        answer=answer,
        contexts=contexts
    )

    print(f"\nSCORES:")
    if "error" in result:
        print(f"  Error: {result['error']}")
    else:
        print(f"  Faithfulness:       {result['faithfulness']:.4f}")
        print(f"  Response Relevancy: {result['response_relevancy']:.4f}")
        print(f"  Context Precision:  {result['context_precision']:.4f}")

    return result


def main():
    print("\n" + "="*80)
    print("RAGAS EVALUATION DEMO - Apollo 13 Technical Transcript")
    print("="*80)

    # =========================================================================
    # ERROR HANDLING SCENARIOS
    # =========================================================================
    print("\n\n" + "#"*80)
    print("# ERROR HANDLING SCENARIOS")
    print("#"*80)

    print_evaluation(
        name="Empty Contexts",
        question="What problem occurred on Apollo 13?",
        answer="The crew reported a main B bus undervolt after hearing a loud bang.",
        contexts=[]
    )

    # =========================================================================
    # HIGH FAITHFULNESS SCENARIOS
    # =========================================================================
    print("\n\n" + "#"*80)
    print("# HIGH FAITHFULNESS SCENARIOS (answers grounded in context)")
    print("#"*80)

    print_evaluation(
        name="High Faithfulness - Direct Quote",
        question="What did Commander Lovell report to Houston?",
        answer="CDR 'Houston, we've had a problem. We've had a MAIN B BUS UNDERVOLT.'",
        contexts=[
            "LMP Okay, Houston - -",
            "CDR I believe we've had a problem here.",
            "CC This is Houston. Say again, please.",
            "CDR Houston, we've had a problem. We've had a MAIN B BUS UNDERVOLT.",
            "CC Roger. MAIN B UNDERVOLT.",
            "CC Okay, stand by, 13. We're looking at it.",
            "LMP Okay. Right now, Houston, the voltage is - is looking good. And we had a pretty large bang associated with the CAUTION AND WARNING there. And as I recall, MAIN B was the one that had had an amp spike on it once before."
        ]
    )

    print_evaluation(
        name="High Faithfulness - Voltage Readings",
        question="What voltage readings were reported during the Apollo 13 crisis?",
        answer="The crew reported a MAIN B BUS UNDERVOLT initially, then later a MAIN BUS A UNDERVOLT was also showing. They observed 25 volts on MAIN A and used BUS TIE BAT AC to help the voltage.",
        contexts=[
            "CDR Houston, we've had a problem. We've had a MAIN B BUS UNDERVOLT.",
            "LMP Okay. Right now, Houston, the voltage is - is looking good.",
            "CDR Yes. We got a MAIN BUS A UNDERVOLT now, too, showing.",
            "CC MAIN A UNDERVOLT.",
            "CMP Okay. Let me give you some reading ... in the interim to help MAIN A voltage. Jack. I've got BUS TIE AC on.",
            "CMP In the interim, to help out MAIN A voltage, I've got MAIN BUS TIE BAT AC on. Or would you rather accept the 25 volts we are seeing on MAIN A?"
        ]
    )

    print_evaluation(
        name="High Faithfulness - Launch Sequence",
        question="What was the sequence of events during Apollo 13's launch?",
        answer="The clock started running, followed by the yaw program, tower clearance, then yaw complete and roll program. Houston gave GO at 30 seconds, and the crew confirmed roll complete and pitching.",
        contexts=[
            "CDR The clock is running.",
            "CMP Okay. P11, Jim.",
            "CDR Yaw program.",
            "CMP Clear the tower.",
            "CDR Yaw complete. Roll program.",
            "CC Houston, Roger. Roll.",
            "CC 13, Houston. GO at 30 seconds.",
            "CDR Roll complete, and we are pitching.",
            "CC Roger that. Stand by for mode I Bravo.",
            "CC MARK.",
            "CC I Bravo.",
            "CMP I Bravo.",
            "CDR RCS COMMAND.",
            "CC 13, Houston. GO at 1. We show the cabin relieving.",
            "CDR 13; Roger."
        ]
    )


    # =========================================================================
    # LOW FAITHFULNESS SCENARIOS
    # =========================================================================
    print("\n\n" + "#"*80)
    print("# LOW FAITHFULNESS SCENARIOS (hallucinated content)")
    print("#"*80)

    print_evaluation(
        name="Low Faithfulness - Hallucinated Details",
        question="What happened during the Apollo 13 incident?",
        answer="The oxygen tank exploded due to a faulty thermostat, causing severe damage to the heat shield. The crew had to use the lunar module as a lifeboat and perform a manual course correction around the moon.",
        contexts=[
            "LMP Okay, Houston - -",
            "CDR I believe we've had a problem here.",
            "CC This is Houston. Say again, please.",
            "CDR Houston, we've had a problem. We've had a MAIN B BUS UNDERVOLT.",
            "CC Roger. MAIN B UNDERVOLT.",
            "CC Okay, stand by, 13. We're looking at it.",
            "LMP Okay. Right now, Houston, the voltage is - is looking good. And we had a pretty large bang associated with the CAUTION AND WARNING there. And as I recall, MAIN B was the one that had had an amp spike on it once before.",
            "CC Roger, Fred."
        ]
    )

    print_evaluation(
        name="Low Faithfulness - Fabricated Quotes",
        question="What did the crew say after the problem?",
        answer="Commander Lovell declared 'This is a catastrophic emergency, we're losing all power!' and Houston responded 'Prepare for immediate abort procedures, this is a code red situation!'",
        contexts=[
            "LMP Okay, Houston - -",
            "CDR I believe we've had a problem here.",
            "CC This is Houston. Say again, please.",
            "CDR Houston, we've had a problem. We've had a MAIN B BUS UNDERVOLT.",
            "CC Roger. MAIN B UNDERVOLT.",
            "CC Okay, stand by, 13. We're looking at it.",
            "LMP Okay. Right now, Houston, the voltage is - is looking good. And we had a pretty large bang associated with the CAUTION AND WARNING there. And as I recall, MAIN B was the one that had had an amp spike on it once before.",
            "CC Roger, Fred."
        ]
    )

    print_evaluation(
        name="Low Faithfulness - Wrong Attribution (CDR said it, not LMP)",
        question="Who reported the problem to Houston?",
        answer="The Lunar Module Pilot CDR first reported 'Houston, we've had a problem. We've had a MAIN B BUS UNDERVOLT.' Houston initially missed the call and had to ask him to repeat it.",
        contexts=[
            "LMP Okay, Houston - -",
            "CDR I believe we've had a problem here.",
            "CC This is Houston. Say again, please.",
            "CDR Houston, we've had a problem. We've had a MAIN B BUS UNDERVOLT.",
            "CC Roger. MAIN B UNDERVOLT.",
            "CC Okay, stand by, 13. We're looking at it.",
            "LMP Okay. Right now, Houston, the voltage is - is looking good. And we had a pretty large bang associated with the CAUTION AND WARNING there. And as I recall, MAIN B was the one that had had an amp spike on it once before.",
            "CC Roger, Fred."
        ]
    )

    # =========================================================================
    # HIGH RESPONSE RELEVANCY SCENARIOS
    # =========================================================================
    print("\n\n" + "#"*80)
    print("# HIGH RESPONSE RELEVANCY SCENARIOS (answers address the question)")
    print("#"*80)

    print_evaluation(
        name="High Relevancy - Direct Answer",
        question="When did the Apollo 13 crew first report a problem to Houston?",
        answer="The crew first reported a problem when CDR said 'I believe we've had a problem here.' This was followed by a more detailed report when he specified it was a MAIN B BUS UNDERVOLT.",
        contexts=[
            "LMP Okay, Houston - -",
            "CDR I believe we've had a problem here.",
            "CC This is Houston. Say again, please.",
            "CDR Houston, we've had a problem. We've had a MAIN B BUS UNDERVOLT.",
            "CC Roger. MAIN B UNDERVOLT.",
            "CC Okay, stand by, 13. We're looking at it.",
            "LMP Okay. Right now, Houston, the voltage is - is looking good. And we had a pretty large bang associated with the CAUTION AND WARNING there. And as I recall, MAIN B was the one that had had an amp spike on it once before.",
            "CC Roger, Fred."
        ]
    )

    print_evaluation(
        name="High Relevancy - Comprehensive Answer",
        question="What electrical failures were observed during the Apollo 13 crisis?",
        answer="Two main electrical failures were observed: First, a MAIN B BUS UNDERVOLT, followed by a MAIN BUS A UNDERVOLT. The crew attempted to stabilize MAIN A voltage using BUS TIE BAT AC, observing 25 volts on MAIN A.",
        contexts=[
            "CDR Houston, we've had a problem. We've had a MAIN B BUS UNDERVOLT.",
            "LMP Okay. Right now, Houston, the voltage is - is looking good.",
            "CDR Yes. We got a MAIN BUS A UNDERVOLT now, too, showing.",
            "CC MAIN A UNDERVOLT.",
            "CMP Okay. Let me give you some reading ... in the interim to help MAIN A voltage. Jack. I've got BUS TIE AC on.",
            "CMP In the interim, to help out MAIN A voltage, I've got MAIN BUS TIE BAT AC on. Or would you rather accept the 25 volts we are seeing on MAIN A?"
        ]
    )

    # =========================================================================
    # LOW RESPONSE RELEVANCY SCENARIOS
    # =========================================================================
    print("\n\n" + "#"*80)
    print("# LOW RESPONSE RELEVANCY SCENARIOS (answers don't address question)")
    print("#"*80)

    print_evaluation(
        name="Low Relevancy - Off-Topic Answer",
        question="What voltage readings were observed during the crisis?",
        answer="Apollo 13 launched on April 11, 1970, from Kennedy Space Center in Florida. The Saturn V rocket used for the mission was the same type used for the Apollo 11 moon landing. The crew consisted of three astronauts.",
        contexts=[
            "CDR Houston, we've had a problem. We've had a MAIN B BUS UNDERVOLT.",
            "LMP Okay. Right now, Houston, the voltage is - is looking good.",
            "CDR Yes. We got a MAIN BUS A UNDERVOLT now, too, showing.",
            "CC MAIN A UNDERVOLT.",
            "CMP Okay. Let me give you some reading ... in the interim to help MAIN A voltage. Jack. I've got BUS TIE AC on.",
            "CMP In the interim, to help out MAIN A voltage, I've got MAIN BUS TIE BAT AC on. Or would you rather accept the 25 volts we are seeing on MAIN A?"
        ]
    )

    print_evaluation(
        name="Low Relevancy - Tangential (describes crew, not Houston instructions)",
        question="What instructions did Houston give to the Apollo 13 crew after the problem report?",
        answer="Commander Lovell reported that they had a problem and specified it was a MAIN B BUS UNDERVOLT. The Lunar Module Pilot mentioned hearing a loud bang and recalled that MAIN B had an amp spike before.",
        contexts=[
            "LMP Okay, Houston - -",
            "CDR I believe we've had a problem here.",
            "CC This is Houston. Say again, please.",
            "CDR Houston, we've had a problem. We've had a MAIN B BUS UNDERVOLT.",
            "CC Roger. MAIN B UNDERVOLT.",
            "CC Okay, stand by, 13. We're looking at it.",
            "LMP Okay. Right now, Houston, the voltage is - is looking good. And we had a pretty large bang associated with the CAUTION AND WARNING there. And as I recall, MAIN B was the one that had had an amp spike on it once before.",
            "CC Roger, Fred."
        ]
    )

    # =========================================================================
    # HIGH CONTEXT PRECISION SCENARIOS
    # =========================================================================
    print("\n\n" + "#"*80)
    print("# HIGH CONTEXT PRECISION SCENARIOS (relevant context chunks)")
    print("#"*80)

    print_evaluation(
        name="High Context Precision - All Relevant Technical Context",
        question="What electrical issues were reported on Apollo 13?",
        answer="Multiple bus undervolt conditions were reported, affecting both MAIN A and MAIN B buses, with attempts to stabilize using bus ties.",
        contexts=[
            "CDR Houston, we've had a problem. We've had a MAIN B BUS UNDERVOLT.",
            "LMP Okay. Right now, Houston, the voltage is - is looking good.",
            "CDR Yes. We got a MAIN BUS A UNDERVOLT now, too, showing.",
            "CC MAIN A UNDERVOLT.",
            "CMP Okay. Let me give you some reading ... in the interim to help MAIN A voltage. Jack. I've got BUS TIE AC on.",
            "CMP In the interim, to help out MAIN A voltage, I've got MAIN BUS TIE BAT AC on. Or would you rather accept the 25 volts we are seeing on MAIN A?"
        ]
    )

    # =========================================================================
    # LOW CONTEXT PRECISION SCENARIOS
    # =========================================================================
    print("\n\n" + "#"*80)
    print("# LOW CONTEXT PRECISION SCENARIOS (irrelevant context chunks)")
    print("#"*80)

    print_evaluation(
        name="Low Context Precision - Irrelevant TLI/Water Context",
        question="What voltage problems occurred on Apollo 13?",
        answer="The spacecraft experienced MAIN B BUS UNDERVOLT followed by MAIN A UNDERVOLT.",
        contexts=[
            "CC Apollo 13, Houston. We have your TLI PAD ready to copy.",
            "CMP Stand by. Ready to copy.",
            "CC Roger. TLI, SPS/G&N: 36691, minus 157, plus 163.",
            "CC NOUN 33, 002 50 3799 plus 10780 minus 04529 plus 04125.",
            "CC Roll 180, pitch 343, yaw 010.",
            "LMP Houston, I can see some water floating around in the cabin.",
            "CC Copy that. Any estimate on the quantity?",
            "LMP Looks like a few drops. No concern at the moment.",
            "CC Roger. Keep us posted if it increases."
        ]
    )

    print_evaluation(
        name="Low Context Precision - Wrong Time (splashdown for crisis question)",
        question="What happened during the Apollo 13 crisis?",
        answer="Commander Lovell reported 'Houston, we've had a problem. We've had a MAIN B BUS UNDERVOLT.'",
        contexts=[
            "R Apollo 13 and Recovery passing through 1000 feet.",
            "IWO Iwo Jima; Roger.",
            "R Through 500 feet.",
            "S-1 Swim 1 on station.",
            "S-2 - - 2 is on station.",
            "P-1 Photo 1's on station. Photo 1 observes splashdown at this time.",
            "P-1 Photo-1. Splashdown at this time. The three chutes are displaced. They're in the water."
        ]
    )

    # =========================================================================
    # COMBINED SCENARIOS
    # =========================================================================
    print("\n\n" + "#"*80)
    print("# COMBINED SCENARIOS")
    print("#"*80)

    print_evaluation(
        name="PERFECT SCENARIO - Faithful, Relevant, Precise Context",
        question="What exactly did the Apollo 13 crew say when they reported the problem to Houston?",
        answer="Commander Lovell first said 'I believe we've had a problem here.' Houston asked him to repeat, and he clarified: 'Houston, we've had a problem. We've had a MAIN B BUS UNDERVOLT.' Houston acknowledged with 'Roger. MAIN B UNDERVOLT.'",
        contexts=[
            "LMP Okay, Houston - -",
            "CDR I believe we've had a problem here.",
            "CC This is Houston. Say again, please.",
            "CDR Houston, we've had a problem. We've had a MAIN B BUS UNDERVOLT.",
            "CC Roger. MAIN B UNDERVOLT.",
            "CC Okay, stand by, 13. We're looking at it.",
            "LMP Okay. Right now, Houston, the voltage is - is looking good. And we had a pretty large bang associated with the CAUTION AND WARNING there. And as I recall, MAIN B was the one that had had an amp spike on it once before.",
            "CC Roger, Fred."
        ]
    )

    print_evaluation(
        name="POOR SCENARIO - Hallucinated, Irrelevant Context",
        question="What voltage problems occurred during the Apollo 13 crisis?",
        answer="The spacecraft suffered from thermal runaway in the fuel cells due to improper insulation. Mission control had to perform emergency procedures to prevent complete power failure. The astronauts manually rewired the backup systems.",
        contexts=[
            "CC Apollo 13, Houston. We have your TLI PAD ready to copy.",
            "CMP Stand by. Ready to copy.",
            "CC Roger. TLI, SPS/G&N: 36691, minus 157, plus 163.",
            "CC NOUN 33, 002 50 3799 plus 10780 minus 04529 plus 04125.",
            "CC Roll 180, pitch 343, yaw 010."
        ]
    )

    # =========================================================================
    # EDGE CASES
    # =========================================================================
    print("\n\n" + "#"*80)
    print("# EDGE CASES")
    print("#"*80)

    print_evaluation(
        name="Edge Case - Minimal Context (single chunk)",
        question="What did Commander Lovell say?",
        answer="He said they had a problem.",
        contexts=["CDR Houston, we've had a problem."]
    )

    print_evaluation(
        name="Edge Case - Technical Acronyms (PGNCS, RCS, CSM)",
        question="What systems showed problems according to the acronyms mentioned?",
        answer="The PGNCS (Primary guidance, navigation, and control system) showed a warning light. The RCS (Reaction control system) HELIUM indicators showed barber pole readings on positions B and D.",
        contexts=[
            "CDR And, Houston, we had a RESTART on our computer and we had a PGNCS light and the RESTART RESET.",
            "CDR Okay. And we're looking at our SERVICE MODULE RCS HELIUM 1. We have B is barber poled and D is barber poled.",
            "CDR RCS COMMAND.",
            "PGNCS - Primary guidance, navigation, and control system (CM)",
            "RCS - Reaction control system",
            "CSM - Command and service module"
        ]
    )

    print("\n\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
