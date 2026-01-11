from src.predict import GrievancePredictor

predictor = GrievancePredictor()

test_cases = [
    ("Lekhpal is asking for money.", "Revenue & Disaster Management"),
    ("Tehsildar refused to sign my caste certificate.", "Revenue & Disaster Management"),
    ("My chakbandi process is stuck.", "Revenue & Disaster Management"),
    ("Dirty water is coming from the tap.", "Sanitation & Water (Jal Nigam)"),
]

print("Running Verification Tests...")
for text, expected_category in test_cases:
    cat, prio, conf = predictor.predict(text)
    print(f"\nInput: {text}")
    print(f"Predicted: {cat}")
    print(f"Expected: {expected_category}")
    
    if cat == expected_category:
        print("✅ PASS")
    else:
        print(f"❌ FAIL (Got {cat})")
