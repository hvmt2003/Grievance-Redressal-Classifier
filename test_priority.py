from src.predict import GrievancePredictor
p = GrievancePredictor()

test_cases = [
    # Intended Low Priority
    "The park bench in Block C is a bit dusty on the side in Delhi.",
    "Speed post EM123456789IN is showing In Transit for 2 days in Mumbai.", 
    "The escalator at platform 4 of Pune station was moving slightly slow today.",
    "There is some construction debris left on the side of the road in Sector 15.",
    
    # Intended Medium
    "I want to change my broadband plan to a cheaper one in Bangalore.",
    "My atm card is about to expire next month in Chennai.",
    
    # Intended High
    "Urgent fire in the building at Connaught Place!",
    "Train derailed near Kanpur."
]

print(f"{'Text':<80} | {'Pred Priority'}")
print("-" * 100)
for t in test_cases:
    cat, prio = p.predict(t)
    print(f"{t[:80]:<80} | {prio}")
