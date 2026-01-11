from src.predict import GrievancePredictor
import pandas as pd

def validate():
    print("Loading UP Grievance Model...")
    p = GrievancePredictor()
    
    test_cases = [
        # Urban vs Rural Context
        ("Entire village of Rampur is in dark for 4 days. Transformer burnt.", "Rural", "Electricity (UPPCL)", "High"),
        ("Huge pothole in Hazratganj market causing traffic jam.", "Urban", "Roads (PWD/NHAI)", "High"),
        ("Pradhan has not released money for MNREGA wages in Block Malihabad.", "Rural", "Rural Development (Panchayati Raj)", "High"),
        ("Chain snatching incident near Noida Sector 18 metro station.", "Urban", "Police (Home Dept)", "High"),
        
        # Priority Balance Test
        ("Street light pole looking tilted in Unnao.", "Low Priority", "Electricity (UPPCL)", "Low"),
        ("Request to fogging machine for mosquitoes in Gomti Nagar.", "Low Priority", "Sanitation & Water (Jal Nigam/Nagar Nigam)", "Low"),
        ("Passport verification pending at Hazratganj thana for 2 weeks.", "Low Priority", "Police (Home Dept)", "Low"),
        ("Enquiry about vaccination schedule for babies in Basti.", "Low Priority", "Health (Medical)", "Low"),
        
        # Medium Priority
        ("Garbage truck not coming to Unnao Civil Lines area.", "Medium Priority", "Sanitation & Water (Jal Nigam/Nagar Nigam)", "Medium"),
        ("Potholes needs repair in Barabanki market before monsoon.", "Medium Priority", "Roads (PWD/NHAI)", "Medium"),
    ]
    
    print(f"\n{'Test Case':<70} | {'Expected':<30} | {'Predicted':<30}")
    print("-" * 140)
    
    correct_count = 0
    
    for text, context, exp_cat, exp_prio in test_cases:
        pred_cat, pred_prio = p.predict(text)
        
        match = "✅" if (pred_cat == exp_cat and pred_prio == exp_prio) else "❌"
        if match == "✅": correct_count += 1
        
        # Shorten text for display
        display_text = (text[:65] + '..') if len(text) > 65 else text
        
        print(f"{display_text:<70} | {exp_cat[:15]} / {exp_prio} | {pred_cat[:15]} / {pred_prio} {match}")

    print(f"\nAccuracy on Test Set: {correct_count}/{len(test_cases)}")

if __name__ == "__main__":
    validate()
