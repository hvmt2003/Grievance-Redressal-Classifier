import pandas as pd
import random
import os

def generate_up_data(num_samples=3000):
    # 1. Define UP Departments (Categories) and their specific issues
    # Structure: {Category: {Severity: [Templates]}}
    departments = {
        'Electricity (UPPCL)': {
            'High': [
                "transformer caught fire in {location} and oil is leaking", 
                "poore gaon mein light nahi hai {location} mein pichle 2 din se", # Hinglish
                "{location} me transformer jal gaya hai, turant change karein", # Hinglish
                "live wire fell on the street in {location}, huge danger to kids"
            ],
            'Medium': [
                "frequent power cuts in {location} is affecting exam preparation",
                "bijli ka bill bahut jyada aaya hai {location} mein", # Hinglish
                "meter reading galat hai {location} wale ghar ki", # Hinglish
                "low voltage problem in {location} during evening hours"
            ],
            'Low': [
                "street light pole looking tilted in {location}",
                "meter name change application pending in {location}",
                "solar panel subsidy ki jankari chahiye {location} ke liye" # Hinglish
            ]
        },
        'Sanitation & Water (Jal Nigam/Nagar Nigam)': {
            'High': [
                "sewage water mixing with drinking water in {location}, kids falling sick",
                "clean water nahi aa raha hai {location} mein, paani bohot ganda hai", # Hinglish
                "ganda paani nalo se aa raha hai {location} area me", # Hinglish
                "open manhole on main road in {location}, heavy accident risk"
            ],
            'Medium': [
                "garbage truck not coming to {location} for last 3 days",
                "kachra uthane vali gaadi nahi aayi {location} mein", # Hinglish
                "handpump kharab hai {location} mein", # Hinglish
                "water pipeline leakage wasting water in {location}"
            ],
            'Low': [
                "park needs cleaning in {location} residential area",
                "public toilet tap is dripping in {location}",
                " मच्छर बहुत हो गए हैं {location} में, फॉगिंग करवाओ" # Hindi
            ]
        },
        'Roads (PWD/NHAI)': {
            'High': [
                "huge crater in middle of road at {location} causing accidents daily",
                "sadak par bada gadda hai {location} market ke paas", # Hinglish
                "bridge railing toot gayi hai {location} mein", # Hinglish
                "road caved in due to heavy rain in {location}, traffic blocked"
            ],
            'Medium': [
                "potholes needs repair in {location} before monsoon",
                "sadak toot gayi hai {location} mein", # Hinglish
                "road construction incomplete in {location}",
                "encroachment by shopkeepers on road in {location}"
            ],
            'Low': [
                "need a sign board direction for {location} village",
                "zebra crossing repaint kara do {location} chowk par", # Hinglish
                "footpath tiles broken in some places at {location}"
            ]
        },
        'Police (Home Dept)': {
            'High': [
                "armed robbery in shop at {location} market last night",
                "chori ho gayi hai {location} mein, police bhejo", # Hinglish
                "ladayi ho rahi hai {location} chourahe par", # Hinglish
                "child kidnapped from {location} school area"
            ],
            'Medium': [
                "mobile phone snatched while walking in {location}",
                "phone chori ho gaya {location} market me", # Hinglish
                "loudspeaker shor kar raha hai {location} mein raat ko", # Hinglish
                "traffic police asking for bribe at {location} crossing"
            ],
            'Low': [
                "passport verification pending at {location} thana for 2 weeks",
                "character certificate chahiye {location} police station se", # Hinglish
                "general query about filing fir online for {location} resident"
            ]
        },
        'Health (Medical)': {
            'High': [
                "doctor refused to treat critical patient in {location} phc",
                "ambulance nahi aa rahi {location} mein, patient serious hai", # Hinglish
                "oxygen cylinder available nahi hai {location} hospital me", # Hinglish
                "fake medicine racket operating in {location} medical stores"
            ],
            'Medium': [
                "dogs roaming inside the ward at {location} chc",
                "doctor late aate hain {location} hospital mein", # Hinglish
                "x-ray machine kharab hai {location} mein", # Hinglish
                "unhygienic toilets in the {location} government hospital"
            ],
            'Low': [
                "long waiting line for opd registration in {location}",
                "vaccine kab lagega {location} center par?", # Hinglish
                "birth certificate spelling correction request at {location}"
            ]
        },
        'Rural Development (Panchayati Raj)': {
            'High': [
                "pradhan embezzled all money for road construction in {location}",
                "mnrega ka paisa nahi mila {location} mein", # Hinglish
                "pradhan ne paisa kha liya {location} road ka", # Hinglish
                "flood water entered village houses in {location}, need rescue"
            ],
            'Medium': [
                "village pond is being encroached by builder in {location}",
                "talaab par kabza ho raha hai {location} gaon mein", # Hinglish
                "primary school roof is leaking in {location}, dangerous for kids",
                "stray cattle destroying crops in {location} farms"
            ],
            'Low': [
                "request to start sewing center for women in {location}",
                "internet nahi chal raha {location} panchayat bhavan me", # Hinglish
                "library books request for {location} village school"
            ]
        }
    }

    # 2. UP Specific Locations (Urban, Semi-Urban, Rural)
    loc_urban = ["Hazratganj, Lucknow", "Noida Sector 18", "Civil Lines, Kanpur", "Sigra, Varanasi", "Prayagraj City", "Gomti Nagar, Lucknow", "Ghaziabad", "Meerut City", "Bareilly Cantt", "Agra Fort area"]
    loc_semi = ["Unnao City", "Barabanki Town", "Raebareli Market", "Sitapur One", "Faizabad Chowk", "Etawah Main", "Hardoi City", "Orai", "Basti", "Gonda City"]
    loc_rural = ["Village Rampur", "Mau Tehsil", "Post Kakori", "Gram Panchayat Bojhi", "Village Basrehar", "Tehsil Bikapur", "Block Malihabad", "Village Chhata", "Gram Kachhona", "Remote area of Sonbhadra"]
    
    locations = loc_urban + loc_semi + loc_rural

    data = []
    
    # 3. Generate Data ensuring Balance
    # We want roughly equal Low, Medium, High for each category
    
    samples_per_category = num_samples // len(departments)
    
    for category, difficulties in departments.items():
        for _ in range(samples_per_category):
            # Pick urgency first to ensure balance (33% each)
            urgency = random.choice(['High', 'Medium', 'Low'])
            
            # Pick a template for that urgency
            templates = difficulties[urgency]
            template = random.choice(templates)
            
            # Pick a suitable location type based on context maybe? 
            # For simplicity, random UP location, but bias 'Rural' dept to Rural locs
            if category == 'Rural Development (Panchayati Raj)':
                loc = random.choice(loc_rural)
            elif category == 'Electricity (UPPCL)' and 'village' in template:
                 loc = random.choice(loc_rural)
            else:
                loc = random.choice(locations)
                
            text = template.format(location=loc)
            
            data.append({
                'text': text,
                'category': category,
                'priority': urgency
            })

    # Shuffle
    random.shuffle(data)
    
    df = pd.DataFrame(data)
    
    output_dir = os.path.join("data", "raw")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "grievances.csv")
    
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} UP-specific samples at {output_path}")
    print("\nPriority Distribution:\n", df['priority'].value_counts())
    print("\nCategory Distribution:\n", df['category'].value_counts())

if __name__ == "__main__":
    generate_up_data()
