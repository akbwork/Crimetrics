import random
import pandas as pd
from faker import Faker

fake = Faker()

# Define templates for 5 crime types with narrative placeholders
templates = {
    "IPC 302 (Murder)": [
        "The body of the victim was discovered at a {crime_location_type} with signs of struggle.",
        "Eyewitnesses reported a fatal altercation involving a {weapon_used} near the {crime_location_type}.",
        "A local resident was found murdered in a {crime_location_type} under mysterious circumstances.",
        "The victim was attacked and killed using a {weapon_used} at a {crime_location_type}.",
        "The accused used a {weapon_used} to fatally wound the victim at their residence.",
    ],
    "IPC 376 (Rape)": [
        "The victim reported being sexually assaulted by a {relationship} in a {crime_location_type}.",
        "An incident of rape was reported where the accused lured the victim to a {crime_location_type}.",
        "The accused forcibly entered the victim’s home and committed the crime.",
        "A rape case was registered involving a {relationship} in an isolated {crime_location_type}.",
        "The victim was abducted and raped in a moving vehicle by unknown individuals.",
    ],
    "IPC 323 (Assault)": [
        "A violent brawl broke out at a {crime_location_type}, leading to serious injuries.",
        "The accused physically assaulted the victim using a {weapon_used} at a {crime_location_type}.",
        "A heated argument turned violent at the {crime_location_type}, injuring one person.",
        "The victim was attacked from behind with a {weapon_used} at a {crime_location_type}.",
        "Police received a report of assault involving two individuals at a {crime_location_type}.",
    ],
    "IPC 379 (Theft)": [
        "A theft was reported where items were stolen from a {crime_location_type}.",
        "Unknown suspects broke into a {crime_location_type} and stole valuables.",
        "The victim’s vehicle was broken into and items were stolen.",
        "Pickpocketing occurred in a crowded {crime_location_type}.",
        "Jewelry and cash were stolen from a {crime_location_type} during daytime.",
    ],
    "IPC 392 (Robbery)": [
        "An armed robbery took place at a {crime_location_type}, threatening the occupants.",
        "The victim was robbed at gunpoint while returning home from work.",
        "A gang of masked robbers looted a store at a {crime_location_type}.",
        "Robbers stormed a house and made away with cash and gold.",
        "The accused forcefully snatched the victim's belongings and fled on a {vehicle_involved}.",
    ]
}

# Full Rajasthan Districts (uneven/random distribution)
districts = [
    "Ajmer", "Alwar", "Balotra", "Banswara", "Baran", "Barmer", "Beawar", "Bharatpur",
    "Bhilwara", "Bikaner", "Bundi", "Chittorgarh", "Churu", "Dausa", "Deeg",
    "Didwana-Kuchaman", "Dholpur", "Dungarpur", "Hanumangarh", "Jaipur", "Jaisalmer",
    "Jalore", "Jhalawar", "Jhunjhunu", "Jodhpur", "Karauli", "Khairthal-Tijara",
    "Kotputli-Behror", "Kota", "Nagaur", "Pali", "Phalodi", "Pratapgarh", "Rajsamand",
    "Salumbar", "Sawai Madhopur", "Sikar", "Sirohi", "Sri Ganganagar", "Tonk", "Udaipur"
]

# Other feature options
police_stations = ["Station A", "Station B", "Station C"]
location_types = ["market", "residential area", "bus stop", "highway", "school", "park"]
weapons = ["knife", "gun", "stick", "bare hands"]
vehicles = ["bike", "car", "scooter", "auto"]
genders = ["Male", "Female", "Other"]
reporters = ["Victim", "Relative", "Eyewitness", "Police"]
relationships = ["Stranger", "Acquaintance", "Neighbor", "Relative", "Friend"]

ipc_labels = list(templates.keys())

ipc_to_category = {
    "IPC 302 (Murder)": "Murder",
    "IPC 376 (Rape)": "Rape",
    "IPC 323 (Assault)": "Assault",
    "IPC 379 (Theft)": "Theft",
    "IPC 392 (Robbery)": "Robbery"
}

# Function to generate one FIR record
def generate_record():
    ipc = random.choice(ipc_labels)
    template = random.choice(templates[ipc])
    location = random.choice(location_types)
    weapon = random.choice(weapons)
    vehicle = random.choice(vehicles)
    relation = random.choice(relationships)
    narrative = template.format(
        crime_location_type=location,
        weapon_used=weapon,
        relationship=relation,
        vehicle_involved=vehicle
    )
    date = fake.date_between(start_date="-1y", end_date="today")
    crime_category = ipc_to_category[ipc]
    
    return {
        "fir_narrative": narrative,
        "district": random.choice(districts),
        "police_station": random.choice(police_stations),
        "crime_location_type": location,
        "date_of_crime": date,
        "time_of_crime": fake.time(),
        "day_of_week": date.strftime("%A"),
        "season": random.choice(["Summer", "Monsoon", "Winter", "Spring"]),
        "reported_by": random.choice(reporters),
        "victim_gender": random.choice(genders),
        "victim_age": random.randint(10, 80),
        "accused_gender": random.choice(genders),
        "accused_age": random.randint(15, 70),
        "weapon_used": weapon,
        "vehicle_involved": vehicle,
        "num_accused": random.randint(1, 5),
        "relationship": relation,
        "ipc_section": ipc,
        "crime_category": crime_category
    }

# Generate the full dataset
def generate_dataset(n_rows=1000):
    data = [generate_record() for _ in range(n_rows)]
    df = pd.DataFrame(data)
    df.to_csv("crime_type_prediction_dataset_rajasthan.csv", index=False)
    print(f"✅ Dataset generated: {n_rows} rows saved to 'crime_type_prediction_dataset_rajasthan.csv'")

# Run the script
if __name__ == "__main__":
    generate_dataset(1000)
