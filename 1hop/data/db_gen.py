import csv
import random
import os
import datetime

def load_list(filename):
    """Loads a list from a file, stripping whitespace and trailing periods."""
    with open(filename, "r") as f:
        # .strip() removes leading/trailing whitespace
        # .rstrip('.') removes trailing periods
        return [line.strip().rstrip('.') for line in f if line.strip()]

def get_random_date():
    """Generates a random date and returns it in Day,Month,Year format."""
    year = random.randint(1950, 2005)
    month = random.randint(1, 12)
    day = random.randint(1, 28) # Avoid issues with month lengths
    date_obj = datetime.datetime(year, month, day)
    return f"{date_obj.day},{date_obj.strftime('%B')},{date_obj.year}"

def generate_biography(person_data, templates):
    """Generates a formatted biography for a person."""
    pronoun_map = templates["pronouns"][person_data["Pronoun_Key"]]

    # Ensure the name is properly capitalized for all uses.
    capitalized_name = " ".join(word.capitalize() for word in person_data["People_Name"].split())

    # First sentence uses the full name and birth date
    first_sentence_template_data = {
        "PRONOUN": capitalized_name,
        "PRONOUN_POSSESSIVE": f"{capitalized_name}'s",
        "BIRTH_DATE": person_data["Birth_Date"],
        "FULL_NAME": capitalized_name, 
    }
    first_sentence = random.choice(templates["birth_date"]).format(**first_sentence_template_data)

    # Subsequent sentences use pronouns
    template_data = {
        "FULL_NAME": capitalized_name,
        "BIRTH_DATE": person_data["Birth_Date"],
        "BIRTH_CITY": person_data["Birth_City_Name"],
        "UNIVERSITY": person_data["University_Name"],
        "MAJOR": person_data["Major_Name"],
        "EMPLOYER": person_data["Company_Name"],
        "COMPANY_CITY": person_data["Work_City_Name"],
        **pronoun_map
    }

    # Create a pool of 6 sentences from the other attributes
    sentence_templates = {
        "birth_city": templates["birth_city"],
        "university": templates["university"],
        "major": templates["major"],
        "employer": templates["employer"],
        "company_city": templates["company_city"],
    }
    
    # Generate 5 standard sentences + 1 extra random one to make 6
    sentences = [random.choice(sentence_templates[key]).format(**template_data) for key in sentence_templates]
    random_attribute_templates = random.choice(list(sentence_templates.values()))
    sentences.append(random.choice(random_attribute_templates).format(**template_data))
    random.shuffle(sentences)

    # Capitalize first sentence, preserve original capitalization for others
    all_sentences = [first_sentence] + sentences
    
    # Join with ". " and add the end token
    return ". ".join(all_sentences) + ".<|endoftext|>"

def generate_qa_pairs(person_data, qa_templates):
    """Generates a list of Q&A pairs for a person."""
    qa_pairs = []
    q_template_map = {
        "Birth_Date": qa_templates["birth_date"],
        "Birth_City_Name": qa_templates["birth_city"],
        "University_Name": qa_templates["university"],
        "Major_Name": qa_templates["major"],
        "Company_Name": qa_templates["employer"],
        "Work_City_Name": qa_templates["company_city"],
    }
    for key, templates in q_template_map.items():
        question = random.choice(templates).format(FULL_NAME=person_data["People_Name"])
        answer = person_data[key]
        qa_pairs.append({"question": question, "answer": answer})
    return qa_pairs

def main():
    """Main function to generate and write all data."""
    random.seed(42)  # for reproducibility
    num_entries = 10
    data_dir = "entities"
    templates_dir =  "templates"
    qa_templates_dir = "qa_templates"

    # --- Load all data and templates ---
    first_names = load_list(os.path.join(data_dir, "first_names.txt"))
    middle_names = load_list(os.path.join(data_dir, "middle_names.txt"))
    last_names = load_list(os.path.join(data_dir, "last_names.txt"))
    birth_cities = load_list(os.path.join(data_dir, "birth_cities.txt"))
    universities = load_list(os.path.join(data_dir, "universities.txt"))
    major_names = load_list(os.path.join(data_dir, "major_names.txt"))

    # --- Generate Data ---
    possible_names = len(first_names) * len(middle_names) * len(last_names)
    if num_entries > possible_names:
        raise ValueError(f"Cannot generate {num_entries} unique names. Only {possible_names} are possible.")

    # Load companies and templates (restored from original)
    companies = []
    with open(os.path.join(data_dir, "companies.csv"), "r", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)  # Skip header
        for row in reader:
            if len(row) == 2:
                companies.append({"name": row[0], "city": row[1]})
            elif len(row) == 1:
                companies.append({"name": row[0], "city": row[0]})

    templates = {
        "birth_date": load_list(os.path.join(templates_dir, "birth_date_templates.txt")),
        "birth_city": load_list(os.path.join(templates_dir, "birth_city_templates.txt")),
        "university": load_list(os.path.join(templates_dir, "university_templates.txt")),
        "major": load_list(os.path.join(templates_dir, "major_templates.txt")),
        "employer": load_list(os.path.join(templates_dir, "employer_templates.txt")),
        "company_city": load_list(os.path.join(templates_dir, "company_city_templates.txt")),
        "pronouns": {
            "he": {"PRONOUN": "He", "PRONOUN_POSSESSIVE": "his"},
            "she": {"PRONOUN": "She", "PRONOUN_POSSESSIVE": "her"}
            # "they": {"PRONOUN": "They", "PRONOUN_POSSESSIVE": "their"},
        }
    }

    qa_templates = {
        "birth_date": load_list(os.path.join(qa_templates_dir, "birth_date_questions.txt")),
        "birth_city": load_list(os.path.join(qa_templates_dir, "birth_city_questions.txt")),
        "university": load_list(os.path.join(qa_templates_dir, "university_questions.txt")),
        "major": load_list(os.path.join(qa_templates_dir, "major_questions.txt")),
        "employer": load_list(os.path.join(qa_templates_dir, "employer_questions.txt")),
        "company_city": load_list(os.path.join(qa_templates_dir, "company_city_questions.txt")),
    }

    all_data = []
    used_names = set()
    while len(all_data) < num_entries:
        first = random.choice(first_names).capitalize()
        middle = random.choice(middle_names)
        last = random.choice(last_names)
        full_name = f"{first} {middle} {last}".title()

        if full_name in used_names:
            continue
        used_names.add(full_name)

        company_info = random.choice(companies)
        pronoun_key = random.choice(list(templates["pronouns"].keys()))

        person_data = {
            "People_Name": full_name,
            "Birth_Date": get_random_date(),
            "Birth_City_Name": random.choice(birth_cities),
            "University_Name": random.choice(universities),
            "Company_Name": company_info["name"],
            "Major_Name": random.choice(major_names),
            "Work_City_Name": company_info["city"],
            "Pronoun_Key": pronoun_key
        }

        person_data["Bio"] = generate_biography(person_data, templates)
        person_data["QA_Pairs"] = generate_qa_pairs(person_data, qa_templates)
        all_data.append(person_data)

    # --- Write Files ---
    # Write CSV for QAacc_V2.py
    with open("output.csv", "w", newline="") as f:
        fieldnames = ["People_Name", "Birth_Date", "Birth_City_Name", "University_Name", "Company_Name", "Major_Name", "Work_City_Name"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for person in all_data:
            writer.writerow({k: v for k, v in person.items() if k in fieldnames})
    print(f"Generated {len(all_data)} entries in output.csv")

    # Write biographies file
    with open("bio.txt", "w") as f:
        for person in all_data:
            f.write(person["Bio"] + "\n")
    print(f"Generated {len(all_data)} biographies in bio.txt")

    # Write Q&A file
    with open("qa.txt", "w") as f:
        for person in all_data:
            for pair in person["QA_Pairs"]:
                f.write(f"{pair['question']} Answer: {pair['answer']} .<|endoftext|>\n")
    print(f"Generated {len(all_data) * 6} Q&A pairs in qa.txt")

if __name__ == "__main__":
    main()