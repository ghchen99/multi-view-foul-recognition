import json

# Function to load annotation data from a JSON file
def load_annotations(file_path):
    """
    Loads annotations from a JSON file.
    """
    with open(file_path, 'r') as f:
        return json.load(f)

# Function to process actions and calculate attribute counts
def process_actions(actions):
    """
    Processes actions to count distinct values and their occurrences for each attribute.
    """
    attribute_counts = {}

    for action_id, attributes in actions.items():
        for key, value in attributes.items():
            if key != "Clips":  # Exclude 'Clips'
                if key not in attribute_counts:
                    attribute_counts[key] = {}
                if value not in attribute_counts[key]:
                    attribute_counts[key][value] = 0
                attribute_counts[key][value] += 1
    
    return attribute_counts

# Function to display attribute counts
def display_counts(attribute_counts):
    """
    Displays distinct values and their occurrences for each attribute in a formatted manner.
    """
    print("Attribute Counts:\n=================")
    for key, values in attribute_counts.items():
        print(f"{key}:")
        for value, count in values.items():
            print(f"  {value if value else 'Empty'}: {count} occurrences")
        print("-----------------")
    
# Main function to load, process, and save data
def main():
    annotations = load_annotations('data/dataset/train/annotations.json')
    
    # Extract actions
    actions = annotations["Actions"]

    attribute_counts = process_actions(actions)
    display_counts(attribute_counts)
   

if __name__ == "__main__":
    main()
    
    
'''
-----------------
Offence:
  Offence: 2495 occurrences
  No offence: 324 occurrences
  Between: 96 occurrences
  Empty: 1 occurrences
-----------------
Contact:
  With contact: 2835 occurrences
  Without contact: 81 occurrences
-----------------
Bodypart:
  Upper body: 1048 occurrences
  Under body: 1831 occurrences
  Empty: 37 occurrences
-----------------
Upper body part:
  Use of shoulder: 332 occurrences
  Empty: 1899 occurrences
  Use of arms: 670 occurrences
  Use of shoulders: 15 occurrences
-----------------
Action class:
  Challenge: 383 occurrences
  Tackling: 448 occurrences
  Standing tackling: 1264 occurrences
  High leg: 103 occurrences
  Dive: 28 occurrences
  Elbowing: 178 occurrences
  Empty: 11 occurrences
  Holding: 361 occurrences
  Dont know: 52 occurrences
  Pushing: 88 occurrences
-----------------
Severity:
  1.0: 1402 occurrences
  3.0: 687 occurrences
  5.0: 27 occurrences
  Empty: 353 occurrences
  2.0: 403 occurrences
  4.0: 44 occurrences
-----------------
Multiple fouls:
  Empty: 377 occurrences
  Yes: 304 occurrences
  No: 2234 occurrences
  yes: 1 occurrences
-----------------
Try to play:
  Empty: 1133 occurrences
  Yes: 1650 occurrences
  No: 133 occurrences
-----------------
Touch ball:
  Empty: 1135 occurrences
  No: 1543 occurrences
  Yes: 192 occurrences
  Maybe: 46 occurrences
-----------------
Handball:
  No handball: 2892 occurrences
  Handball: 24 occurrences
-----------------
Handball offence:
  Empty: 2892 occurrences
  No offence: 6 occurrences
  Offence: 18 occurrences
  -----------------'''