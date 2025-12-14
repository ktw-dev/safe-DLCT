import numpy as np
import spacy

# Load Spacy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Spacy model 'en_core_web_sm' not found. Please download it.")
    nlp = None

def parse_spatial_triplets(caption):
    if nlp is None:
        return []
    doc = nlp(caption)
    triplets = []
    # Simplified parsing for demo: Subject -> Preposition -> Object
    # Look for "person in front of car"
    for token in doc:
        # Handle "in front of"
        if token.text == "in" and len(doc) > token.i + 2 and doc[token.i+1].text == "front" and doc[token.i+2].text == "of":
            # Head of "in" is subject (e.g. person)
            head = token.head
            # Object is child of "of" (token+2)
            of_token = doc[token.i+2]
            children = [child for child in of_token.children if child.dep_ == "pobj"]
            if children:
                triplets.append((head.text, "in front of", children[0].text))
        
        # Handle simple prepositions
        elif token.pos_ == "ADP" and token.text in ["on", "under", "behind", "near"]:
            head = token.head
            children = [child for child in token.children if child.dep_ == "pobj"]
            if children:
                triplets.append((head.text, token.text, children[0].text))
    
    # print(f"DEBUG: Caption: '{caption}', Triplets: {triplets}")
    return triplets

def get_geo_reward(caps_gen, depth_regions, classes):
    # caps_gen: list of strings
    # depth_regions: (bs, n_regions)
    # classes: (bs, n_regions)
    
    rewards = []
    
    for i, caption in enumerate(caps_gen):
        # print(f"DEBUG: caption type: {type(caption)}, content: {caption}")
        if isinstance(caption, list):
            caption = " ".join(caption)
            
        triplets = parse_spatial_triplets(caption)
        if not triplets:
            rewards.append(0.0)
            continue
            
        current_depths = depth_regions[i]
        current_classes = classes[i]
        
        is_consistent = True
        
        for subj, rel, obj in triplets:
            if subj == "person" and obj == "car":
                # Find indices of person (1) and car (3)
                person_indices = np.where(current_classes == 1)[0]
                car_indices = np.where(current_classes == 3)[0]
                
                if len(person_indices) > 0 and len(car_indices) > 0:
                    d_p = np.mean(current_depths[person_indices])
                    d_c = np.mean(current_depths[car_indices])
                    
                    # "person in front of car" -> person should be closer (smaller depth)
                    if rel in ["front", "in front of"]:
                        if d_p < d_c: 
                             rewards.append(1.0)
                        else:
                             rewards.append(-1.0)
                        is_consistent = False # Handled
                        break
        else:
            rewards.append(0.0) # No relevant triplets found or checked
        
    return np.array(rewards, dtype=np.float32)
