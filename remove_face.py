import pickle
import os

embeddings_file = "face_embeddings.pkl"

def remove_face_from_database(name):
    if not os.path.exists(embeddings_file):
        print("❌ Database file not found.")
        return

    with open(embeddings_file, 'rb') as f:
        embeddings = pickle.load(f)

    if name in embeddings:
        del embeddings[name]
        with open(embeddings_file, 'wb') as f:
            pickle.dump(embeddings, f)
        print(f"✅ Removed {name} from the database.")
    else:
        print(f"❌ Name '{name}' not found in database.")

# Show stored names
if os.path.exists(embeddings_file):
    with open(embeddings_file, 'rb') as f:
        embeddings = pickle.load(f)
    
    print("\nStored Faces:")
    for person in embeddings.keys():
        print(f"- {person}")

    # Ask user to remove a face
    person_name = input("\nEnter the name to remove (or 'q' to quit): ")
    if person_name != 'q':
        remove_face_from_database(person_name)
