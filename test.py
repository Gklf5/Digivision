import pickle
import os
embeddings_file = "face_embeddings.pkl"

if not os.path.exists(embeddings_file):
    print("❌ Face database not found!")
else:
    with open(embeddings_file, 'rb') as f:
        embeddings = pickle.load(f)

    if not embeddings:
        print("❌ No faces found in the database!")
    else:
        print("\n✅ Stored Faces in Database:")
        for name, emb_list in embeddings.items():
            print(f"- {name} ({len(emb_list)} embeddings)")
