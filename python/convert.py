import json
import os
import urllib.request
import shutil

folders = [
    "dataset/images/train", "dataset/images/val",
    "dataset/labels/train", "dataset/labels/val"
]
for folder in folders:
    os.makedirs(folder, exist_ok=True)

ndjson_file = "handsyolo26.ndjson"
print(f"Analyse de {ndjson_file} en cours...\n")

with open(ndjson_file, "r", encoding="utf-8") as file:
    for line in file:
        try:
            data = json.loads(line.strip())
        except json.JSONDecodeError:
            continue
        
        if data.get("type") == "dataset":
            with open("dataset/dataset.yaml", "w", encoding="utf-8") as yaml_file:
                yaml_file.write("path: ./\n")
                yaml_file.write("train: images/train\n")
                yaml_file.write("val: images/val\n\n")
                yaml_file.write("names:\n")
                # rennomage des classes -> je trouve ça + explicite
                yaml_file.write("  0: hand_0\n")
                yaml_file.write("  1: hand_1\n")
                yaml_file.write("  2: hand_2\n")
                yaml_file.write("  3: hand_3\n")
                yaml_file.write("  4: hand_4\n")
                yaml_file.write("  5: hand_5\n")
            print("✅ dataset.yaml généré.")
            
        elif data.get("type") == "image":
            filename = data.get("file")
            img_url = data.get("url")
            split = data.get("split", "train")
            
            img_path = f"dataset/images/{split}/{filename}"
            if img_url and not os.path.exists(img_path):
                print(f"⬇️ Téléchargement : {filename}")
                try:
                    urllib.request.urlretrieve(img_url, img_path)
                except Exception as e:
                    print(f"❌ Erreur sur {filename}: {e}")
                    continue
            
            txt_filename = filename.rsplit('.', 1)[0] + ".txt"
            txt_path = f"dataset/labels/{split}/{txt_filename}"
            
            annotations = data.get("annotations", {})
            boxes = annotations.get("boxes", [])
            
            with open(txt_path, "w", encoding="utf-8") as txt_file:
                for box in boxes:
                    # format [class_id, x, y, w, h]
                    class_id = int(box[0])
                    x, y, w, h = box[1], box[2], box[3], box[4]
                    txt_file.write(f"{class_id} {x} {y} {w} {h}\n")
                    
print("\n📦 Création de l'archive dataset.zip en cours (cela peut prendre quelques secondes)...")
# shutil.make_archive(nom_du_fichier_cible, format, dossier_source)
shutil.make_archive("dataset", "zip", "dataset")

print("✅ Archive dataset.zip créée avec succès !")
print("Glisser-déposer le fichier 'dataset.zip' directement sur Ultralytics HUB.")