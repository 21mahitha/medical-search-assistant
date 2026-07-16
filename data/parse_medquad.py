import os
import xml.etree.ElementTree as ET

data_dir = "data/MedQuAD"
qa_pairs = []

for folder in os.listdir(data_dir):
    folder_path = os.path.join(data_dir, folder)
    if not os.path.isdir(folder_path):
        continue

    for filename in os.listdir(folder_path):
        if not filename.endswith(".xml"):
            continue

        file_path = os.path.join(folder_path, filename)
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()

            focus_elem = root.find("Focus")
            focus = focus_elem.text if focus_elem is not None else ""

            for qa_pair in root.findall(".//QAPair"):
                question_elem = qa_pair.find("Question")
                answer_elem = qa_pair.find("Answer")

                question = question_elem.text if question_elem is not None else ""
                answer = answer_elem.text if answer_elem is not None else ""

                if question and answer:
                    qa_pairs.append({
                        "focus": focus,
                        "question": question,
                        "answer": answer,
                        "source": folder
                    })
        except ET.ParseError:
            continue

print(f"Total QA pairs collected: {len(qa_pairs)}")
print(qa_pairs[0])

import json

with open("data/qa_pairs.json", "w", encoding="utf-8") as f:
    json.dump(qa_pairs, f, indent=2)

print("Saved to data/qa_pairs.json")