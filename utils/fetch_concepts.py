# Create a Class that given .txt files with concepts, creates a list of concepts

import os
import sys
import re
import numpy as np
import pandas as pd
from pathlib import Path
        
class ECIIConceptExtractor:
    def __init__(self, threshold=0.9, concept_dir=None):
        self.threshold = threshold
        self.concept_dir = concept_dir
        self.unique_concepts = set()

    def get_concepts(self, file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()

        solution = None
        coverage_score = None

        for line in lines:
            line = line.strip()
            if line.startswith("solution "):
                solution = line.split("imageContains.")[1].strip()
            elif line.startswith("coverage_score:"):
                coverage_score = float(line.split(":")[1].strip())
                if coverage_score >= self.threshold and solution is not None:
                    # Use regular expression to extract the desired word (e.g., "Building")
                    matches = re.findall(r'hcbdwsu:(\w+)', solution)
                    if matches:
                        solution_strings = [word.split(":")[-1].split("WN_")[-1].lower() if ":" in word else word.split("WN_")[-1].lower() if "WN_" in word else word.lower() for word in matches]
                        solution_strings = list(set(solution_strings))
                        self.unique_concepts.update(solution_strings)

                # Reset solution and coverage_score
                solution = None
                coverage_score = None

    def get_all_concepts(self):
        for root, _, files in os.walk(self.concept_dir):
            for file in files:
                if file.endswith(".txt"):
                    file_path = os.path.join(root, file)
                    self.get_concepts(file_path)

    def get_unique_concepts(self):
        self.get_all_concepts()
        return list(self.unique_concepts)

class AnnotationExtractor:
    def __init__(self, concept_dir=None):
        self.concept_dir = concept_dir
        self.unique_concepts = set()

    def get_concepts(self, file_path):
        with open(file_path, 'r') as file:
            for line in file:
                parts = line.strip().split('#')
                for part in parts:
                    # Use regular expression to extract words consisting of alphabets
                    words = re.findall(r'\b[a-zA-Z\s]+\b', part)
                    for word in words:
                        if ' ' in word:
                            # If it's a multi-word phrase, join with underscores
                            word = '_'.join(word.split())
                        self.unique_concepts.add(word.lower())

    def get_all_concepts(self):
        for root, _, files in os.walk(self.concept_dir):
            for file in files:
                if "ADE_train_" in file and file.endswith(".txt"):
                    file_path = os.path.join(root, file)
                    self.get_concepts(file_path)

    def get_unique_concepts(self):
        self.get_all_concepts()
        return list(self.unique_concepts)

if __name__ == "__main__":
    # Usage example:
    ecii_concepts_dir = Path(Path.cwd(), "data/scene/ecii_concepts/")
    ecii_concepts_threshold = 0.00
    concept_extractor = ECIIConceptExtractor(
        threshold=ecii_concepts_threshold, concept_dir=ecii_concepts_dir
    )
    ecii_concepts = concept_extractor.get_unique_concepts()
    ecii_concepts.sort()
    print(f"Number of ECII concepts: {len(ecii_concepts)}")

    annotations_dir = Path(Path.cwd(), "data/scene/ade20k_annotations/")
    annotation_extractor = AnnotationExtractor(concept_dir=annotations_dir)
    baseline_concepts = annotation_extractor.get_unique_concepts()
    baseline_concepts.sort()
    print(f"Number of baseline concepts: {len(baseline_concepts)}")


    # Task 1: Find overlaps
    overlaps = set(ecii_concepts) & set(baseline_concepts)
    overlap_count = len(overlaps)

    # Task 2: Find unique items in ecii_concepts not in baseline_concepts
    unique_ecii_concepts = [item for item in ecii_concepts if item not in baseline_concepts]
    unique_count_ecii = len(unique_ecii_concepts)

    # Task 3: Find unique items in baseline_concepts not in ecii_concepts
    unique_baseline_concepts = [item for item in baseline_concepts if item not in ecii_concepts]
    unique_count_baseline = len(unique_baseline_concepts)

    # Printing results
    print("Task 1: Overlaps")
    print("Overlaps:", overlaps)
    print("Overlap Count:", overlap_count)

    print("\nTask 2: Unique items in ecii_concepts not in baseline_concepts")
    print("Unique List in ecii_concepts:", unique_ecii_concepts)
    print("Unique Count in ecii_concepts:", unique_count_ecii)

    print("\nTask 3: Unique items in baseline_concepts not in ecii_concepts")
    print("Unique List in baseline_concepts:", unique_baseline_concepts)
    print("Unique Count in baseline_concepts:", unique_count_baseline)
    print("----------------------------------")