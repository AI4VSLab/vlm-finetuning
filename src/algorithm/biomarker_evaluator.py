
# import logging
# import csv
# import random
# from tqdm import tqdm
# import numpy as np
# import pandas as pd
# from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# logger = logging.getLogger(__name__)

# class Evaluator:
#     def __init__(self, model, data, config):
#         """
#         Args:
#             model: The VLM model instance.
#             data: Data provider containing 'biomarkers_dataset' and 'vqa_dataset'.
#             config: Configuration object.
#         """
#         self.config = config
#         self.model = model
#         self.data = data

#         # 1. DEFINITIONS
#         # The canonical list of biomarkers based on "Groups (1).pdf"
#         # "Fluid" is placed last to avoid partial matches with "Intraretinal Fluid", etc.
#         self.biomarker_schema = [
#             "Pigment Epithelial Detachment",
#             "Drusen",
#             "Intraretinal Fluid",
#             "Subretinal Fluid",
#             "Choroidal Neovascularization",
#             "Posterior Vitreous Detachment",
#             "Geographic Atrophy",
#             "Epiretinal Membrane",
#             "Photoreceptor Layer Disruption",
#             "Hemorrhage",
#             "Retinal Pigment Epithelium Migration",
#             "Retinal Pigment Epithelium Atrophy",
#             "Disciform Scar",
#             "Outer Retinal Tubulation",
#             "Subretinal Hyperreflective Material",
#             "Retinal Pigment Epithelium Tear",
#             "Choroidal Fold",
#             "Fluid"
#         ]

#         # Synonyms from "Groups (1).pdf"
#         # Mapping canonical group names to their allowed synonyms/abbreviations.
#         self.term_mapping = {
#             "Pigment Epithelial Detachment": [
#                 "pigment epithelial detachment", "ped"
#             ],
#             "Drusen": [
#                 "drusen", "soft drusen", "hard drusen", "reticular drusen"
#             ],
#             "Intraretinal Fluid": [
#                 "intraretinal fluid", "irf", "cystoid macular edema", "cme"
#             ],
#             "Subretinal Fluid": [
#                 "subretinal fluid", "srf"
#             ],
#             "Choroidal Neovascularization": [
#                 "choroidal neovascularization", "choroidal neovascular membrane",
#                 "cnv", "cnvm"
#             ],
#             "Posterior Vitreous Detachment": [
#                 "posterior vitreous detachment", "pvd"
#             ],
#             "Geographic Atrophy": [
#                 "geographic atrophy", "ga"
#             ],
#             "Epiretinal Membrane": [
#                 "epiretinal membrane", "erm"
#             ],
#             "Photoreceptor Layer Disruption": [
#                 "photoreceptor layer disruption", "photoreceptor degeneration"
#             ],
#             "Hemorrhage": [
#                 "hemorrhage"
#             ],
#             "Retinal Pigment Epithelium Migration": [
#                 "retinal pigment epithelium migration", "rpe migration"
#             ],
#             "Retinal Pigment Epithelium Atrophy": [
#                 "retinal pigment epithelium atrophy"
#             ],
#             "Disciform Scar": [
#                 "disciform scar"
#             ],
#             "Outer Retinal Tubulation": [
#                 "outer retinal tubulation"
#             ],
#             "Subretinal Hyperreflective Material": [
#                 "subretinal hyperreflective material"
#             ],
#             "Retinal Pigment Epithelium Tear": [
#                 "retinal pigment epithelium tear", "rpe tear"
#             ],
#             "Choroidal Fold": [
#                 "choroidal fold"
#             ],
#             "Fluid": [
#                 "fluid"
#             ]
#         }

#         self.model.inference_mode()
#         logger.info(f"Evaluator initialized. Tracking {len(self.biomarker_schema)} biomarkers.")

#     # =========================================================================
#     #  HELPER: Text Parsing
#     # =========================================================================

#     def _normalize_text(self, text):
#         return text.lower().strip()

#     def _text_has_biomarker(self, text, marker_code):
#         """Returns True if the marker or its synonyms appear in the text."""
#         text = self._normalize_text(text)
#         synonyms = self.term_mapping.get(marker_code, [marker_code.lower()])

#         for term in synonyms:
#             if term in text:
#                 return True
#         return False

#     def _identify_target_biomarker_in_question(self, question):
#         """
#         For VQA: Figures out which biomarker the question is asking about.
#         Example: "Is intraretinal fluid present?" -> Returns "Intraretinal Fluid"
#         """
#         for code in self.biomarker_schema:
#             if self._text_has_biomarker(question, code):
#                 return code
#         return "UNKNOWN"

#     def _parse_yes_no(self, text):
#         """
#         Parses VQA output.
#         Returns: 1 (Yes), 0 (No), or -1 (Unclear/Other)
#         """
#         text = self._normalize_text(text)
#         if text.startswith("yes") or " yes " in f" {text} " or text == "yes.":
#             return 1
#         if text.startswith("no") or " no " in f" {text} " or text == "no.":
#             return 0
#         return -1

#     # =========================================================================
#     #  TASK 1: LIST GENERATION EVALUATION
#     # =========================================================================

#     def evaluate_list_task(self, dataset, output_csv="results_list_gen.csv"):
#         """
#         Evaluates the model's ability to generate a full list of biomarkers.
#         Method: Multi-Label Classification.
#         """
#         logger.info(f"--- Evaluating List Generation Task ({len(dataset)} samples) ---")

#         all_pred_vectors = []
#         all_ref_vectors = []
#         csv_rows = []

#         for idx, sample in tqdm(enumerate(dataset), total=len(dataset), desc="List Gen Eval"):
#             image = sample["image"]
#             prompt = sample["instruction"]
#             reference = sample["response"]

#             try:
#                 predicted = self.model.generate(prompt, image, use_streamer=False)
#             except Exception:
#                 predicted = ""

#             # Convert text to binary vectors based on schema
#             pred_vec = []
#             ref_vec = []

#             for marker in self.biomarker_schema:
#                 # Check if marker is in Reference
#                 ref_has = 1 if self._text_has_biomarker(reference, marker) else 0
#                 ref_vec.append(ref_has)

#                 # Check if marker is in Prediction
#                 pred_has = 1 if self._text_has_biomarker(predicted, marker) else 0
#                 pred_vec.append(pred_has)

#             all_pred_vectors.append(pred_vec)
#             all_ref_vectors.append(ref_vec)

#             csv_rows.append({
#                 "sample_id": idx,
#                 "reference": reference,
#                 "prediction": predicted,
#                 "ref_vector": str(ref_vec),
#                 "pred_vector": str(pred_vec)
#             })

#         self._save_csv(csv_rows, output_csv)

#         # Calculate Metrics (Per-Column in the matrix)
#         y_true = np.array(all_ref_vectors)
#         y_pred = np.array(all_pred_vectors)

#         precision, recall, f1, support = precision_recall_fscore_support(
#             y_true, y_pred, average=None, zero_division=0
#         )

#         results = {}
#         for i, marker in enumerate(self.biomarker_schema):
#             results[marker] = {
#                 "precision": precision[i],
#                 "recall": recall[i],
#                 "f1": f1[i],
#                 "support": support[i]
#             }
#         return results

#     # =========================================================================
#     #  TASK 2: VQA EVALUATION
#     # =========================================================================

#     def evaluate_vqa_task(self, dataset, output_csv="results_vqa.csv"):
#         """
#         Evaluates specific Yes/No questions per biomarker.
#         Method: Aggregated Binary Classification.
#         """
#         logger.info(f"--- Evaluating VQA Task ({len(dataset)} samples) ---")

#         # Structure to hold true/pred lists for each biomarker independently
#         # e.g., vqa_stats["IRF"] = {"true": [1, 0, 1], "pred": [1, 0, 0]}
#         vqa_stats = {m: {"true": [], "pred": []} for m in self.biomarker_schema}

#         csv_rows = []

#         for idx, sample in tqdm(enumerate(dataset), total=len(dataset), desc="VQA Eval"):
#             image = sample["image"]
#             question = sample["instruction"]
#             reference = sample["response"] # Should be "Yes" or "No"

#             # 1. Identify which biomarker this question is about
#             target_marker = self._identify_target_biomarker_in_question(question)

#             if target_marker == "UNKNOWN":
#                 # logger.warning(f"Could not identify biomarker in question: '{question}'")
#                 continue

#             try:
#                 predicted = self.model.generate(question, image, use_streamer=False)
#             except Exception:
#                 predicted = "Error"

#             # 2. Parse Yes/No
#             y_true = self._parse_yes_no(reference)
#             y_pred = self._parse_yes_no(predicted)

#             # Skip invalid parsing if necessary, or treat invalid as wrong (here treating -1 as wrong)
#             if y_true == -1: continue

#             # 3. Store Data
#             vqa_stats[target_marker]["true"].append(y_true)
#             vqa_stats[target_marker]["pred"].append(1 if y_pred == 1 else 0)

#             csv_rows.append({
#                 "sample_id": idx,
#                 "biomarker": target_marker,
#                 "question": question,
#                 "reference": reference,
#                 "prediction": predicted,
#                 "parsed_true": y_true,
#                 "parsed_pred": y_pred,
#                 "correct": 1 if y_true == (1 if y_pred == 1 else 0) else 0
#             })

#         self._save_csv(csv_rows, output_csv)

#         # Calculate Metrics Per Biomarker Group
#         results = {}
#         for marker in self.biomarker_schema:
#             y_t = vqa_stats[marker]["true"]
#             y_p = vqa_stats[marker]["pred"]

#             if len(y_t) == 0:
#                 results[marker] = {"precision": 0, "recall": 0, "f1": 0, "support": 0}
#             else:
#                 p, r, f, _ = precision_recall_fscore_support(
#                     y_t, y_p, average='binary', zero_division=0
#                 )
#                 results[marker] = {
#                     "precision": p,
#                     "recall": r,
#                     "f1": f,
#                     "support": len(y_t) # Number of questions asked about this marker
#                 }

#         return results

#     # =========================================================================
#     #  RUNNER & UTILS
#     # =========================================================================

#     def _save_csv(self, rows, filename):
#         if not rows: return
#         keys = rows[0].keys()
#         with open(filename, 'w', newline='', encoding='utf-8') as f:
#             writer = csv.DictWriter(f, fieldnames=keys)
#             writer.writeheader()
#             writer.writerows(rows)
#         logger.info(f"Saved details to {filename}")

#     def _print_comparison_table(self, list_res, vqa_res):
#         logger.info("\n" + "="*85)
#         logger.info(f"{'BIOMARKER':<35} | {'LIST GEN (P/R/F1)':<22} | {'VQA (P/R/F1)':<22} | {'SAMPLES (L/V)':<15}")
#         logger.info("-" * 85)

#         for m in self.biomarker_schema:
#             l = list_res.get(m, {})
#             v = vqa_res.get(m, {})

#             l_str = f"{l.get('precision',0):.2f}/{l.get('recall',0):.2f}/{l.get('f1',0):.2f}"
#             v_str = f"{v.get('precision',0):.2f}/{v.get('recall',0):.2f}/{v.get('f1',0):.2f}"
#             count_str = f"{l.get('support',0)} / {v.get('support',0)}"

#             logger.info(f"{m:<35} | {l_str:<22} | {v_str:<22} | {count_str:<15}")
#         logger.info("="*85 + "\n")

#     def run(self):
#         logger.info("Starting Comprehensive Biomarker Evaluation...")

#         # 1. Evaluate List Generation Task
#         list_metrics = self.evaluate_list_task(
#             self.data.biomarkers_dataset,
#             output_csv="eval_list_generation.csv"
#         )

#         # 2. Evaluate VQA Task
#         vqa_metrics = self.evaluate_vqa_task(
#             self.data.vqa_dataset,
#             output_csv="eval_vqa.csv"
#         )

#         # 3. Print Consolidated Report
#         self._print_comparison_table(list_metrics, vqa_metrics)
