import json
import os
import re
from collections import defaultdict

from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score


def eval_realworldqa(results_file=None, results=None, output_dir=None):
    if results is None:
        if not os.path.exists(results_file):
            raise FileNotFoundError(f"Results file not found: {results_file}")
        with open(results_file, "r", encoding="utf-8") as handle:
            results = json.load(handle)

    correct = 0
    total = 0
    for item in results:
        pred = item["prediction"].strip()
        gt = item["gt_answer"].strip()
        is_correct = False

        if gt.lower() in ["yes", "no"] or gt.isdigit():
            pred_norm = re.sub(r"[^\w\s]", "", pred).lower()
            gt_norm = re.sub(r"[^\w\s]", "", gt).lower()
            pred_words = pred_norm.split()
            if pred_words and pred_words[0] == gt_norm:
                is_correct = True
        elif len(gt) == 1 and gt.upper() in "ABCDEF":
            gt_char = gt.upper()
            check_contents = [pred]
            lines = pred.strip().split("\n")
            if len(lines) > 1:
                check_contents.append(lines[-1])

            found_match = False
            for content in check_contents:
                if not content:
                    continue

                match = re.match(r"^\s*\(([A-F])\)", content, re.IGNORECASE)
                if match and match.group(1).upper() == gt_char:
                    is_correct = True
                    found_match = True
                    break

                match = re.match(r"^\s*([A-F])[\.\)\s]", content, re.IGNORECASE)
                if match and match.group(1).upper() == gt_char:
                    is_correct = True
                    found_match = True
                    break

                if content.strip().upper() == gt_char:
                    is_correct = True
                    found_match = True
                    break

                if any(keyword in content.lower() for keyword in ["answer", "option", "choice"]):
                    match = re.search(
                        r"(?:answer|option|choice)(?: is|:| is:)?\s*\(?([A-F])\)?",
                        content,
                        re.IGNORECASE,
                    )
                    if match and match.group(1).upper() == gt_char:
                        is_correct = True
                        found_match = True
                        break

                match = re.search(r"(?:^|\s|[^a-zA-Z0-9])([A-F])[\.\)]?$", content.strip(), re.IGNORECASE)
                if match and match.group(1).upper() == gt_char:
                    is_correct = True
                    found_match = True
                    break

            if not found_match and pred and pred[0].upper() == gt_char:
                is_correct = True
        elif pred.lower() == gt.lower():
            is_correct = True

        if is_correct:
            correct += 1
        total += 1

    acc = correct / total if total > 0 else 0.0
    print(f"\nAccuracy: {acc:.4f} ({correct}/{total})")

    if output_dir is None:
        output_dir = os.path.dirname(results_file)
    os.makedirs(output_dir, exist_ok=True)

    acc_file = os.path.join(output_dir, "accuracy.json")
    with open(acc_file, "w", encoding="utf-8") as handle:
        json.dump({"accuracy": acc, "correct": correct, "total": total}, handle, indent=2)
    print(f"Results saved to {output_dir}")


def eval_mmbench(results_file=None, results=None, output_dir=None):
    if results is None:
        if not os.path.exists(results_file):
            raise FileNotFoundError(f"Results file not found: {results_file}")
        with open(results_file, "r", encoding="utf-8") as handle:
            results = json.load(handle)

    def check_answer_correct(pred, gt):
        pred = pred.strip()
        gt = gt.strip()

        if gt.lower() in ["yes", "no"] or gt.isdigit():
            pred_norm = re.sub(r"[^\w\s]", "", pred).lower()
            gt_norm = re.sub(r"[^\w\s]", "", gt).lower()
            pred_words = pred_norm.split()
            return bool(pred_words and pred_words[0] == gt_norm)

        if len(gt) == 1 and gt.upper() in "ABCDEF":
            gt_char = gt.upper()
            check_contents = [pred]
            lines = pred.strip().split("\n")
            if len(lines) > 1:
                check_contents.append(lines[-1])

            for content in check_contents:
                if not content:
                    continue

                match = re.match(r"^\s*\(([A-F])\)", content, re.IGNORECASE)
                if match and match.group(1).upper() == gt_char:
                    return True

                match = re.match(r"^\s*([A-F])[\.\)\s]", content, re.IGNORECASE)
                if match and match.group(1).upper() == gt_char:
                    return True

                if content.strip().upper() == gt_char:
                    return True

                if any(keyword in content.lower() for keyword in ["answer", "option", "choice"]):
                    match = re.search(
                        r"(?:answer|option|choice)(?: is|:| is:)?\s*\(?([A-F])\)?",
                        content,
                        re.IGNORECASE,
                    )
                    if match and match.group(1).upper() == gt_char:
                        return True

                match = re.search(r"(?:^|\s|[^a-zA-Z0-9])([A-F])[\.\)]?$", content.strip(), re.IGNORECASE)
                if match and match.group(1).upper() == gt_char:
                    return True

            return bool(pred and pred[0].upper() == gt_char)

        return pred.lower() == gt.lower()

    def get_base_id(qid):
        try:
            qid_int = int(qid)
        except (ValueError, TypeError):
            return qid

        if qid_int >= 3000000:
            return qid_int - 3000000
        if qid_int >= 2000000:
            return qid_int - 2000000
        if qid_int >= 1000000:
            return qid_int - 1000000
        return qid_int

    question_groups = defaultdict(list)
    for item in results:
        qid = item.get("id", item.get("index"))
        if qid is None:
            continue
        question_groups[get_base_id(qid)].append(item)

    correct_groups = 0
    total_groups = len(question_groups)
    total_samples = 0
    correct_samples = 0

    for samples in question_groups.values():
        group_correct = True
        for sample in samples:
            pred = sample.get("prediction", "")
            gt = sample.get("gt_answer", sample.get("answer", ""))
            if check_answer_correct(pred, gt):
                correct_samples += 1
            else:
                group_correct = False
            total_samples += 1
        if group_correct:
            correct_groups += 1

    acc = correct_groups / total_groups if total_groups > 0 else 0.0
    sample_acc = correct_samples / total_samples if total_samples > 0 else 0.0

    print("=" * 40)
    print("MMBench Evaluation Results")
    print("=" * 40)
    print(f"Total Unique Questions (Groups): {total_groups}")
    print(f"Correct Groups (All Pass):       {correct_groups}")
    print(f"Circular Accuracy:               {acc:.4f}")
    print("-" * 40)
    print(f"Total Individual Samples:        {total_samples}")
    print(f"Correct Samples:                 {correct_samples}")
    print(f"Sample Accuracy:                 {sample_acc:.4f}")
    print("=" * 40)

    if output_dir is None:
        output_dir = os.path.dirname(results_file) if results_file else "./results"
    os.makedirs(output_dir, exist_ok=True)

    acc_file = os.path.join(output_dir, "accuracy.json")
    with open(acc_file, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "accuracy": acc,
                "sample_accuracy": sample_acc,
                "correct_questions": correct_groups,
                "total_questions": total_groups,
                "total_samples": total_samples,
            },
            handle,
            indent=2,
        )
    print(f"Detailed metrics saved to {acc_file}")


def eval_mme(results_file=None, results=None, output_dir=None):
    eval_type_dict = {
        "Perception": [
            "existence",
            "count",
            "position",
            "color",
            "posters",
            "celebrity",
            "scene",
            "landmark",
            "artwork",
            "OCR",
        ],
        "Cognition": [
            "commonsense_reasoning",
            "numerical_calculation",
            "text_translation",
            "code_reasoning",
        ],
    }

    class CalculateMetrics:
        def divide_chunks(self, values, chunk_size=2):
            for idx in range(0, len(values), chunk_size):
                yield values[idx : idx + chunk_size]

        def parse_pred_ans(self, pred_ans):
            if pred_ans in ["yes", "no"]:
                return pred_ans
            prefix = pred_ans[:4]
            if "yes" in prefix:
                return "yes"
            if "no" in prefix:
                return "no"
            return "other"

        def compute_metric(self, gts, preds):
            label_map = {"yes": 1, "no": 0, "other": -1}
            gts = [label_map[x] for x in gts]
            preds = [label_map[x] for x in preds]
            acc = accuracy_score(gts, preds)

            clean_gts = []
            clean_preds = []
            other_num = 0
            for gt, pred in zip(gts, preds):
                if pred == -1:
                    other_num += 1
                    continue
                clean_gts.append(gt)
                clean_preds.append(pred)

            conf_mat = confusion_matrix(clean_gts, clean_preds, labels=[1, 0])
            precision = precision_score(clean_gts, clean_preds, average="binary")
            recall = recall_score(clean_gts, clean_preds, average="binary")
            tp, fn = conf_mat[0]
            fp, tn = conf_mat[1]
            return {
                "TP": tp,
                "FN": fn,
                "TN": tn,
                "FP": fp,
                "precision": precision,
                "recall": recall,
                "other_num": other_num,
                "acc": acc,
            }

        def process_result(self, results_dir):
            for eval_type, task_name_list in eval_type_dict.items():
                print("===========", eval_type, "===========")
                scores = 0
                task_score_dict = {}

                for task_name in task_name_list:
                    task_txt = os.path.join(results_dir, task_name + ".txt")
                    lines = open(task_txt, "r", encoding="utf-8").readlines()
                    chunk_lines = list(self.divide_chunks(lines))

                    img_num = len(chunk_lines)
                    acc_plus_correct_num = 0
                    gts = []
                    preds = []

                    for img_items in chunk_lines:
                        if len(img_items) != 2:
                            continue
                        img_correct_num = 0
                        for img_item in img_items:
                            _img_name, _question, gt_ans, pred_ans = img_item.split("\t")
                            gt_ans = gt_ans.lower()
                            pred_ans = self.parse_pred_ans(pred_ans.lower())
                            gts.append(gt_ans)
                            preds.append(pred_ans)
                            if gt_ans == pred_ans:
                                img_correct_num += 1
                        if img_correct_num == 2:
                            acc_plus_correct_num += 1

                    metric_dict = self.compute_metric(gts, preds)
                    metric_dict["acc_plus"] = acc_plus_correct_num / img_num if img_num > 0 else 0.0
                    task_score = 0
                    for key, value in metric_dict.items():
                        if key in ["acc", "acc_plus"]:
                            task_score += value * 100
                    task_score_dict[task_name] = task_score
                    scores += task_score

                print("total score:", scores, "\n")
                for task_name, score in task_score_dict.items():
                    print("\t", task_name, " score:", score)
                print("\n")

    if results is not None:
        return eval_realworldqa(results=results, output_dir=output_dir)

    if not os.path.exists(results_file):
        raise FileNotFoundError(f"Results file not found: {results_file}")
    with open(results_file, "r", encoding="utf-8") as handle:
        results = json.load(handle)

    _ = results
    calculator = CalculateMetrics()
    calculator.process_result(results_file)
