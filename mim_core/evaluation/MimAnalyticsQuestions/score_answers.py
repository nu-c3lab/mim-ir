'''
This file is part of Mim.
Mim is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software Foundation,
either version 3 of the License, or (at your option) any later version.
Mim is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with Mim.
If not, see <https://www.gnu.org/licenses/>.
'''
"""
Note: This is copied from the original evaluation script here: https://github.com/beerqa/IRRR/blob/main/utils/eval_beerqa.py
"""
from __future__ import print_function
from collections import Counter
import string
import re
import argparse
import json
import random
from num_parse.NumParser import NumParser

num_parser = NumParser()

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def f1_score_normalized(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)

    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall

def exact_match_score_numeric(prediction, ground_truth):
    try:
        pred = num_parser.parse_num(prediction)
        gt = num_parser.parse_num(ground_truth)
        return (pred == gt)
    except Exception as e:
        return 0.0

def threshold_match_score(prediction, ground_truth, threshold):
    try:
        # Parse in the two values using the numeric parser
        pred = num_parser.parse_num(prediction)
        gt = num_parser.parse_num(ground_truth)

        # Check if the prediction is within the given threshold of the ground_truth value
        gt_1 = gt + (gt * threshold)
        gt_2 = gt - (gt * threshold)
        if gt_1 < gt_2:
            gt_lower, gt_upper = gt_1, gt_2
        else:
            gt_lower, gt_upper = gt_2, gt_1

        ret = (pred >= gt_lower and pred <= gt_upper)

        return ret
    except Exception as e:
        return 0.0

def update_answer(metrics, prediction, gold, prefix=None):
    best_f1 = 0.0
    best_em = 0.0
    best_prec = 0.0
    best_recall = 0.0
    for gold_answer in gold:
        em = exact_match_score(prediction, gold_answer)
        f1, prec, recall = f1_score_normalized(prediction, gold_answer)

        if best_f1 < f1:
            best_f1 = f1
            best_em = em
            best_prec = prec
            best_recall = recall

    metrics['em'] += float(best_em)
    metrics['f1'] += best_f1
    metrics['prec'] += best_prec
    metrics['recall'] += best_recall

    if prefix is not None:
        metrics[f'{prefix}_em'] += float(best_em)
        metrics[f'{prefix}_f1'] += best_f1
        metrics[f'{prefix}_prec'] += best_prec
        metrics[f'{prefix}_recall'] += best_recall

    return best_em, best_prec, best_recall

def update_answer_numeric(metrics, prediction, gold, threshold, prefix=None):
    best_threshold_match = 0.0
    best_em = 0.0
    for gold_answer in gold:
        em = exact_match_score_numeric(prediction, gold_answer)
        threshold_match = threshold_match_score(prediction, gold_answer, threshold)

        if best_threshold_match < threshold_match:
            best_threshold_match = threshold_match
            best_em = em

    metrics['em'] += float(best_em)
    metrics['threshold_match'] += best_threshold_match

    if prefix is not None:
        metrics[f'{prefix}_em'] += float(best_em)
        metrics[f'{prefix}_threshold_match'] += best_threshold_match

    return best_em, best_threshold_match

def evaluate(gold_data, prediction, sampled=False):
    metrics = Counter()
    counts = Counter()
    for dp in gold_data:
        cur_id = dp['id']
        counts[dp['operator']] += 1
        if cur_id not in prediction['answers']:
            if sampled is False:
                print('missing answer {}'.format(cur_id))
        else:
            em, prec, recall = update_answer(metrics, prediction['answers'][cur_id], dp['answer'], prefix=dp['operator'])

    if sampled is True:
        N = len(prediction["answers"])
    else:
        N = len(gold_data)
    for k in ['em', 'f1', 'prec', 'recall']:
        metrics[k] /= N
        for prefix in counts.keys():
            metrics[f'{prefix}_{k}'] /= counts[prefix]
            metrics[f'macro_{k}'] += metrics[f'{prefix}_{k}']
        metrics[f'macro_{k}'] /= len(counts.keys())

    return dict(metrics)

def evaluate_numeric(gold_data, prediction, threshold, sampled=False):
    metrics = Counter()
    counts = Counter()
    for dp in gold_data:
        cur_id = dp['id']
        counts[dp['operator']] += 1
        if cur_id not in prediction['answers']:
            if sampled is False:
                print('missing answer {}'.format(cur_id))
        else:
            em, threshold_match = update_answer_numeric(metrics, prediction['answers'][cur_id], dp['answer'], threshold, prefix=dp['operator'])

    if sampled is True:
        N = len(prediction["answers"])
    else:
        N = len(gold_data)
    for k in ['em', 'threshold_match']:
        metrics[k] /= N
        for prefix in counts.keys():
            metrics[f'{prefix}_{k}'] /= counts[prefix]
            metrics[f'macro_{k}'] += metrics[f'{prefix}_{k}']
        metrics[f'macro_{k}'] /= len(counts.keys())

    return dict(metrics)

def compute_full_em_score(metrics, numeric_metrics, other_predictions, numeric_predictions):
    other_em_count = metrics["em"] * len(other_predictions["answers"])
    numeric_em_count = numeric_metrics["em"] * len(numeric_predictions["answers"])

    total_em_count = other_em_count + numeric_em_count
    total_preds = len(other_predictions["answers"]) + len(numeric_predictions["answers"])

    return total_em_count / total_preds

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Merge hop1 and hop2 results.')
    parser.add_argument('gold_answers_file')
    parser.add_argument('answers')
    parser.add_argument('threshold', type=float, default=0.0)
    args = parser.parse_args()

    with open(args.answers) as f:
        answers = json.load(f)

    if 'answers' in answers:
      answers=answers['answers']
    res = {'answers':answers }

    with open(args.gold_answers_file) as f:
        gold_data = json.load(f)

    # Build the separate list of gold data for the numeric vs. non-numeric questions
    numeric_gold_data = [dp for dp in gold_data if dp['operator'] in ['addition', 'subtraction']]
    other_gold_data = [dp for dp in gold_data if dp['operator'] not in ['addition', 'subtraction']]

    # Build the separate answer dictionaries for the numeric vs. non-numeric questions
    numeric_predictions = {"answers": {ex["id"]:res["answers"][ex["id"]] for ex in numeric_gold_data}}
    other_predictions = {"answers": {ex["id"]:res["answers"][ex["id"]] for ex in other_gold_data}}

    metrics = evaluate(other_gold_data, other_predictions, False)
    numeric_metrics = evaluate_numeric(numeric_gold_data, numeric_predictions, args.threshold, False)
    print(json.dumps(metrics, indent=4))
    print(json.dumps(numeric_metrics, indent=4))

    full_exact_match_metric = compute_full_em_score(metrics, numeric_metrics, other_predictions, numeric_predictions)
    print(f"Full EM Score: {full_exact_match_metric}")
