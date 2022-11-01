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
Analyze Errors

Author(s): Marko Sterbentz

Usage:
From the mim_core/evaluation/MimAnalyticsQuestions directory, run the following command:
    python analyze_errors.py <path/to/plans.json>
"""

import json
import plac
from typing import List, Dict


def get_questions(q_ids: List[str]) -> List[Dict]:
    # Read in the original question data
    with open('./data/analytic_data.json') as f:
        data = json.load(f)

    questions = []
    for ex in data:
        if ex['id'] in q_ids:
            questions.append(ex)

    return questions


@plac.annotations(
    plan_file=('Path to a JSON file of MQR/plans.', 'positional', None, str)
)
def main(plan_file: str):
    # Read in the json file of the HotpotQA questions
    with open(plan_file) as f:
        # Read in the question file
        data = json.load(f)

        formatted_errors = []
        errors_to_ignore = ("NoRelationshipFoundWarning",
                            "MultipleRelationshipsFoundWarning",
                            # "ValueError: min() arg is an empty sequence",
                            # "UnexpectedOperationArgsError",
                            # "MissingPreviousStepDataError",
                            # "TypeError",
                            # "UnhandledSubOperationTypeError",
                            # "ValueError: could not convert string to float: 'twenty'",
                            # "ValueError: could not convert string to float: 'Phil King (bass) and Chris Acland (drums).'"
                            )

        for p in data.values():
            q_id = p['plan']['q_id']
            question = p['plan']['steps'][0]['qdmr']
            mqr_errors = p['plan']['errors']
            for s in p['plan']['steps']:
                mqr_errors.extend(s['errors'])

            # Filter out errors to ignore
            final_errors = [x for x in mqr_errors if not x.startswith(errors_to_ignore)]

            if final_errors:
                formatted_errors.append({
                    "question": question,
                    "q_id": q_id,
                    "errors": final_errors
                })

        print("{} questions with errors found".format(len(formatted_errors)))
        error_counts = {}
        for q in formatted_errors:
            for error in q["errors"]:
                error_counts[error] = error_counts.get(error, 0) + 1
        print("Error Counts:")
        print(json.dumps(dict(sorted(error_counts.items(), key=lambda item: item[1], reverse=True)), indent=4))

        error_count_by_operator = {"addition": 0, "subtraction": 0, "boolean_equality": 0, "boolean_existence": 0, "comparison": 0, "discard": 0, "intersection": 0, "bridge": 0}
        for q in formatted_errors:
            question = get_questions([q["q_id"]])[0]
            error_count_by_operator[question["operator"]] += 1
        print("Error Counts by Operator:")
        print(json.dumps(error_count_by_operator, indent=4))

        # Write out the analysis file
        with open("analysis/error_analysis.json", "w") as outfile:
            outfile.write(json.dumps(formatted_errors, ensure_ascii=False, indent=4))

        # Write out question file for testing
        error_questions = get_questions([q['q_id'] for q in formatted_errors])
        with open("analysis/error_questions.json", "w") as outfile:
            outfile.write((json.dumps(error_questions, ensure_ascii=False)))


if __name__ == "__main__":
    plac.call(main)