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
Convert to BeerQA Form

A script for converting the Mim Analytics Questions dataset to the same format as the BeerQA dataset.

September 27, 2022

Example Usage:

    python convert_to_beerqa_form.py <path/to/analytic_data.json>

"""

import json
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Converts the Mim Analytics Questions dataset to the BeerQA format.')
    parser.add_argument('original_data')
    args = parser.parse_args()

    # Read in the original data
    with open(args.original_data) as f:
        data = json.load(f)

    # Get only the required fields
    reformatted_data = [{"id": q["id"], "question": q["question"], "answers": q["answer"], "src": "mim"} for q in data]

    final_output = {
        "version": "1.0",
        "data": reformatted_data
    }

    # Write out the data
    with open("data/analytic_data_bqa_form.json", "w") as outfile:
        outfile.write(json.dumps(final_output, ensure_ascii=False))
