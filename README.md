# Mim IR
Paper link coming soon...

### License

This file is part of Mim.
Mim is free software: you can redistribute it and/or modify it under 
the terms of the GNU General Public License as published by the Free Software Foundation, 
either version 3 of the License, or (at your option) any later version.
Mim is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; 
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 
See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with Mim. 
If not, see <https://www.gnu.org/licenses/>.

## Installation

The following installation instructions assume you are working with a new virtual environment.

### Install requirements to virtual environment

From the Mim repository's root directory, install by running:
```commandline
pip install .
```

### Install other packages

#### SpaCy Models:
```commandline
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_md
python -m spacy download en_core_web_trf
```

## Usage

```python
from mim_core.components.Mim import Mim

system = Mim(config="mim_core/evaluation/MimAnalyticsQuestions/mim_config.json")

answer, plan = system.answer_question("What is the capital of Illinois?")

print(answer)
```
