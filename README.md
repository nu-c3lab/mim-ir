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

The following installation instructions assume you are working with a new virtual environment. All code was tested on 
machine running Ubuntu 18.04.6 with Python 3.9.

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

### Setup ElasticSearch and Document Index

The following command will run the `setup.sh` script that downloads and creates a document index in ElasticSearch which is used as the knowledge source for retrieval.

```commandline
bash setup.sh
```

This process requires a significant amount of disk space, and we recommend running it on a system with at least 100GB of open space.

## Usage

To run an instance of Mim using Python, you can use the following code.

```python
from mim_core.components.Mim import Mim

system = Mim(config="mim_core/evaluation/MimAnalyticsQuestions/mim_config.json")

answer, plan = system.answer_question("What is the capital of Illinois?")

print(answer)
```

## Running with User Interface
This section details how to run an instance of Mim with the provided GUI.

First, you'll need to set up the environment variable:
```commandline
export FLASK_APP=flaskr
```

Then, you can run the following command from the base directory of this repo in order to start FLask app:
```commandline
flask run
```

The application should now be running at `http://127.0.0.1:5000/`. Navigate there to try out the app.
