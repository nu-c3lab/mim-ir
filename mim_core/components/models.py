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

import spacy

def get_model(model: str, extraction: bool = True):
    # Load the SpaCy models on the first GPU, if available
    if model not in spacy_models:
        if spacy.prefer_gpu(0):
            print("Loading SpaCy model {} to GPU.".format(model))
        else:
            print("Loading SpaCy model {} to (non-GPU) memory.".format(model))

        nlp = spacy.load(model)
        if extraction:
            nlp.add_pipe("entityLinker", last=True)
        spacy_models[model] = nlp
    return spacy_models[model]
    
spacy_models = {}

