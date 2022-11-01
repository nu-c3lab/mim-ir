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

import re

'''
Question Decomposition Argument Parser

Author(s): AI2 Israel
Edited by Chris Coleman on 10.12.2021
    - parser now includes RelationalProject operator
    - new aggregate types for temporal resolution
'''

DELIMITER = ';'
REF = '#'


def parse_decomposition(qdmr):
    """Parses the decomposition into an ordered list of steps

    Parameters
    ----------
    qdmr : str
        String representation of the QDMR

    Returns
    -------
    list
        returns ordered list of qdmr steps
    """
    # parse commas as separate tokens
    qdmr = qdmr.replace(",", " , ")
    crude_steps = qdmr.split(DELIMITER)
    steps = []
    for i in range(len(crude_steps)):
        step = crude_steps[i]
        tokens = step.split()
        step = ""
        # remove 'return' prefix
        for tok in tokens[1:]:
            step += tok.strip() + " "
        step = step.strip()
        steps += [step]
    return steps


class IdentifyOperator(object):
    def __init__(self):
        self.step = None
        self.operator = None
        self.references = None

    def extract_references(self, step):
        """Extracts a list of references to previous steps"""
        # make sure decomposition does not contain a mere '# ' other than a reference.
        step = step.replace("# ", "hashtag ")
        references = []
        l = step.split(REF)
        for chunk in l[1:]:
            if len(chunk) > 1:
                ref = chunk.split()[0]
                ref = int(ref)
                references += [ref]
            if len(chunk) == 1:
                ref = int(chunk)
                references += [ref]
        return references

    def extract_aggregate(self):
        """Extract aggregate expression from QDMR step

        Returns
        -------
        str
            string of the aggregate: max/min/count/sum/avg.
        """
        for aggregate in ['max', 'highest', 'largest', 'most', 'longest', 'bigger', 'greater', 'greatest',
                          'biggest', 'more', 'longer', 'higher', 'larger']:
            if aggregate in self.step:
                return "max"
        for aggregate in ['min', 'lowest', 'smallest', 'least', 'shortest',
                          'less', 'shorter', 'lower', 'fewer', 'smaller']:
            if aggregate in self.step:
                return "min"
        for aggregate in ['number of']:
            if aggregate in self.step:
                return "count"
        for aggregate in ['sum', 'total']:
            if aggregate in self.step:
                return "sum"
        for aggregate in ['avg', 'average', 'mean ']:
            if aggregate in self.step:
                return "avg"
        for aggregate in ['second', 'later', 'after', 'latest', 'last']:
            if aggregate in self.step:
                return "later"
        for aggregate in ['first', 'before', 'earlier', 'earliest']:
            if aggregate in self.step:
                return "before"
        return None

    def identify_op(self, step):
        self.step = step
        self.references = self.extract_references(self.step)
        self.operator = self._identify_op()
        return self.operator

    def extract_args(self, step):
        args = self._extract_args()
        return [a.strip() for a in args]

    def _identify_op(self):
        raise NotImplementedError
        return True

    def _extract_args(self):
        raise NotImplementedError
        return True


class IdentifyOperatorSelect(IdentifyOperator):
    """
    Example: "countries"
    """

    def __init__(self):
        super(IdentifyOperatorSelect, self).__init__()

    def _identify_op(self):
        # SELECT step has no references to previous steps.
        if len(self.references) == 0:
            return True
        return False

    def _extract_args(self):
        args = [self.step]
        return args


class IdentifyOperatorFilter(IdentifyOperator):
    """
    Example: "#2 that is wearing #3"
    Example: "#1 from Canada"
    """

    def __init__(self):
        super(IdentifyOperatorFilter, self).__init__()

    def _identify_op(self):
        # FILTER starts with '#'
        refs = len(self.references)
        if refs > 0 and refs <= 3 and self.step.startswith("#"):
            return True
        return False

    def _extract_args(self):
        # extract the reference to be filtered
        to_filter = "#%s" % self.references[0]
        # extract the filter condition
        filter_condition = self.step.split(to_filter)[1]
        return [to_filter, filter_condition]


class IdentifyOperatorProject(IdentifyOperator):
    """
    Example: "first name of #2"
    Example: "who was #1 married to"
    """

    def __init__(self):
        super(IdentifyOperatorProject, self).__init__()

    def _identify_op(self):
        if len(self.references) == 1 and \
                re.search("[\s]+[#]+[0-9\s]+", self.step):
            return True
        return False

    def _extract_args(self):
        # extract the referenced objects
        ref = "#%s" % self.references[0]
        # extract the projected relation phrase, anonymized
        projection = self.step.replace(ref, "#REF")
        return [projection, ref]


class IdentifyOperatorRelationalProject(IdentifyOperator):
    """
    Example: "distance between #1 and #2"
    Example: "full years passed between #1 and #2"
    """

    def __init__(self):
        super(IdentifyOperatorRelationalProject, self).__init__()

    def _identify_op(self):
        if len(self.references) > 1 and any([x in self.step for x in [' from ', ' between ']]):
            return True
        return False

    def _extract_args(self):
        # extract the referenced objects
        refs = ['#' + str(x) for x in self.references]
        projection = None
        if ' from ' in self.step:
            projection = self.step.split(' from ')[0]
        elif ' between ' in self.step:
            projection = self.step.split(' between ')[0]
        return [projection] + refs


class IdentifyOperatorAggregate(IdentifyOperator):
    """
    Example: "lowest of #2"
    Example: "the number of #1"
    """

    def __init__(self):
        super(IdentifyOperatorAggregate, self).__init__()

    def _identify_op(self):
        # AGGREGATION step - aggregation applied to one reference
        if len(self.references) != 1:
            return False
        aggregators = ['number of', 'highest', 'largest', 'lowest', 'smallest', 'maximum', 'minimum', \
                       'max', 'min', 'sum', 'total', 'average', 'avg', 'mean of', 'first', 'last', \
                       'longest', 'shortest']
        for aggr in aggregators:
            aggr_ref = aggr + ' #'
            aggr_of_ref = aggr + ' of #'
            if (aggr_ref in self.step) or (aggr_of_ref in self.step):
                return True
        return False

    def _extract_args(self):
        # extract the referenced objects
        ref = "#%s" % self.references[0]
        # extract the aggregate function
        aggregate = self.extract_aggregate()
        return [aggregate, ref]


class IdentifyOperatorGroup(IdentifyOperator):
    """
    Example: "number of #3 for each #2"
    Example: "average of #1 for each #2"
    """

    def __init__(self):
        super(IdentifyOperatorGroup, self).__init__()

    def _identify_op(self):
        # GROUP step - contains the phrase 'for each'
        if 'for each' in self.step and len(self.references) > 0:
            return True
        return False

    def _extract_args(self):
        aggregate = self.extract_aggregate()
        # need to extract the group values and keys
        # split the step to the aggregated values (prefix) and keys (suffix)
        value, key = self.step.split('for each')
        val_refs = self.extract_references(value)
        key_refs = self.extract_references(key)
        # check if both parts actually contained references
        arg_value = value.split()[-1] if len(val_refs) == 0 else "#%s" % val_refs[0]
        arg_key = key.split()[-1] if len(key_refs) == 0 else "#%s" % key_refs[0]
        return [aggregate, arg_value, arg_key]


class IdentifyOperatorSuperlative(IdentifyOperator):
    """
    Example: "#1 where #2 is highest"
    Example: "#1 where #2 is smallest"
    """

    def __init__(self):
        super(IdentifyOperatorSuperlative, self).__init__()

    def _identify_op(self):
        superlatives = ['highest', 'largest', 'most', 'smallest', 'lowest', 'smallest', 'least', \
                        'longest', 'shortest', 'biggest']
        superlatives_is = ["is %s" % sup for sup in superlatives]
        superlatives_are = ["are %s" % sup for sup in superlatives]
        superlatives = superlatives_is + superlatives_are
        if self.step.startswith('#') and len(self.references) == 2 \
                and ('where' in self.step) and (self.step.startswith('#') or self.step.startswith('the #')):
            for s in superlatives:
                if s in self.step:
                    return True
        return False

    def _extract_args(self):
        aggregate = self.extract_aggregate()
        entity_ref, attribute_ref = self.references
        return [aggregate, "#%s" % entity_ref, "#%s" % attribute_ref]


class IdentifyOperatorComparative(IdentifyOperator):
    """
    Example: "#1 where #2 is at most three"
    Example: "#3 where #4 is higher than #2"
    """

    def __init__(self):
        super(IdentifyOperatorComparative, self).__init__()

    def _identify_op(self):
        comparatives = ['same as', 'higher than', 'larger than', 'smaller than', 'lower than', \
                        'more', 'less', 'at least', 'at most', 'equal', 'is ', 'are', 'was', 'contain', \
                        'include', 'has', 'have', 'end with', 'start with', 'ends with', \
                        'starts with', 'begin']
        if len(self.references) >= 2 and len(self.references) <= 3 \
                and ('where' in self.step) and (self.step.startswith('#') or self.step.startswith('the #')):
            for comp in comparatives:
                if comp in self.step:
                    return True
        return False

    def _extract_args(self):
        to_filter = "#%s" % self.references[0]
        attribute = "#%s" % self.references[1]
        condition = self.step.split(attribute)[1]
        return [to_filter, attribute, condition]


class IdentifyOperatorUnion(IdentifyOperator):
    """
    Example: "#1 or #2"
    Example: "#1, #2, #3, #4"
    Example: "#1 and #2"
    """

    def __init__(self):
        super(IdentifyOperatorUnion, self).__init__()

    def _identify_op(self):
        if len(self.references) > 1:
            substitute_step = self.step.replace('and', ',').replace('or', ',')
            is_union = re.search("^[#0-9,\s]+$", substitute_step)
            return is_union
        return False

    def _extract_args(self):
        args = []
        for ref in self.references:
            args += ["#%s" % ref]
        return args


class IdentifyOperatorIntersect(IdentifyOperator):
    """
    Example: "countries in both #1 and #2"
    Example: "#3 of both #4 and #5"
    """

    def __init__(self):
        super(IdentifyOperatorIntersect, self).__init__()

    def _identify_op(self):
        if len(self.references) >= 2 and ('both' in self.step) and (' and' in self.step):
            return True
        return False

    def _extract_args(self):
        interesect_expr = None
        for expr in ['of both', 'in both', 'by both', 'between both', 'for both', 'are both', 'both of', 'at both']:
            if expr in self.step:
                interesect_expr = expr
        if interesect_expr is not None:
            projection, intersection = self.step.split(interesect_expr)
            args = [projection]
            # add all previous references as the intersection arguments
            refs = self.extract_references(intersection)
            for ref in refs:
                args += ["#%s" % ref]
            return args
        else:
            if len(self.references) > 2:
                # TODO: check for existence of 3-reference cases with "both"
                # Handle parsing with commas if case exists
                pass
            refs = self.step.split(' and ')
            refs[0] = refs[0].replace('both ', '')
            return refs


class IdentifyOperatorDiscard(IdentifyOperator):
    """
    Example: "#2 besides #3"
    Exmple: "#1 besides cats"
    """

    def __init__(self):
        super(IdentifyOperatorDiscard, self).__init__()

    def _identify_op(self):
        if (len(self.references) >= 1) and (len(self.references) <= 2) and \
                (re.search("^[#]+[0-9]+[\s]+", self.step) or re.search("[#]+[0-9]+$", self.step)) and \
                ('besides' in self.step or 'not in' in self.step):
            return True
        return False

    def _extract_args(self):
        discard_expr = None
        for expr in ['besides', 'not in']:
            if expr in self.step:
                discard_expr = expr
        set_1, set_2 = self.step.split(discard_expr)
        return [set_1, set_2]


class IdentifyOperatorSort(IdentifyOperator):
    """
    Example: "#1 sorted by #2"
    Example: "#1 ordered by #2"
    """

    def __init__(self):
        super(IdentifyOperatorSort, self).__init__()

    def _identify_op(self):
        for expr in [' sorted by', ' order by', ' ordered by']:
            if expr in self.step:
                return True
        return False

    def _extract_args(self):
        sort_expr = None
        for expr in [' sorted by', ' order by', ' ordered by']:
            if expr in self.step:
                sort_expr = expr
        objects, order = [frag.strip() for frag in self.step.split(sort_expr)]
        return [objects, order]


class IdentifyOperatorBoolean(IdentifyOperator):
    """
    Example: "if both #2 and #3 are true"
    Example: "is #2 more than #3"
    Example: "if #1 is american"
    """

    def __init__(self):
        super(IdentifyOperatorBoolean, self).__init__()

    def _identify_op(self):
        # BOOLEAN step - starts with either 'if', 'is' or 'are'
        if self.step.lower().startswith('if ') or self.step.lower().startswith('is ') or \
                self.step.lower().startswith('are ') or self.step.lower().startswith('did '):
            return True
        return False

    def _extract_args(self):
        # logical or/and boolean steps, e.g., "if either #1 or #2 are true"
        logical_op = None
        if len(self.references) == 2 and "both" in self.step and "and" in self.step:
            logical_op = "logical_and"
        elif len(self.references) == 2 and "either" in self.step and "or" in self.step:
            logical_op = "logical_or"
        if logical_op is not None:
            bool_expr = "false" if "false" in self.step else "true"
            sub_expressions = ["#%s" % ref for ref in self.references]
            return [logical_op, bool_expr] + sub_expressions
        elif self.step.split()[1].startswith("#"):
            # filter boolean, e.g., "if #1 is american"
            objects = "#%s" % self.references[0]
            condition = self.step.split(objects)[1]
            return [objects, condition]
        elif len(self.references) == 1 \
                and not self.step.split()[1].startswith("#"):
            # projection boolean "if dinner is served on #1"
            objects = "#%s" % self.references[0]
            condition = self.step.replace(objects, "#REF")
            return [objects, condition]
        elif len(self.references) == 2:
            objects = "#%s" % self.references[0]
            prefix = self.step.split(objects)[0].lower()
            if "any" in prefix or "is there" in prefix \
                    or "there is" in prefix or "there are" in prefix:
                # exists boolean "if any #2 are the same as #3"
                condition = self.step.split(objects)[1]
                return ["if_exist", objects, condition]
        elif len(self.references) == 0:
            return [self.step]
        else:
            return None


class IdentifyOperatorArithmetic(IdentifyOperator):
    """
    Example: "difference of #3 and #5"
    """

    def __init__(self):
        super(IdentifyOperatorArithmetic, self).__init__()

    def _identify_op(self):
        # ARITHMETIC step - starts with arithmetic operation
        arithmetics = ['sum', 'difference', 'multiplication', 'division', 'percentage difference']
        for a in arithmetics:
            if (self.step.startswith(a) or self.step.startswith('the ' + a)) \
                    and len(self.references) > 1:
                return True
        return False

    def _extract_args(self):
        arithmetic = None
        for a in ['sum', 'difference', 'multiplication', 'division']:
            if a in self.step:
                arithmetic = a
        # arithmetic with constant number, e.g. "difference of 100 and #1"
        if self.references == 1:
            prefix, suffix = self.step.split('and')
            first_arg = prefix.split()[-1]
            return [arithmetic, first_arg, suffix]
        # arithmetic with references, e.g. "difference of #3 and #1"
        else:
            refs = ['#%d' % ref for ref in self.references]
            return [arithmetic] + refs


class IdentifyOperatorComparison(IdentifyOperator):
    """
    Example: "which is highest of #1, #2"
    """

    def __init__(self):
        super(IdentifyOperatorComparison, self).__init__()

    def _identify_op(self):
        # COMPARISON step - 'which is better A or B or C'
        if self.step.lower().startswith('which') and len(self.references) > 1:
            return True
        return False

    def _extract_args(self):
        comp = self.extract_aggregate()
        # check if boolean comparison "which is true of #1, #2"
        if comp is None and \
                ("true" in self.step or "false" in self.step):
            comp = "true" if "true" in self.step else "false"
        assert (comp is not None)
        args = ["#%s" % ref for ref in self.references]
        return [comp] + args


class QDMRStep:
    def __init__(self, step_text, operator, arguments):
        self.step = step_text
        self.operator = operator
        self.arguments = arguments

    def __str__(self):
        return "%s%a" % (self.operator.upper(), self.arguments)


class StepIdentifier(object):
    def __init__(self):
        self.identifiers = {"select": IdentifyOperatorSelect(),
                            "filter": IdentifyOperatorFilter(),
                            "project": IdentifyOperatorProject(),
                            "relationalproject": IdentifyOperatorRelationalProject(),
                            "aggregate": IdentifyOperatorAggregate(),
                            "group": IdentifyOperatorGroup(),
                            "superlative": IdentifyOperatorSuperlative(),
                            "comparative": IdentifyOperatorComparative(),
                            "union": IdentifyOperatorUnion(),
                            "intersection": IdentifyOperatorIntersect(),
                            "discard": IdentifyOperatorDiscard(),
                            "sort": IdentifyOperatorSort(),
                            "boolean": IdentifyOperatorBoolean(),
                            "arithmetic": IdentifyOperatorArithmetic(),
                            "comparison": IdentifyOperatorComparison()}
        self.operator = None

    def step_type(self, step_text):
        potential_operators = set()
        for op in self.identifiers:
            identifier = self.identifiers[op]
            if identifier.identify_op(step_text):
                potential_operators.add(op)
        # no matching operator found
        if len(potential_operators) == 0:
            return None
        operators = potential_operators.copy()
        # duplicate candidates
        if len(operators) > 1:
            # avoid project duplicity with aggregate
            if "project" in operators:
                operators.remove("project")
            # # TODO: decide ordering
            # if "relationalproject" in operators:
            #     operators.remove("relationalproject")
            # avoid filter duplcitiy with comparative, superlative, sort, discard
            if "filter" in operators:
                operators.remove("filter")
            # return boolean (instead of intersect)
            if "boolean" in operators:
                operators = {"boolean"}
            # return intersect (instead of filter)
            if "intersect" in operators:
                operators = {"intersect"}
            # return superlative (instead of comparative)
            if "superlative" in operators:
                operators = {"superlative"}
            # return group (instead of arithmetic)
            if "group" in operators:
                operators = {"group"}
            # return comparative (instead of discard)
            if "comparative" in operators:
                operators = {"comparative"}
            # return intersection (instead of comparison)
            if "intersection" in operators:
                operators = {"intersection"}
        assert (len(operators) == 1)
        operator = list(operators)[0]
        self.operator = operator
        return operator

    def step_args(self, step_text):
        self.operator = self.step_type(step_text)
        identifier = self.identifiers[self.operator]
        args = identifier.extract_args(step_text)
        return args

    def identify(self, step_text):
        self.operator = self.step_type(step_text)
        args = self.step_args(step_text)
        return QDMRStep(step_text, self.operator, args)


class QDMRProgramBuilder(object):
    def __init__(self, qdmr_text):
        self.qdmr_text = qdmr_text
        self.steps = None
        self.operators = None
        self.program = None

    def build(self):
        try:
            self.get_operators()
            self.build_steps()
        except:
            # print("Unable to identify all steps: %s" % self.qdmr_text)
            pass
        return True

    def build_steps(self):
        self.steps = []
        steps = parse_decomposition(self.qdmr_text)
        step_identifier = StepIdentifier()
        for step_text in steps:
            try:
                step = step_identifier.identify(step_text)
            except:
                # print("Unable to identify step: %s" % step_text)
                step = None
            self.steps += [step]
        return self.steps

    def get_operators(self):
        self.operators = []
        steps = parse_decomposition(self.qdmr_text)
        step_identifier = StepIdentifier()
        for step_text in steps:
            try:
                op = step_identifier.step_type(step_text)
            except:
                # print("Unable to identify operator: %s" % step_text)
                op = None
            self.operators += [op]
        return self.operators
