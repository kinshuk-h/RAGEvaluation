import enum

concept_transforms = [
    lambda x: x,
    lambda x: x.lower(),
    lambda x: x.upper(),
    lambda x: x[0].upper(),
    lambda x: x[0].lower()
]

class DecisionType(enum.Enum):
    TRUE_OR_FALSE        = 'tof'
    YES_OR_NO            = 'yon'
    CORRECT_OR_INCORRECT = 'coi'
    OPTION_A_OR_B        = 'aob'
    OPTION_1_OR_2        = '1o2'

    @classmethod
    def values(cls):
        return map(lambda x: x.value, cls._member_map_.values())

    @classmethod
    def keys(cls):
        return cls._member_names_

    @property
    def positive(self):
        match self.value:
            case 'yon': return 'Yes'
            case 'tof': return 'True'
            case 'coi': return 'Correct'
            case 'aob': return 'A'
            case '1o2': return '1'

    @property
    def negative(self):
        match self.value:
            case 'yon': return 'No'
            case 'tof': return 'False'
            case 'coi': return 'Incorrect'
            case 'aob': return 'B'
            case '1o2': return '2'

    @property
    def completion(self):
        match self.value:
            case 'aob': return 'The answer is'
            case '1o2': return 'The answer is'
            case 'yon': return 'The answer is'
            case _: return 'The statement is'

    @property
    def positive_concept(self):
        value = self.positive
        if self.value in ('aob', '1o2'):
            concept_list = set([ value.upper(), value.lower(), 'True' ])
        else:
            concept_list = set(transform(value) for transform in concept_transforms)
        if value in concept_list: concept_list.remove(value)
        return [ value, *concept_list ]

    @property
    def negative_concept(self):
        value = self.negative
        if self.value in ('aob', '1o2'):
            concept_list = set([ value.upper(), value.lower(), 'False' ])
        else:
            concept_list = set(transform(value) for transform in concept_transforms)
        if value in concept_list: concept_list.remove(value)
        return [ value, *concept_list ]
