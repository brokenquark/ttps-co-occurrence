from typing import List

class Tactic:
    def __init__(self, id : str, name : str):
        self.id : str = id
        self.name : str = name
        self.techniques : List['Technique'] = []
        self.sequence : int

class Technique:
    def __init__(self, id : str, name : str):
        self.id : str = id
        self.name : str = name
        self.tactics : List['Tactic'] = []
        self.years : List[int] = []

class Procedure:
    def __init__(self, id):
        self.id : str = id
        self.name : str = ''
        self.type : str = ''
        self.year : int = 0
        self.technique : Technique
        self.reference : str = ''

class Group:
    def __init__(self, id):
        self.id : str = id
        self.techniques : List['Technique'] = []
        self.procedures : List['Procedure'] = []

class Software:
    def __init__(self, id):
        self.id : str = id
        self.techniques : List['Technique'] = []
        self.procedures : List['Procedure'] = []

class Tuples:
    def __init__(self, entity, technique) -> None:
        self.entity = entity
        self.techniques = []

