import enum


# Five-valued logic constants
class LogicValue(enum.IntEnum):
    ZERO = 0  # Logic 0
    ONE = 1  # Logic 1
    XD = 2  # Unknown/Don't care
    D = 3  # D (good=1, faulty=0)
    DB = 4  # D-bar (good=0, faulty=1)


# Gate type constants
class GateType(enum.IntEnum):
    INPT = 1  # Primary Input
    FROM = 2  # STEM BRANCH
    BUFF = 3  # BUFFER
    NOT = 4  # INVERTER
    AND = 5  # AND
    NAND = 6  # NAND
    OR = 7  # OR
    NOR = 8  # NOR
    XOR = 9  # XOR
    XNOR = 10  # XNOR


class Gate:
    """Gate structure similar to C implementation"""

    def __init__(
        self,
        name: str = "",
        gate_type: int = 0,
        nfi: int = 0,
        nfo: int = 0,
        mark: int = 0,
        val: int = LogicValue.XD,
    ):
        self.name = name
        self.type = gate_type
        self.nfi = nfi  # number of fanins
        self.nfo = nfo  # number of fanouts
        self.mark = mark
        self.val = val
        self.fin = []  # fanin list
        self.fot = []  # fanout list
        self.cc0 = -1  # Controllability to 0
        self.cc1 = -1  # Controllability to 1
        self.co = -1  # Observability

    def __str__(self):
        return f"Gate(name={self.name}, type={self.type}, nfi={self.nfi}, nfo={self.nfo}, mark={self.mark}, val={self.val}, fin={self.fin}, fot={self.fot}, cc0={self.cc0}, cc1={self.cc1}, co={self.co})"

    def __repr__(self):
        return self.__str__()