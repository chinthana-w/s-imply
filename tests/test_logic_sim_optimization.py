from src.atpg.logic_sim_three import d_frontier, fault_is_at_po, logic_sim, reset_gates
from src.util.struct import Fault, Gate, GateType, LogicValue


def _make_fanout_circuit():
    circuit = [None] * 5
    circuit[0] = Gate("dummy", GateType.INPT, 0, 0)
    circuit[1] = Gate("a", GateType.INPT, 0, 1)
    circuit[2] = Gate("b", GateType.INPT, 0, 1)
    circuit[3] = Gate("and", GateType.AND, 2, 1)
    circuit[4] = Gate("out", GateType.BUFF, 1, 0)
    circuit[3].fin = [1, 2]
    circuit[4].fin = [3]
    circuit[1].fot = [3]
    circuit[2].fot = [3]
    circuit[3].fot = [4]
    return circuit, 4


def test_d_frontier_keeps_unique_membership_after_sort_and_clear():
    d_frontier.clear()

    d_frontier.add(10)
    d_frontier.add(10)
    d_frontier.add(3)
    d_frontier.sort(lambda gid: gid)

    assert d_frontier.gates == [3, 10]
    assert d_frontier.get_first() == 3

    d_frontier.remove(3)
    d_frontier.add(3)
    assert d_frontier.gates == [10, 3]

    d_frontier.clear()
    assert d_frontier.gates == []
    assert d_frontier.is_empty()


def test_fault_is_at_po_uses_cached_outputs_without_changing_result():
    circuit, total_gates = _make_fanout_circuit()
    reset_gates(circuit, total_gates)

    assert not fault_is_at_po(circuit, total_gates)

    circuit[4].val = LogicValue.D
    assert fault_is_at_po(circuit, total_gates)

    circuit[4].val = LogicValue.DB
    assert fault_is_at_po(circuit, total_gates)


def test_logic_sim_with_topo_order_preserves_d_frontier_behavior():
    circuit, total_gates = _make_fanout_circuit()
    reset_gates(circuit, total_gates)

    circuit[1].val = LogicValue.ONE
    circuit[2].val = LogicValue.XD
    logic_sim(circuit, total_gates, Fault(1, LogicValue.D), topo_order=[1, 2, 3, 4])

    assert circuit[1].val == LogicValue.D
    assert circuit[3].val == LogicValue.XD
    assert circuit[4].val == LogicValue.XD
    assert d_frontier.gates == [3]
