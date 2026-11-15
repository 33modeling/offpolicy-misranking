"""One GPU set that crashes natively in different cohorts stops taking tasks (a node serves several cohorts
since the 11-15 spill). CPU metadata only."""
import types

import srgc_v8_nodes as nodes


def pools(tmp_path):
    made={}
    for cohort in ('olmo_math','qwen_mbpp','olmo_mbpp'):
        pool=nodes.Pool(tmp_path/cohort/'.multi-node',dict(cohort=cohort));pool.initialize();made[cohort]=pool
    return made,types.SimpleNamespace(gpus=['0','1','2','3'])


def crash(pool,runtime,name):
    nodes.record_node_native_fault(pool,runtime,dict(id=name,exit_code=-11,worker='h.1'))


def test_crashes_in_two_cohorts_hold_the_gpu_set_for_every_cohort(tmp_path):
    made,runtime=pools(tmp_path)
    crash(made['olmo_math'],runtime,'a1')
    assert all(nodes.node_native_hold(p,runtime) is None for p in made.values())
    crash(made['qwen_mbpp'],runtime,'b1')
    for pool in made.values():
        hold=nodes.node_native_hold(pool,runtime)
        assert hold and '2 native crashes across cohorts (olmo_math, qwen_mbpp)' in hold


def test_other_gpu_set_of_the_host_is_not_held(tmp_path):
    made,runtime=pools(tmp_path)
    crash(made['olmo_math'],runtime,'a1');crash(made['qwen_mbpp'],runtime,'b1')
    other=types.SimpleNamespace(gpus=['4','5','6','7'])
    assert all(nodes.node_native_hold(p,other) is None for p in made.values())
