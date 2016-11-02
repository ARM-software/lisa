from energy_model import ActiveState, EnergyModelNode, PowerDomain, EnergyModel

from collections import OrderedDict

a53_cluster_active_states = OrderedDict([
    (450000, ActiveState(power=26)),
    (575000, ActiveState(power=30)),
    (700000, ActiveState(power=39)),
    (775000, ActiveState(power=47)),
    (850000, ActiveState(power=57)),
])

# TODO warn if any of the idle states aren't represented by power domains
a53_cluster_idle_states = OrderedDict([
    ("WFI",               56),
    ("cpu-sleep-0",       56),
    ("cluster-sleep-0",   17),
])

a53_cpu_active_states = OrderedDict([
    (450000, ActiveState(capacity=235, power=33)),
    (575000, ActiveState(capacity=302, power=46)),
    (700000, ActiveState(capacity=368, power=61)),
    (775000, ActiveState(capacity=406, power=76)),
    (850000, ActiveState(capacity=447, power=93)),
])

a53_cpu_idle_states = OrderedDict([
    ("WFI",               6),
    ("cpu-sleep-0",       0),
    ("cluster-sleep-0",   0),
])

a53s = [0, 3, 4, 5]
a53_pd = PowerDomain(cpus=a53s, idle_states=["cluster-sleep-0"], parent=None)

def a53_cpu_node(cpu):
    cpu_pd=PowerDomain(cpus=[cpu],
                       parent=a53_pd,
                       idle_states=["WFI", "cpu-sleep-0"])

    return EnergyModelNode([cpu],
                           active_states=a53_cpu_active_states,
                           idle_states=a53_cpu_idle_states,
                           power_domain=cpu_pd,
                           freq_domain=a53s)

a57_cluster_active_states = OrderedDict([
    (450000,  ActiveState(power=24)),
    (625000,  ActiveState(power=32)),
    (800000,  ActiveState(power=43)),
    (950000,  ActiveState(power=49)),
    (1100000, ActiveState(power=64)),
])

a57_cluster_idle_states = OrderedDict([
    ("WFI",               65),
    ("cpu-sleep-0",       65),
    ("cluster-sleep-0",   24),
])

a57_cpu_active_states = OrderedDict([
    (450000,  ActiveState(capacity=417,   power=168)),
    (625000,  ActiveState(capacity=579,   power=251)),
    (800000,  ActiveState(capacity=744,   power=359)),
    (950000,  ActiveState(capacity=883,   power=479)),
    (1100000, ActiveState(capacity=1023,  power=616)),
])

a57_cpu_idle_states = OrderedDict([
    ("WFI",               15),
    ("cpu-sleep-0",       0),
    ("cluster-sleep-0",   0),
])

a57s = [1, 2]
a57_pd = PowerDomain(cpus=a57s, idle_states=["cluster-sleep-0"], parent=None)

def a57_cpu_node(cpu):
    cpu_pd = PowerDomain(cpus=[cpu],
                         parent=a57_pd,
                         idle_states=["WFI", "cpu-sleep-0"])

    return EnergyModelNode([cpu],
                           active_states=a57_cpu_active_states,
                           idle_states=a57_cpu_idle_states,
                           power_domain=cpu_pd,
                           freq_domain=a57s)

juno_energy_levels = {
    'cluster': [
        EnergyModelNode(cpus=[0, 3, 4, 5],
                        active_states=a53_cluster_active_states,
                        idle_states=a53_cluster_idle_states),
        EnergyModelNode(cpus=[1, 2],
                        active_states=a57_cluster_active_states,
                        idle_states=a57_cluster_idle_states)
    ],
    'cpu': [
        a53_cpu_node(0),
        a57_cpu_node(1),
        a57_cpu_node(2),
        a53_cpu_node(3),
        a53_cpu_node(4),
        a53_cpu_node(5)
    ]
}

juno_energy = EnergyModel(levels=juno_energy_levels)
