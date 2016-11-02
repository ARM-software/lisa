from energy_model import ActiveState, EnergyModelNode, PowerDomain, EnergyModel

from collections import OrderedDict

cluster_active_states = OrderedDict([
    (208000, ActiveState(capacity=178, power=16)),
    (432000, ActiveState(capacity=369, power=29)),
    (729000, ActiveState(capacity=622, power=47)),
    (960000, ActiveState(capacity=819, power=75)),
    (1200000, ActiveState(capacity=1024, power=112))
])

cluster_idle_states = OrderedDict([
    ('WFI', 107),
    ('cpu-sleep', 47),
    ('cluster-sleep', 0)
])

cpu_active_states = OrderedDict([
    (208000,  ActiveState(capacity=178,  power=69)),
    (432000,  ActiveState(capacity=369,  power=125)),
    (729000,  ActiveState(capacity=622,  power=224)),
    (960000,  ActiveState(capacity=819,  power=367)),
    (1200000, ActiveState(capacity=1024, power=670))
])

cpu_idle_states = OrderedDict([('WFI', 15), ('cpu-sleep', 0), ('cluster-sleep', 0)])

cpus = range(8)

cluster_pds = [
    PowerDomain(cpus=[0, 1, 2, 3], idle_states=["cluster-sleep"], parent=None),
    PowerDomain(cpus=[4, 5, 6, 7], idle_states=["cluster-sleep"], parent=None),
]

def cpu_node(cpu):
    cpu_pd=PowerDomain(cpus=[cpu],
                       parent=cluster_pds[cpu / 4],
                       idle_states=["WFI", "cpu-sleep"])

    return EnergyModelNode([cpu],
                           active_states=cpu_active_states,
                           idle_states=cpu_idle_states,
                           power_domain=cpu_pd,
                           freq_domain=cpus)
hikey_energy_levels = {
    'cluster': [
        EnergyModelNode(cpus=[0, 1, 2, 3],
                        active_states=cluster_active_states,
                        idle_states=cluster_idle_states),
        EnergyModelNode(cpus=[4, 5, 6, 7],
                        active_states=cluster_active_states,
                        idle_states=cluster_idle_states)
    ],
    'cpu': [cpu_node(c) for c in cpus]
}

hikey_energy = EnergyModel(levels=hikey_energy_levels)
