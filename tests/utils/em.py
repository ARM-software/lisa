import itertools

# The utilization rate limit to switch to the overutilization mode
OVERUTILIZED_RATE = 0.8

def _get_powers_list(node):
    """
    Get the list of powers for a given node.
    :param node: A node of the energy model. It can be a cluster or cpu.
    :type node: EnergyModelNode

    :returns: The list of powers associates with the given node.
    """
    powers = []
    for freq in node.active_states.keys():
        powers.append(node.active_states[freq].power)
    return powers

def _get_capacities_list(node):
    """
    Get the list of capacities for a given node.
    :param node: A node of the energy model. It can be a cluster or cpu.
    :type node: EnergyModelNode

    :returns: The list of capacities associates with the given node.
    """
    cap = []
    for freq in node.active_states.keys():
        cap.append(node.active_states[freq].capacity)
    return cap

def _is_list_strickly_increasing(l):
    return all(x<y for x, y in zip(l, l[1:]))

def is_power_increasing(energy_model):
    """
    Check that for all the nodes of the energy model the power is an increasing
    sequence.
    :params energy_model: The energy model to control
    :type energy_model: EnergyModel

    :returns: A tuple that contains True if all nodes respect the
              previous assumptions, False otherwise; and an error message.
    """
    powers = []
    msg = ''
    failed_nodes = []
    succeed = True
    for cluster in energy_model.root.children:
        power = _get_powers_list(cluster)
        # Check that clusters powers are strickly increasing
        if not _is_list_strickly_increasing(power):
            failed_nodes.append(cluster.name)
            succeed = False

    for cpu in energy_model.cpu_nodes:
        powers = _get_powers_list(cpu)
        # Check that cpu powers are strickly increasing
        if not _is_list_strickly_increasing(powers):
            failed_nodes.append(cpu.name)
            succeed = False

    if not succeed:
        msg = 'The power lists is not increasing for {}'.format(failed_nodes)
    return (succeed, msg)

def is_efficiency_decreasing(energy_model):
    """
    Check that for all the cpus of the energy model the energy efficiency is
    decreasing.
    :params energy_model: The energy model to control
    :type energy_model: EnergyModel

    :returns: A tuple that contains True if the energy_model respects
              the previous assumptions, False otherwise; and an error message.
    """
    msg = ''
    powers = []
    cap = []
    energy = []
    failed_nodes = []
    succeed = True
    for cpu in energy_model.cpu_nodes:
        powers = _get_powers_list(cpu)
        cap = _get_capacities_list(cpu)
        energy = [float(powers[i])/cap[i] for i in range(len(powers))]
        # Check that power / cap is strickly increasing which is equivalent
        # to check that the cap / power (energy efficiency) is decreasing
        if not _is_list_strickly_increasing(energy):
            failed_nodes.append(cpu.name)
            succeed = False
    if not succeed:
        msg = ('The energy efficiency is not decreasing '
               'for {}'.format(failed_nodes))

    return (succeed, msg)

def check_active_states_nb(energy_model, freqs_nb):
    """
    Control that the active states number is equal to the number of
    frequencies in an energy model.
    :params energy_model: The energy model to control
    :type energy_model: EnergyModel

    :params freqs_nb: A list that contains the number of frequencies per cluster
    :type freqs_nb: [int]

    :returns: A tuple that contains True if the number of active states
              correspond to the number of frequencies for each cluster, False
              otherwise; and an error message.
    """
    msg = ''
    for i, cluster in enumerate(energy_model.root.children):
        for cpu in cluster.children:
            active_state_nb = len(cpu.active_states.keys())
            if active_state_nb != freqs_nb[i]:
                msg = ('The number of active states is not equal to the '
                       'frequency number for {}. It is {} and should '
                       'be {}'.format(cpu.name, active_state_nb, freqs_nb[i]))
                return (False, msg)

        active_state_nb = len(cluster.active_states.keys())
        if active_state_nb != freqs_nb[i]:
            msg = ('The number of active states is not equal to the '
                   'frequency number for {}. It is {} and should '
                   'be {}'.format(cluster.name, active_state_nb, freqs_nb[i]))
            return (False, msg)
    return (True, msg)

def get_opp_overutilized(energy_model):
    """
    Get the capacity limit after which the cpu enter in overutilized mode
    and indicate all the opp for the group that are over this limit.
    :params energy_model: The energy model to control
    :type energy_model: EnergyModel

    :returns: a list containing a tupple per group containing the limit
              capacity and an list of the opp that are in the overutilized zone
    """
    opp = []
    for group in energy_model.cpu_groups:
        # Get the first cpu of the group
        cpu = energy_model.cpu_nodes[group[0]]
        cap = _get_capacities_list(cpu)
        cap_limit = cap[-1] * OVERUTILIZED_RATE

        # Get the elements that are bigger than cap limit
        cap_overutilized = [x for x in cap if x > cap_limit]
        opp.append((cap_limit, cap_overutilized))
    return opp

def get_avg_cap(energy_model):
    """
    Get a task utilization which should fits on a groups of cpu.

    :params energy_model: The energy model to get the information
    :type energy_model: EnergyModel

    :returns: a list of average capacities per group of cpus.
    """
    avg_cap = []
    max_cap = 0
    min_cap = 0
    for group in energy_model.cpu_groups:
        cap = _get_capacities_list(energy_model.cpu_nodes[group[0]])

        # Choose the min capacity of the task or the max capacity of the
        # previous group to be sure that the task cannot fit on the previous
        # group
        min_cap = cap[0] if cap[0] > max_cap else max_cap
        max_cap = cap[-1]
        avg_cap.append((min_cap + max_cap * OVERUTILIZED_RATE) / 2)
    return avg_cap

def _are_placements_equal(energy_model, expected, placement):
    """
    Given an optimal placement and the placement simulated on the energy
    model controls if both placements are equal per group.
    :params energy_model: The energy model from which the placement is tested
    :type energy_model: EnergyModel

    :params expected: a list of group containing a list of util per cpus. For
                      instance a placement could be : [[0, 0, 0, 100], [0,0]]
                      for an energy model containing 4 little cpus and 2 big
                      cpus and an util of 100 is expected on one of the little
                      cpus.
    :type expected: [[int]]

    :params placement: an exhaustive list of placements obtaining by the
                       function get_optimal_placements from the EnergyModel.
    :type placement: [(int)]

    :returns: True if both placements are equal, False otherwise.
    """
    for i, group in enumerate(energy_model.cpu_groups):
        # Extract the placement for the given group
        s1 = set([tuple(l[g] for g in group) for l in placement])
        # Generate all the placements possible for this group of cpus
        s2 = set(itertools.permutations(expected[i]))
        if s1 != s2:
            return False
    return True

def _get_expected_placement(energy_model, group_util):
    """
    Create the expected placement given an utilization per group.
    :params energy_model: The energy model for which the expected placement is
                          computed
    :type energy_model: EnergyModel

    :params group_util: an utilization value for each group of cpus
    :type group_util: [int]

    :returns: a list for each group that contain a list of utilization per cpu.
              For instance for a model that has four little cpus and two big
              cpus it will return the following data:
              [[group_util[0], 0, 0, 0], [group_util[1], 0]]
    """
    expected = []
    for i, group in enumerate(energy_model.cpu_groups):
        cpus_nb = len(group)
        cpus_util = [0] * cpus_nb
        cpus_util[0] = group_util[i]
        expected.append(cpus_util)
    return expected

def ideal_placements(energy_model):
    """
    For each group, generates a workload that should run only on it and
    controls that obtained placement is equal to the expected placement
    :params energy_model: The energy model for which these workloads are
                          generated
    :type energy_model: EnergyModel

    :returns: True if the expected placement is equal to the obtained one,
              False otherwise
    """
    msg = ''
    avg_cap = get_avg_cap(energy_model)
    for i, group in enumerate(energy_model.cpu_groups):
        one_task = {'t0': avg_cap[i]}
        placement = energy_model.get_optimal_placements(one_task)
        exp_cap = [0] * len(energy_model.cpu_groups)
        exp_cap[i] = avg_cap[i]
        expected = _get_expected_placement(energy_model, exp_cap)
        if not _are_placements_equal(energy_model, expected, placement):
            msg = ('The expected placement for the task {} is {} but the '
                    'obtained placement \n\t\t'
                    'was {}'.format(one_task, expected, placement))
            return (False, msg)
    return (True, msg)

def _get_efficiency(util, cpu):
    """
    For a given utilization, computes the energy efficiency (capacity / power)
    to run at this utilization. If it does not correspond to an opp for the
    cpu, the efficiency is computed by selecting the opp above the
    utilization and the power is computed proportionally to it.
    :params util: targeted utilization
    :type util: int

    :params cpu: The cpu for which the energy efficiency is computed.
    :type cpu: EnergyModelNode

    :returns: The ratio between the capacity and the power for the targeted opp
    """

    # Get the list of capacities and powers for the cpu
    capacities = _get_capacities_list(cpu)
    powers = _get_powers_list(cpu)

    # Compute the efficiency for the first capacity larger than the
    # requiered opp
    for cap in capacities:
        if cap >= util:
            power = powers[capacities.index(cap)] * (float(util) / cap)
            return float(cap) / power

    raise ValueError('The utilization {} is larger than the possible '
                     'opp for {}'.format(util, cpu.name))

def check_overutilized_area(energy_model):
    """
    For the overutilized zone of the little cpu controls that it is indeed more
    efficient to run a workload on the big cpu.
    :params energy_model: the energy model that contains the information
                          for the cpus
    :type energy_model: EnergyModel

    :returns: A tupple containing True if the conditions are respected, False
              otherwise; and an error message.
    """
    msg = ''
    groups = energy_model.cpu_groups
    if len(groups) < 2:
        return (True, msg)

    # Get the first little and big cpus
    little_cpu = energy_model.cpu_nodes[groups[0][0]]
    big_cpu = energy_model.cpu_nodes[groups[1][0]]

    # Get the opp in overutilized mode for the little cpu
    (limit_opp, overutilized_opp) = get_opp_overutilized(energy_model)[0]

    for opp in [limit_opp]+overutilized_opp:
        little_efficiency = _get_efficiency(opp, little_cpu)
        big_efficiency = _get_efficiency(opp, big_cpu)
        if little_efficiency > big_efficiency:
            msg = ('It is more energy efficient to run for the utilization {} '
                   'on the little cpu \n\t\tbut it is run on the big cpu due '
                   'to the overutilization zone'.format(opp))
            return (False, msg)
    return (True, msg)