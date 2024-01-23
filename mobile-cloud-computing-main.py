from copy import deepcopy
import sys
import time

class Node(object):
    # create data structure: NODE
    def __init__(self, id, core_speed, cloud_speed):
        self.task_id = id
        self.number_cores = len(core_speed)
        self.parents = None
        self.children = None
        self.core_speeds = core_speed
        self.cloud_speed = cloud_speed
        self.finish_time_local = 0
        self.finish_time_wireless_send = 0
        self.finish_time_cloud = 0
        self.finish_time_wireless_recieve = 0
        self.ready_time_local = -1
        self.ready_time_wireless_send = -1
        self.ready_time_cloud = -1
        self.ready_time_wireless_recieve = -1
        self.is_core = None
        self._primary_assignment_()
        self._task_prioritization_()
        self.priority_score = None
        self.assignment = -2
        self.start_time = [-1] * self.number_cores
        self.start_time.append(-1) # append for cloud
        self.is_scheduled = None

    def _primary_assignment_(self):
        t_l_min = self.core_speeds[2]
        t_c_min = 5 # assume cloud is always 5
        if t_l_min <= t_c_min:
            self.is_core = True
            self.finish_time_wireless_send = 0
            self.finish_time_cloud = 0
            self.finish_time_wireless_recieve = 0
        else:
            self.is_core = False
            self.finish_time_local = 0

    def _task_prioritization_(self): # section 3
        self.w_i = 0
        if self.is_core == True:
            self.w_i = sum(self.core_speeds) / len(self.core_speeds)
        else:
            self.w_i = 5
            
    def __repr__(self):
        return "Node id: {}".format(self.task_id)
    
    def __str__(self):
        return "Node id: {}".format(self.task_id)


def time_total(nodes):
    '''
    calculate the total time
    '''
    total_t = 0
    for node in nodes:
        if len(node.children) == 0:
            total_t = max(node.finish_time_local, node.finish_time_wireless_recieve)
    return total_t


def energy_total(nodes):
    '''
    calculate the total energy cost
    '''
    core_cloud_power = []
    for i in range(nodes[0].number_cores):
        core_cloud_power.append(2**i)
    core_cloud_power.append(0.5)
    total_energy = 0
    for node in nodes:
        if node.is_core:
            current_node_e = node.core_speeds[node.assignment] * core_cloud_power[node.assignment]
            total_energy += current_node_e
        else:
            current_node_e = node.cloud_speed[0] * core_cloud_power[-1]
            total_energy += current_node_e
    return total_energy


def execution_unit_selection(nodes):
    '''
    compute the primary assignmnets
    '''
    num_cores = nodes[0].number_cores
    local_source = [0] * num_cores
    cloud_source = [0, 0, 0]
    core_seqs = [[] for _ in range(num_cores)]
    cloud_seq = []
    # core and cloud sequence after assignment

    for i, node in enumerate(nodes):
        if node.is_core:
            if len(node.parents) == 0:
                node.ready_time_local = 0
            else:
                for p in node.parents:
                    p_ft = max(p.finish_time_local, p.finish_time_wireless_recieve)
                    if p_ft > node.ready_time_local:
                        node.ready_time_local = p_ft

            core_finish_times = []
            for k, speed in enumerate(node.core_speeds):
                core_finish_times.append(max(local_source[k], node.ready_time_local) + speed)
            # choose the fastest core
            core_assign_id = 0
            fastest = float("inf")
            for k, finish_time in enumerate(core_finish_times):
                if fastest > finish_time:
                    fastest = finish_time
                    core_assign_id = k
            node.assignment = core_assign_id
            node.finish_time_local = fastest
            node.start_time[core_assign_id] = max(local_source[core_assign_id], node.ready_time_local)

            local_source[core_assign_id] = node.finish_time_local

            core_seqs[core_assign_id].append(node.task_id)
            if node.is_core:
                print(f"node id:{node.task_id}, primary assigenment:{node.assignment + 1}, ready time: {node.ready_time_local}, "
                      f"local start_time: {node.start_time[node.assignment]}, duration:{node.core_speeds[node.assignment]}")
                print()

        else: # cloud task
            for p in node.parents:
                p_ws = max(p.finish_time_local, p.finish_time_wireless_send)
                if p_ws > node.ready_time_wireless_send:
                    node.ready_time_wireless_send = p_ws
            cloud_ws_finishtime = max(cloud_source[0], node.ready_time_wireless_send) + node.cloud_speed[0]
            node.finish_time_wireless_send = cloud_ws_finishtime
            # (5)
            p_max_ft_c = 0
            for p in node.parents:
                if p.finish_time_cloud > p_max_ft_c:
                    p_max_ft_c = p.finish_time_cloud
            node.ready_time_cloud = max(node.finish_time_wireless_send, p_max_ft_c)
            cloud_c_finishtime = max(cloud_source[1], node.ready_time_cloud) + node.cloud_speed[1]
            node.finish_time_cloud = cloud_c_finishtime
            #(6)
            node.ready_time_wireless_recieve = node.finish_time_cloud
            cloud_wr_finishtime = max(cloud_source[2], node.ready_time_wireless_recieve) + node.cloud_speed[2]
            node.finish_time_wireless_recieve = cloud_wr_finishtime
            node.assignment = 3 # 3 is cloud
            node.start_time[3] = max(cloud_source[0], node.ready_time_wireless_send) 
            # cloud task start time is sending start time

            cloud_source[0] = cloud_ws_finishtime
            cloud_source[1] = cloud_c_finishtime
            cloud_source[2] = cloud_wr_finishtime

            cloud_seq.append(node.task_id)
            print(f"Node ID:{node.task_id}, Assignment:{node.assignment + 1}, Wireless Send Ready Time: {node.ready_time_wireless_send},"
                  f" Cloud Ready Time: {node.ready_time_cloud}, Wireless Recieve Ready Time: {node.ready_time_wireless_recieve}, "
                  f"Cloud Start Time: {node.start_time[3]}")
            print(local_source)
            print()
    seq = []
    for s in core_seqs:
        seq.append(s)
    seq.append(cloud_seq)
    return seq


def compute_final_schedule(nodes, tar_id, k, seq):
    """
    compute new scheduling seq
    """
    node_index = {}
    temp_id = 0
    for _node in nodes:
        node_index[_node.task_id] = temp_id
        temp_id += 1
        if _node.task_id == tar_id:
            node_tar = _node
    if node_tar.is_core == True:
        # calculate tar ready time in (19)
        node_tar_rt = node_tar.ready_time_local
    if node_tar.is_core == False:
        node_tar_rt = node_tar.ready_time_wireless_send
    seq[node_tar.assignment].remove(node_tar.task_id)
    # original core seq
    s_new = seq[k] #  (19)
    s_new_prim = []
    flag = False
    for _node_id in s_new:
        _node = nodes[node_index[_node_id]]
        if _node.start_time[k] < node_tar_rt:
            s_new_prim.append(_node.task_id)
        if _node.start_time[k] >= node_tar_rt and flag == False:
            s_new_prim.append(node_tar.task_id)
            flag = True
        if _node.start_time[k] >= node_tar_rt and flag == True:
            s_new_prim.append(_node.task_id)
    if flag == False:
        s_new_prim.append(node_tar.task_id)
    seq[k] = s_new_prim
    node_tar.assignment = k
    if k == 3:
        node_tar.is_core = False
    else:
        node_tar.is_core = True

    return seq


def kernel_algorithm(nodes_new, seq_new):
    """
    kernel algorithm
    """
    local_source = [0] * len(nodes_new[0].core_speeds)
    cloud_source = [0, 0, 0]

    ready1 = [-1]*len(nodes_new)
    ready2 = [-1]*len(nodes_new)
    ready1[nodes_new[0].task_id - 1] = 0
    for each_seq in seq_new:
        if len(each_seq) > 0:
            ready2[each_seq[0] - 1] = 0

    node_index = {}
    temp_id = 0
    for _node in nodes_new:
        node_index[_node.task_id] = temp_id
        _node.ready_time_local = -1
        _node.ready_time_wireless_send = -1
        _node.ready_time_cloud = -1
        _node.ready_time_wireless_recieve = -1
        temp_id += 1

    # start the rescheduling task
    stack = []
    stack.append(nodes_new[0])

    while len(stack) != 0:
        v_i = stack.pop()
        v_i.is_scheduled = "kernel_scheduled"

        if v_i.is_core == True:
            if len(v_i.parents) == 0:
                v_i.ready_time_local = 0
            else: # equation (3)
                for p in v_i.parents:
                    p_ft = max(p.finish_time_local, p.finish_time_wireless_recieve)
                    if p_ft > v_i.ready_time_local:
                        v_i.ready_time_local = p_ft

        # schedule on corresponding core
        if v_i.assignment == 0:
            v_i.start_time = [-1, -1, -1, -1]
            v_i.start_time[0] = max(local_source[0], v_i.ready_time_local)
            v_i.finish_time_local = v_i.start_time[0] + v_i.core_speeds[0]
            v_i.finish_time_wireless_send = -1
            v_i.finish_time_cloud = -1
            v_i.finish_time_wireless_recieve = -1
            local_source[0] = v_i.finish_time_local
        if v_i.assignment == 1:
            v_i.start_time = [-1, -1, -1, -1]
            v_i.start_time[1] = max(local_source[1], v_i.ready_time_local)
            v_i.finish_time_local = v_i.start_time[1] + v_i.core_speeds[1]
            v_i.finish_time_wireless_send = -1
            v_i.finish_time_cloud = -1
            v_i.finish_time_wireless_recieve = -1
            local_source[1] = v_i.finish_time_local
        if v_i.assignment == 2:
            v_i.start_time = [-1, -1, -1, -1]
            v_i.start_time[2] = max(local_source[2], v_i.ready_time_local)
            v_i.finish_time_local = v_i.start_time[2] + v_i.core_speeds[2]
            v_i.finish_time_wireless_send = -1
            v_i.finish_time_cloud = -1
            v_i.finish_time_wireless_recieve = -1
            local_source[2] = v_i.finish_time_local

        if v_i.assignment == 3:
            if len(v_i.parents) == 0:
                v_i.ready_time_wireless_send = 0
            else:
                for p in v_i.parents:
                    p_ws = max(p.finish_time_local, p.finish_time_wireless_send)
                    if p_ws > v_i.ready_time_wireless_send:
                        v_i.ready_time_wireless_send = p_ws
            v_i.finish_time_wireless_send = max(cloud_source[0], v_i.ready_time_wireless_send) + v_i.cloud_speed[0]
            v_i.start_time = [-1, -1, -1, -1]
            v_i.start_time[3] = max(cloud_source[0], v_i.ready_time_wireless_send)
            cloud_source[0] = v_i.finish_time_wireless_send

            p_max_ft_c = 0
            for p in v_i.parents:
                if p.finish_time_cloud > p_max_ft_c:
                    p_max_ft_c = p.finish_time_cloud
            v_i.ready_time_cloud = max(v_i.finish_time_wireless_send, p_max_ft_c)
            v_i.finish_time_cloud = max(cloud_source[1], v_i.ready_time_cloud) + v_i.cloud_speed[1]
            cloud_source[1] = v_i.finish_time_cloud

            v_i.ready_time_wireless_recieve = v_i.finish_time_cloud
            v_i.finish_time_wireless_recieve = max(cloud_source[2], v_i.ready_time_wireless_recieve) + v_i.cloud_speed[2]
            v_i.finish_time_local = -1
            cloud_source[2] = v_i.finish_time_wireless_recieve
        corresponding_seq = seq_new[v_i.assignment]

        v_i_index = corresponding_seq.index(v_i.task_id)
        if v_i_index != len(corresponding_seq) - 1:
            next_node_id = corresponding_seq[v_i_index + 1]
        else:
            next_node_id = -1

        for _node in nodes_new:
            flag = 0
            for p in _node.parents:
                if p.is_scheduled != "kernel_scheduled":
                    flag += 1
                ready1[_node.task_id - 1] = flag
            if _node.task_id == next_node_id:
                ready2[_node.task_id - 1] = 0

        for _node in nodes_new:
            if (ready1[_node.task_id - 1] == 0) and (ready2[_node.task_id - 1] == 0) and (_node.is_scheduled != "kernel_scheduled") and (_node not in stack):
                stack.append(_node)

    for node in nodes_new:
        node.is_scheduled = None
    return nodes_new


def get_cloud_and_core_speed():
    cs = [3, 1, 1]
    core_speed_list = [[],[9, 7, 5], [8, 6, 5], [6, 5, 4], [7,5,3],[5,4,2],[7,6,4],[8,5,3], [6,4,2], [5,3,2], [7,4,2] ]
    return cs, core_speed_list


def test1():
    cs, core_speed_list = get_cloud_and_core_speed()
    node_list = [0]
    for i in range(1,11):
        node_list.append(Node(id=i, core_speed=core_speed_list[i], cloud_speed=cs))
    
    node_list[1].parents = []
    node_list[2].parents = [node_list[1]]
    node_list[3].parents = [node_list[1]]
    node_list[4].parents = [node_list[1]]
    node_list[5].parents = [node_list[1]]
    node_list[6].parents = [node_list[1]]
    node_list[7].parents = [node_list[3]]
    node_list[8].parents = [node_list[2], node_list[4], node_list[6]]
    node_list[9].parents = [node_list[2], node_list[4], node_list[5]]
    node_list[10].parents = [node_list[7], node_list[8], node_list[9]]

    node_list[1].children = [node_list[2], node_list[3], node_list[4], node_list[5], node_list[6]]
    node_list[2].children = [node_list[8], node_list[9]]
    node_list[3].children = [node_list[7]]
    node_list[4].children = [node_list[8], node_list[9]]
    node_list[5].children = [node_list[9]]
    node_list[6].children = [node_list[8]]
    node_list[7].children = [node_list[10]]
    node_list[8].children = [node_list[10]]
    node_list[9].children = [node_list[10]]
    node_list[10].children = []

    node_list[1].ready_time_local = 0
    del node_list[0] 
    node_list.reverse()
    return node_list


def test2():
    cs, core_speed_list = get_cloud_and_core_speed()
    node_list = [0]
    for i in range(1,11):
        node_list.append(Node(id=i, core_speed=core_speed_list[i], cloud_speed=cs))
    
    node_list[1].parents = []
    node_list[2].parents = [node_list[1]]
    node_list[3].parents = [node_list[1]]
    node_list[4].parents = [node_list[1]]
    node_list[5].parents = [node_list[2], node_list[3]]
    node_list[6].parents = [node_list[3], node_list[4]]
    node_list[7].parents = [node_list[5]]
    node_list[8].parents = [node_list[5], node_list[6]] 
    node_list[9].parents = [node_list[6]]
    node_list[10].parents = [node_list[7], node_list[8], node_list[9]]

    node_list[1].children = [node_list[2], node_list[3], node_list[4]]
    node_list[2].children = [node_list[5]]
    node_list[3].children = [node_list[5], node_list[6]]
    node_list[4].children = [node_list[6]]
    node_list[5].children = [node_list[7], node_list[8]]
    node_list[6].children = [node_list[8], node_list[9]]
    node_list[7].children = [node_list[10]]
    node_list[8].children = [node_list[10]]
    node_list[9].children = [node_list[10]]
    node_list[10].children = []

    node_list[1].ready_time_local = 0
    del node_list[0] 
    node_list.reverse()
    return node_list


def test3():
    cs, core_speed_list = get_cloud_and_core_speed()
    core_speed_list.append([20, 19, 17])
    core_speed_list.append([8, 6, 5])
    core_speed_list.append([7, 5, 2])
    core_speed_list.append([6, 4, 2])
    core_speed_list.append([5, 3, 2])
    core_speed_list.append([4, 2, 1])
    core_speed_list.append([3, 2, 1])
    core_speed_list.append([9, 2, 1])
    core_speed_list.append([4, 3, 1])
    core_speed_list.append([3, 2, 1])
    node_list = [0]
    for i in range(1,21):
        node_list.append(Node(id=i, core_speed=core_speed_list[i], cloud_speed=cs))
    
    node_list[1].parents = []
    node_list[2].parents = [node_list[1]]
    node_list[3].parents = [node_list[1]]
    node_list[4].parents = [node_list[1]]
    node_list[5].parents = [node_list[2], node_list[3]]
    node_list[6].parents = [node_list[3], node_list[4]]
    node_list[7].parents = [node_list[5]]
    node_list[8].parents = [node_list[5], node_list[6]] 
    node_list[9].parents = [node_list[6]]
    node_list[10].parents = [node_list[7], node_list[8], node_list[9]]
    node_list[11].parents = [node_list[10]]
    node_list[12].parents = [node_list[10]]
    node_list[13].parents = [node_list[10]]
    node_list[14].parents = [node_list[10]]
    node_list[15].parents = [node_list[11], node_list[12]]
    node_list[16].parents = [node_list[12]]
    node_list[17].parents = [node_list[12], node_list[13]]
    node_list[18].parents = [node_list[14]]
    node_list[19].parents = [node_list[15], node_list[16], node_list[17], node_list[18]]
    node_list[20].parents = [node_list[19]]

    node_list[1].children = [node_list[2], node_list[3], node_list[4]]
    node_list[2].children = [node_list[5]]
    node_list[3].children = [node_list[5], node_list[6]]
    node_list[4].children = [node_list[6]]
    node_list[5].children = [node_list[7], node_list[8]]
    node_list[6].children = [node_list[8], node_list[9]]
    node_list[7].children = [node_list[10]]
    node_list[8].children = [node_list[10]]
    node_list[9].children = [node_list[10]]
    node_list[10].children = [node_list[11], node_list[12], node_list[13], node_list[14]]
    node_list[11].children = [node_list[15]]
    node_list[12].children = [node_list[15], node_list[16], node_list[17]]
    node_list[13].children = [node_list[17]]
    node_list[14].children = [node_list[18]]
    node_list[15].children = [node_list[19]]
    node_list[16].children = [node_list[19]]
    node_list[17].children = [node_list[19]]
    node_list[18].children = [node_list[19]]
    node_list[19].children = [node_list[20]]
    node_list[20].children = []

    node_list[1].ready_time_local = 0
    del node_list[0] # fix index to start from 0 again
    node_list.reverse()
    return node_list


def test5():
    cs, core_speed_list = get_cloud_and_core_speed()
    core_speed_list.append([20, 19, 17])
    core_speed_list.append([8, 6, 5])
    core_speed_list.append([7, 5, 2])
    core_speed_list.append([6, 4, 2])
    core_speed_list.append([5, 3, 2])
    core_speed_list.append([4, 2, 1])
    core_speed_list.append([3, 2, 1])
    core_speed_list.append([9, 2, 1])
    core_speed_list.append([4, 3, 1])
    core_speed_list.append([3, 2, 1])
    node_list = [0]
    for i in range(1,21):
        node_list.append(Node(id=i, core_speed=core_speed_list[i], cloud_speed=cs))
    
    node_list[1].parents = []
    node_list[2].parents = []
    node_list[3].parents = []
    node_list[4].parents = []
    node_list[5].parents = [node_list[1], node_list[2]]
    node_list[6].parents = [node_list[3], node_list[4]]
    node_list[7].parents = [node_list[5]]
    node_list[8].parents = [node_list[5], node_list[6]] 
    node_list[9].parents = [node_list[6]]
    node_list[10].parents = [node_list[7], node_list[8], node_list[9]]
    node_list[11].parents = [node_list[10]]
    node_list[12].parents = [node_list[10]]
    node_list[13].parents = [node_list[10]]
    node_list[14].parents = [node_list[10]]
    node_list[15].parents = [node_list[11], node_list[12]]
    node_list[16].parents = [node_list[12]]
    node_list[17].parents = [node_list[12], node_list[13]]
    node_list[18].parents = [node_list[14]]
    node_list[19].parents = [node_list[15], node_list[16]]
    node_list[20].parents = [node_list[17], node_list[18]]

    node_list[1].children = [node_list[5]]
    node_list[2].children = [node_list[5]]
    node_list[3].children = [node_list[6]]
    node_list[4].children = [node_list[6]]
    node_list[5].children = [node_list[7], node_list[8]]
    node_list[6].children = [node_list[8], node_list[9]]
    node_list[7].children = [node_list[10]]
    node_list[8].children = [node_list[10]]
    node_list[9].children = [node_list[10]]
    node_list[10].children = [node_list[11], node_list[12], node_list[13], node_list[14]]
    node_list[11].children = [node_list[15]]
    node_list[12].children = [node_list[15], node_list[16], node_list[17]]
    node_list[13].children = [node_list[17]]
    node_list[14].children = [node_list[18]]
    node_list[15].children = [node_list[19]]
    node_list[16].children = [node_list[19]]
    node_list[17].children = [node_list[20]]
    node_list[18].children = [node_list[20]]
    node_list[19].children = []
    node_list[20].children = []

    node_list[1].ready_time_local = 0
    del node_list[0] # fix index to start from 0 again
    node_list.reverse()
    return node_list


def test4():
    cs, core_speed_list = get_cloud_and_core_speed()
    core_speed_list.append([20, 19, 17])
    core_speed_list.append([8, 6, 5])
    core_speed_list.append([7, 5, 2])
    core_speed_list.append([6, 4, 2])
    core_speed_list.append([5, 3, 2])
    core_speed_list.append([4, 2, 1])
    core_speed_list.append([3, 2, 1])
    core_speed_list.append([9, 2, 1])
    core_speed_list.append([4, 3, 1])
    core_speed_list.append([3, 2, 1])
    node_list = [0]
    for i in range(1,21):
        node_list.append(Node(id=i, core_speed=core_speed_list[i], cloud_speed=cs))
    
    node_list[1].parents = []
    node_list[2].parents = []
    node_list[3].parents = []
    node_list[4].parents = []
    node_list[5].parents = [node_list[1], node_list[2]]
    node_list[6].parents = [node_list[3], node_list[4]]
    node_list[7].parents = [node_list[5]]
    node_list[8].parents = [node_list[5], node_list[6]] 
    node_list[9].parents = [node_list[6]]
    node_list[10].parents = [node_list[7], node_list[8], node_list[9]]
    node_list[11].parents = [node_list[10]]
    node_list[12].parents = [node_list[10]]
    node_list[13].parents = [node_list[10]]
    node_list[14].parents = [node_list[10]]
    node_list[15].parents = [node_list[11], node_list[12]]
    node_list[16].parents = [node_list[12]]
    node_list[17].parents = [node_list[12], node_list[13]]
    node_list[18].parents = [node_list[14]]
    node_list[19].parents = [node_list[15], node_list[16], node_list[17], node_list[18]]
    node_list[20].parents = [node_list[19]]

    node_list[1].children = [node_list[5]]
    node_list[2].children = [node_list[5]]
    node_list[3].children = [node_list[6]]
    node_list[4].children = [node_list[6]]
    node_list[5].children = [node_list[7], node_list[8]]
    node_list[6].children = [node_list[8], node_list[9]]
    node_list[7].children = [node_list[10]]
    node_list[8].children = [node_list[10]]
    node_list[9].children = [node_list[10]]
    node_list[10].children = [node_list[11], node_list[12], node_list[13], node_list[14]]
    node_list[11].children = [node_list[15]]
    node_list[12].children = [node_list[15], node_list[16], node_list[17]]
    node_list[13].children = [node_list[17]]
    node_list[14].children = [node_list[18]]
    node_list[15].children = [node_list[19]]
    node_list[16].children = [node_list[19]]
    node_list[17].children = [node_list[19]]
    node_list[18].children = [node_list[19]]
    node_list[19].children = [node_list[20]]
    node_list[20].children = []

    node_list[1].ready_time_local = 0
    del node_list[0] # fix index to start from 0 again
    node_list.reverse()
    return node_list


def algorithm(test_num):
    if test_num == 1:
        node_list = test1()
    elif test_num == 2:
        node_list = test2()
    elif test_num == 3:
        node_list = test3()
    elif test_num == 4:
        node_list = test4()
    elif test_num == 5:
        node_list = test5()
    # START algorithm, timed
    start = time.time()

    for node in node_list:
        priority_score = node.w_i
        if len(node.children) == 0:
            node.priority_score = priority_score
            continue
        child_score = max([i.priority_score for i in node.children])

        node.priority_score = priority_score + child_score

    node_list = sorted(node_list, key=lambda node: node.priority_score, reverse=True)

    print("compute priority order")
    for node in node_list:
        print(node.task_id, node.priority_score)

    sequence = execution_unit_selection(node_list)
    time_init = time_total(node_list)
    energy_init = energy_total(node_list)
    print("initial time and energy: ", time_init, energy_init)

    # outer looop
    iter_num = 0
    while iter_num < 100:
        time_init = time_total(node_list)
        energy_init = energy_total(node_list)
        migration_choice = [[] for i in range(len(node_list))]
        for i in range(len(node_list)):
            if node_list[i].assignment == 3:
                current_row_id = node_list[i].task_id - 1
                current_row_value = [1] * 4
                migration_choice[current_row_id] = current_row_value
            else:
                current_row_id = node_list[i].task_id - 1
                current_row_value = [0] * 4
                current_row_value[node_list[i].assignment] = 1
                migration_choice[current_row_id] = current_row_value

        T_max_constraint = 27
        result_table = [[(-1, -1) for j in range(4)] for i in range(len(node_list))]

        for n in range(len(migration_choice)):
            nth_row = migration_choice[n]
            for k in range(len(nth_row)):
                if nth_row[k] == 1:
                    continue
                seq_copy = deepcopy(sequence)
                nodes_copy = deepcopy(node_list)
                seq_copy = compute_final_schedule(nodes_copy, n + 1, k, seq_copy)
                kernel_algorithm(nodes_copy, seq_copy)

                current_time = time_total(nodes_copy)
                current_energy = energy_total(nodes_copy)
                del nodes_copy
                del seq_copy
                result_table[n][k] = (current_time, current_energy)

        n_best = -1
        k_best = -1
        time_best = time_init
        energy_best = energy_init
        ration_best = -1
        for i in range(len(result_table)):
            for j in range(len(result_table[i])):
                val = result_table[i][j]
                if val == (-1, -1):
                    continue
                if val[0] > 27:
                    continue
                ration = (energy_best - val[1]) / abs(val[0] - time_best + 0.00005)
                if ration > ration_best:
                    ration_best = ration
                    n_best = i
                    k_best = j

        if n_best == -1 and k_best == -1:
            break
        n_best += 1
        k_best += 1
        time_best, energy_best = result_table[n_best-1][k_best-1]
        sequence = compute_final_schedule(node_list, n_best, k_best - 1, sequence)
        kernel_algorithm(node_list, sequence)
        time_current = time_total(node_list)
        energy_current = energy_total(node_list)
        energy_diff = energy_init - energy_current
        time_diff = abs(time_current - time_init)
        iter_num += 1

        if energy_diff <= 1:
            break

    print("Final assignment:")
    for node in node_list:
        assign = node.assignment + 1
        if node.is_core:
            print(f"ID={node.task_id}, Core={assign}, Ready Time={node.ready_time_local}, Local Start={node.start_time[node.assignment]}, Local End={node.finish_time_local}")
        else:
            print(
                f"{node.task_id=}, Cloud, Wireless Send Start={node.ready_time_wireless_send}, Send_End={node.cloud_speed[0] + node.ready_time_wireless_send},  Compute Start={node.ready_time_cloud}, Compute End={node.ready_time_cloud + node.cloud_speed[1]}, Wireless Recieve Time={node.ready_time_wireless_recieve},  Recieve End={node.cloud_speed[2] + node.ready_time_wireless_recieve}")
        print()
    elapsed = (time.time() - start)

    print("Final Sequence")
    for s in sequence:
        print([i for i in s])
    final_time = time_total(node_list)
    final_energy = energy_total(node_list)

    # write results to log file
    with open('log.out','a') as f:
        f.write(f"{final_time=}, {final_energy=}\n")
        f.close()
    print(f"{final_time=}, {final_energy=}")
    # END start algorithm, timed

if __name__ == '__main__':
    for i in range(1, 6):
        input(f'press enter to run test case {i}...')
        algorithm(i)
