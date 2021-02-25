#!/usr/bin/env python

'''
=================================================
@Title  ：Leach for WSN
@Author ：Kay
@Date   ：2019-09-27
==================================================
'''
# https://www.geeksforgeeks.org/matplotlib-markers-module-in-python/
# https://stackoverflow.com/questions/22408237/named-colors-in-matplotlib
import random

import numpy as np
import matplotlib.pyplot as plt


class WSN(object):
    """ The network architecture with desired parameters """
    xm = 1000  # Length of the yard
    ym = 1000  # Width of the yard
    n = 100  # total number of nodes

    sink = None  # Sink node, type is 'Node'
    nodes_list = None  # All sensor nodes set

    # Energy model (all values in Joules)
    # Eelec = ETX = ERX
    ETX = 50 * (10 ** (-9))  # Energy for transferring of each bit:Transmission unit message loss energy:50nJ/bit
    ERX = 50 * (10 ** (-9))  # Energy for receiving of each bit:Receive unit message loss energy:50nJ/bit

    # Transmit Amplifier types
    Efs = 10 * (10 ** (-12))  # Energy of free space model:Free space propagation model:10pJ/bit/m2

    # Energy of multi path model:Multipath attenuation spatial energy model:0.0013pJ/bit/m4
    Emp = 0.0013 * (10 ** (-12))
    EDA = 5 * (10 ** (-9))  # Data aggregation energy:Aggregate energy 5nJ/bit
    f_r = 0.6  # fusion_rate:Fusion rate, 0 means perfect fusion

    # Message
    CM = 3200  # Control message size/bit
    DM = 40960  # Data message size/bit

    # computation of do
    do = np.sqrt(Efs / Emp)  # 87.70580193070293

    # Malicious sensor node
    m_n = 0  # the number of malicious sensor nodes

    # Node State in Network
    n_dead = 0  # The number of dead nodes
    flag_first_dead = 0  # Flag tells that the first node died
    flag_all_dead = 0  # Flag tells that all nodes died
    flag_net_stop = 0  # Flag tells that network stop working:90% nodes died
    round_first_dead = 0  # The round when the first node died
    round_all_dead = 0  # The round when all nodes died
    round_net_stop = 0  # The round when the network stop working

    def dist(x, y):
        """ Determine the one-dimensional distance between two nodes """
        distance = np.sqrt(np.power((x.xm - y.xm), 2) + np.power((x.ym - y.ym), 2))
        return distance

    def trans_energy(data, dis):
        if dis > WSN.do:
            energy = WSN.ETX * data + WSN.Emp * data * (dis ** 4)
        else:  # min_dis <= do
            energy = WSN.ETX * data + WSN.Efs * data * (dis ** 2)
        return energy

    def node_state(r: int):
        nodes = WSN.nodes_list
        n_dead = 0
        for node in nodes:
            if node.energy <= Node.energy_threshold:
                n_dead += 1
                if WSN.flag_first_dead == 0 and n_dead == 1:
                    WSN.flag_first_dead = 1
                    WSN.round_first_dead = r - Leach.r_empty
        if WSN.flag_net_stop == 0 and n_dead >= (WSN.n * 0.9):
            WSN.flag_net_stop = 1
            WSN.round_net_stop = r - Leach.r_empty
        if n_dead == WSN.n - 1:
            WSN.flag_all_dead = 1
            WSN.round_all_dead = r - Leach.r_empty
        WSN.n_dead = n_dead


class Leach(object):
    """ Leach """

    p = 0.1  # Optimal selection Probability of a node to become cluster head
    period = int(1 / p)  # cycle

    cluster_head_list = None  # Cluster head node list
    members = None  # List of non-cluster head members
    cluster = None  # Cluster dictionary: {"cluster head 1": [cluster member], "cluster head 2": [cluster member],...}
    current_round_no = 0  # Current round
    rmax = 2000  # 9999 # default maximum round
    r_empty = 0  # Empty wheel

    def show_cluster(self):
        fig = plt.figure()
        # Set title
        # Set X axis label
        plt.xlabel('X/m')

        # Set Y axis label
        plt.ylabel('Y/m')
        icon = ['o', '*', '.', 'x', '+', 's']
        color = ['r', 'b', 'g', 'c', 'y', 'm']

        # Show each cluster classification list
        i = 0
        nodes = WSN.nodes_list
        for key, value in Leach.cluster.items():
            cluster_head = nodes[int(key)]
            # print("First", i + 1, "The cluster center is:", cluster_head)
            for index in value:
                plt.plot([cluster_head.xm, nodes[index].xm], [cluster_head.ym, nodes[index].ym],
                         c=color[i % 6], marker=icon[i % 5], alpha=0.4)
                # If it is a malicious node
                if index >= WSN.n:
                    plt.plot([nodes[index].xm], [nodes[index].ym], 'dk')
            i += 1
        # Show the drawn picture
        plt.show()

    def optimum_number_of_clusters(self):
        """ Optimal number of cluster heads under perfect fusion """

        N = WSN.n - WSN.n_dead
        M = np.sqrt(WSN.xm * WSN.ym)
        d_toBS = np.sqrt((WSN.sink.xm - WSN.xm) ** 2 +
                         (WSN.sink.ym - WSN.ym) ** 2)
        k_opt = (np.sqrt(N) / np.sqrt(2 * np.pi) *
                 np.sqrt(WSN.Efs / WSN.Emp) *
                 M / (d_toBS ** 2))
        p = int(k_opt) / N
        return p

    def cluster_head_selection(self):
        """ Select the cluster head node according to the threshold """

        nodes = WSN.nodes_list
        n = WSN.n  # Non-malicious node
        heads = Leach.cluster_head_list = []  # List of cluster heads, initialized to empty every round
        members = Leach.members = []  # List of non-cluster members
        p = Leach.p
        r = Leach.current_round_no
        period = Leach.period
        Tn = p / (1 - p * (r % period))  # Threshold Tn
        print(Leach.current_round_no, Tn)
        for i in range(n):
            # After the energy dissipated in a given node reached a set threshold, 
            # that node was considered dead for the remainder of the simulation.
            if nodes[i].energy > Node.energy_threshold:  # Node is not dead
                if nodes[i].G == 0:  # The node is not selected as a cluster head in this period
                    temp_rand = np.random.random()
                    # print(temp_rand)

                    # Nodes with random numbers below the threshold are selected as cluster heads
                    if temp_rand <= Tn:
                        nodes[i].type = "CH"  # This node is the cluster head of the current round of the cycle
                        nodes[i].G = 1  # G is set to 1, this cycle can no longer be selected as the CH or (1/p)-1

                        heads.append(nodes[i])
                        # The node is selected as the cluster head and broadcast this message
                        # Announce cluster-head status, wait for join-request messages
                        max_dis = np.sqrt(WSN.xm ** 2 + WSN.ym ** 2)
                        nodes[i].energy -= WSN.trans_energy(WSN.CM, max_dis)
                        # node may die
                if nodes[i].type == "N":  # The node is not a cluster head node
                    members.append(nodes[i])
        m_n = WSN.m_n
        for i in range(m_n):
            j = n + i
            members.append(nodes[j])
        # If the cluster head is not found in this round
        if not heads:
            Leach.r_empty += 1
            print("---> No cluster head found in this round！")
            # Leach.cluster_head_selection()
        print("The number of CHs is:", len(heads), (WSN.n - WSN.n_dead))
        return None  # heads, members

    def cluster_formation(self):
        """ Cluster classification """
        nodes = WSN.nodes_list
        heads = Leach.cluster_head_list
        members = Leach.members
        cluster = Leach.cluster = {}  # Cluster dictionary initialization
        # There is no cluster head in this round, and no cluster is formed
        if not heads:
            return None
        # If the cluster head exists, use the cluster head id as the key value of the cluster dictionary
        for head in heads:
            cluster[str(head.uid)] = []  # Empty list of members
        # print("Classification dictionary with cluster head only:", cluster)
        # Traverse non-cluster head nodes and build clusters

        for member in members:
            # Pick the node with the smallest distance
            min_dis = np.sqrt(WSN.xm ** 2 + WSN.ym ** 2)  # Broadcast radius in the cluster head node area
            head_id = None
            # Receive all cluster head information
            # wait for cluster-head announcements
            member.energy -= WSN.ERX * WSN.CM * len(heads)
            # Judge the distance to each cluster head and add the cluster head with the smallest distance
            for head in heads:
                tmp = WSN.dist(member, head)
                if tmp <= min_dis:
                    min_dis = tmp
                    head_id = head.uid
            member.head_id = head_id  # Cluster head found
            # Send joining information to notify its cluster head to become its member
            # send join-request messages to chosen cluster-head
            member.energy -= WSN.trans_energy(WSN.CM, min_dis)
            # Cluster head receives join message
            # wait for join-request messages
            head = nodes[head_id]
            head.energy -= WSN.ERX * WSN.CM
            cluster[str(head_id)].append(member.uid)  # Add to the corresponding cluster head of the clustering class
        # Assign each node in the cluster the time point to pass data to it
        # Create a TDMA schedule and this schedule is broadcast back to the nodes in the cluster.
        for key, values in cluster.items():
            head = nodes[int(key)]
            if not values:
                # If there are cluster members, the CH sends schedule by broadcasting
                max_dis = np.sqrt(WSN.xm ** 2 + WSN.ym ** 2)
                head.energy -= WSN.trans_energy(WSN.CM, max_dis)
                for x in values:
                    member = nodes[int(x)]
                    # wait for schedule from cluster-head
                    member.energy -= WSN.ERX * WSN.CM
        #        print(cluster)
        return None  # cluster

    def set_up_phase(self):
        Leach.cluster_head_selection(self)
        Leach.cluster_formation(self)

    def steady_state_phase(self):
        """ The cluster members send data to the cluster head, the cluster head gathers the data
        and then sends the data to the sink node """
        nodes = WSN.nodes_list
        cluster = Leach.cluster
        # If no clusters are formed this round, exit
        if not cluster:
            return None
        for key, values in cluster.items():
            head = nodes[int(key)]
            n_member = len(values)  # Number of cluster members
            # The members of the cluster send data to the cluster head node
            for x in values:
                member = nodes[int(x)]
                dis = WSN.dist(member, head)
                member.energy -= WSN.trans_energy(WSN.DM, dis)  # Cluster members send data
                head.energy -= WSN.ERX * WSN.DM  # Cluster head receives data
            d_h2s = WSN.dist(head, WSN.sink)  # The distance of from head to sink
            if n_member == 0:  # If there is no cluster member, only the cluster head collects its own information and sends it to the base station
                energy = WSN.trans_energy(WSN.DM, d_h2s)
            else:
                new_data = WSN.DM * (
                        n_member + 1)  # In addition to the data collected by the cluster head itself, a new data packet after fusion
                E_DA = WSN.EDA * new_data  # Energy consumption of aggregated data
                if WSN.f_r == 0:  # f_r is 0 represents perfect integration of data
                    new_data_ = WSN.DM
                else:
                    new_data_ = new_data * WSN.f_r
                E_Trans = WSN.trans_energy(new_data_, d_h2s)
                energy = E_DA + E_Trans
            head.energy -= energy

    def leach(self):
        Leach.set_up_phase(self)
        Leach.steady_state_phase(self)

    def run_leach(self):
        for r in range(Leach.rmax):
            Leach.current_round_no = r
            nodes_list = WSN.nodes_list

            # When a new cycle starts, G is reset to 0 for all nodes
            if (r % Leach.period) == 0:
                print("================== NEW CYCLE ==================")
                for node in nodes_list:
                    node.G = 0

            # At the beginning of each round, the node type is reset to non-cluster head node
            for node in nodes_list:
                node.type = "N"

            Leach.leach(self)
            WSN.node_state(r)
            if WSN.flag_all_dead:
                print("================== ALL DEAD ==================")
                break
            Leach.show_cluster(self)


class Node(object):
    """ Sensor Node """
    initial_energy = 0.5  # initial energy of a node

    # After the energy dissipated in a given node reaches a set threshold,
    # that node will be considered dead for the remainder of the simulation.
    energy_threshold = 0.001

    def __init__(self):
        """ Create the node with default attributes """
        self.uid = random.randint(0, 1000000000)  # node ID for identification
        self.xm = np.random.random() * WSN.xm  # Random X coordinate
        self.ym = np.random.random() * WSN.ym  # Random Y coordinate

        self.energy = 0.5  # initial energy of a node
        self.type = "N"  # "N" = Normal Node (Non-CH)

        # G is the set of nodes that have not been cluster-heads in the last 1/p rounds.
        # the flag determines whether it's a CH or not.
        # In each cycle, this flag is 0 means it is not selected as the cluster head,
        # 1 means it is selected as the cluster head
        self.G = 0

        self.head_id = None  # The id of its CH：Subordinate cluster, None Means that no cluster has been added

    def init_nodes(self):
        """ Initialize attributes of every node in order """
        nodes_list = []

        # Initial common node
        for i in range(WSN.n):
            node = Node()
            node.uid = i
            nodes_list.append(node)
        # Initial sink node
        sink = Node()
        sink.uid = -1
        sink.xm = 0.5 * WSN.xm  # x coordination of base station
        sink.ym = 0.5 * WSN.ym  # y coordination of base station

        # Add to WSN
        WSN.nodes_list = nodes_list
        WSN.sink = sink

    def init_malicious_nodes(self):
        """ Initialize attributes of every malicious node in order """

        for i in range(WSN.m_n):
            node = Node()
            node.uid = WSN.n + i
            WSN.nodes_list.append(node)

    def plot_wsn(self):
        nodes = WSN.nodes_list
        n = WSN.n
        m_n = WSN.m_n

        # base station
        sink = WSN.sink
        plt.plot([sink.xm], [sink.ym], 'r^', label="base station")

        # Normal node
        n_flag = True
        for i in range(n):
            if n_flag:
                plt.plot([nodes[i].xm], [nodes[i].ym], 'b+', label='Normal node')
                n_flag = False
            else:
                plt.plot([nodes[i].xm], [nodes[i].ym], 'b+')

        # Malicious node
        m_flag = True
        for i in range(m_n):
            j = n + i
            if m_flag:
                plt.plot([nodes[j].xm], [nodes[j].ym], 'kd', label='Malicious node')
                m_flag = False
            else:
                plt.plot([nodes[j].xm], [nodes[j].ym], 'kd')
        plt.legend()
        plt.xlabel('X/m')
        plt.ylabel('Y/m')
        # plt.savefig('0.png', bbox_inches='tight')
        plt.show()


def main():
    Node().init_nodes()
    Node().init_malicious_nodes()
    Node().plot_wsn()  # this will plot all the nodes and sink only. This is before simulation starts

    Leach().run_leach()

    print("The first node died in Round %d!" % (WSN.round_first_dead))
    print("The network stop working in Round %d!" % (WSN.round_net_stop))
    print("All nodes died in Round %d!" % (WSN.round_all_dead))


if __name__ == '__main__':
    main()


# import cv2, pytesseract
#
#
# def start():
#     x = cv2.imread('/home/hritwik/Pictures/Screenshot from 2021-02-14 21-40-03_NNNEW.png', 0)
#     ret, thresh1 = cv2.threshold(x, 127, 255, cv2.THRESH_TOZERO)
#     text = (pytesseract.image_to_string(thresh1))
#     my = ((text).split(" "))
#     print(','.join(my))
#     cv2.imshow('Set to 0 Inverted', thresh1)
#
#     # De-allocate any associated memory usage
#     if cv2.waitKey(0) & 0xff == 27:
#         cv2.destroyAllWindows()
#
#
# start()
