import csv
import sys
from collections import defaultdict

sys.setrecursionlimit(1000000)

edgeFile = 'edges.csv'
Graph = defaultdict(list)  # Use a list to store the graph information
visited = set()  # To record the nodes we have visited
road = []  # Store the nodes we have gone through
visited_node = 0  # Total nodes we visited
path = 0.0  # Total distance we have gone through


def dfs_recur(now, end):
    global Graph, visited, road, visited_node, path
    visited.add(now)  # Update the node we have visited
    visited_node += 1  # Update the number of nodes we have gone through
    if now == end:  # If the reach our destination
        return True
    else:
        for neighbor, dist in Graph[now]:  # The nodes which connect with it
            if neighbor not in visited:  # If the node we haven't visited before
                if dfs_recur(neighbor, end):  # Recursive call the function
                    road.append(neighbor)
                    path += dist
                    return True
        return False


def dfs(start, end):
    # Begin your code (Part 2)
    # raise NotImplementedError("To be implemented")

    global Graph
    # Read all the data from edges.csv and construct the grpah
    with open(edgeFile, newline='') as f:
        edge = csv.reader(f)
        headers = next(edge)
        for line in edge:
            s, e, dis, speed = line
            s = int(s)
            e = int(e)
            dis = float(dis)
            Graph[s].append([e, dis])

    dfs_recur(start, end)  # Start doing DFS
    road.append(start)

    return road, path, visited_node
    # End your code (Part 2)


if __name__ == '__main__':
    path, dist, num_visited = dfs(2270143902, 1079387396)
    print(f'The number of path nodes: {len(path)}')
    print(f'Total distance of path: {dist}')
    print(f'The number of visited nodes: {num_visited}')
