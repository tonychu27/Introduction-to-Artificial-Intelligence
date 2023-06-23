import csv
from collections import defaultdict
edgeFile = 'edges.csv'
heuristicFile = 'heuristic.csv'

def astar(start, end):
    # Begin your code (Part 4)
    # raise NotImplementedError("To be implemented")

    Graph = defaultdict(list) # Use a list to store the graph information 
    visited_node = 0 # Total nodes we visited
    path = 0.0 # Total distance we have gone through
    road = [] # Store the nodes we have gone through
    visited = set() # To record the nodes we have visited
    From = {} # Store the node where we come from
    queue = [] # A priority queue for doning A* search
    heuristic = {} # Heuristic function 

    # Read all the data from edges.csv and construct the grpah
    with open(edgeFile, newline='') as f:
        data = csv.reader(f)
        temp = next(data)
        for line in data:
            s, e, dis, speed = line
            Graph[int(s)].append([int(e), float(dis)])

    # Read all the data from heuristic.csv and use heuristic to record heuristic function
    with open(heuristicFile, newline='') as f:
        lines = csv.reader(f)
        temp = next(lines)
        for line in lines:
            for i in range(1, 4):
                if int(temp[i]) == end:
                    heuristic[int(line[0])] = float(line[i])
    """
    A pirority queue, first element is the node, the second is its 
    heuristic value + distance, the third is its parent, the fourth is distance
    """
    queue.append([start, heuristic[start], start, 0]) 
    From[start] = [start, 0.0] # First element is where the node come from, and the second is the distance

    # Do A* search
    while len(queue) > 0:
        weight = float('inf')
        cur = 0
        previous = 0
        dist = 0.0

        """
        Get the smallest value of heuristic value + distance in the priority 
        queue because we always want to get the lowest cost in A* search
        """
        for tmp, w, p, dis in queue: 
            if w < weight:  
                weight = w
                cur = tmp
                previous = p
                dist = dis
        queue.remove([cur, weight, previous, dist]) # Remove the chosen one

        if cur in visited: # If we have visited the node, then redo the loop to get another node
            continue

        visited.add(cur)
        visited_node += 1
        From[cur] = [previous, dist]
        weight -= heuristic[cur] # Because weight includes the heuristic value

        if cur == end: # If we reach the destination
            break

        # Put the unvisited nodes of cur's neighbors to the priority queue 
        for neighbor, dis in Graph[cur]:
            if neighbor not in visited:
                queue.append([neighbor, dis+weight+heuristic[neighbor], cur, dis])

    # Traverse all the nodes we have visited (start from the destination)
    now = end
    while now != start:
        Previous, dis = From[now] # Get the node where we came from
        road.append(int(now)) # Put the node in the list
        now = Previous # Update the node
        path += dis # Update the distance we have gone through

    road.append(start)    

    return road, path, visited_node
    # End your code (Part 4)


if __name__ == '__main__':
    path, dist, num_visited = astar(2270143902, 1079387396)
    print(f'The number of path nodes: {len(path)}')
    print(f'Total distance of path: {dist}')
    print(f'The number of visited nodes: {num_visited}')
