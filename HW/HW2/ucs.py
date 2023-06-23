import csv
from collections import defaultdict
edgeFile = 'edges.csv'

def ucs(start, end):
    # Begin your code (Part 3)
    # raise NotImplementedError("To be implemented")
    Graph = defaultdict(list) # Use a list to store the graph information 
    path = 0.0 # Total distance we have gone through
    road = [] # Store the nodes we have gone through
    queue = [] # A "priority queue" for doning UCS 
    visited = set() # To record the nodes we have visited
    From = {} # Store the node where we come from
    visited_node = 0 # Total nodes we visited

    # Read all the data from edges.csv and construct the grpah
    with open(edgeFile, newline='') as f:
        data = csv.reader(f)
        temp = next(data)
        for line in data:
            s, e, dis, speed = line
            Graph[int(s)].append([int(e), float(dis)])

    """
    A pirority queue, first element is the node, the second is its weight, 
    the third is its parent, the fourth is distance
    """
    queue.append([start, 0.0, start, 0]) 
    From[start] = [start, 0.0] # First element is where the node come from, and the second is the distance
    visited_node += 1 # Update the number of node we have visited

    while len(queue) > 0:
        weight = float('inf')
        cur = 0
        previous = 0
        d = 0.0
        
        """
        Get the smallest weight in the priority queue because we always want to 
        get the lowest cost in UCS
        """
        for tmp, w, p, dist in queue: 
            if w < weight:
                weight = w
                cur = tmp
                previous = p
                d = dist
        queue.remove([cur, weight, previous, d]) # Remove the lowest cost in the priority queue

        if cur in visited: # If we have visited the node, then redo the loop to get another node
            continue
        visited_node += 1 # Update the number of node we have visited
        From[cur] = [previous, d] # Record the distance and how we arrival the node
        visited.add(cur) # Update since we have visited it

        if cur == end: # If we reach the destination
            break
        for neighbor, dist in Graph[cur]: # Put the unvisited nodes of cur's neighbors to the priority queue  
            if neighbor not in visited:
                queue.append([neighbor, dist+weight, cur, dist])

    # Traverse all the nodes we have visited (start from the destination)
    now = end
    while now != start:
        Previous, dis = From[now] # Get the node where we came from
        road.append(int(now)) # Put the node in the list
        now = Previous # Update the node
        path += dis # Update the distance we have gone through

    road.append(start) 

    return road, path, visited_node
    # End your code (Part 3)


if __name__ == '__main__':
    path, dist, num_visited = ucs(2270143902, 1079387396)
    print(f'The number of path nodes: {len(path)}')
    print(f'Total distance of path: {dist}')
    print(f'The number of visited nodes: {num_visited}')
