import csv
from collections import defaultdict
edgeFile = 'edges.csv'

def bfs(start, end):
    # Begin your code (Part 1)
    # raise NotImplementedError("To be implemented")
    
    Graph = defaultdict(list) # Use a list to store the graph information 

    visited_node = 0 # Total nodes we visited
    path = 0.0 # Total distance we have gone through
    road = [] # Store the nodes we have gone through
    visited = set() # To record the nodes we have visited
    From = {} # Store the node where we come from
    queue = [] # A queue for doning BFS

    queue.append(start) # Put the starting point in the queue
    From[start] = start 
    visited.add(start)
    find = False

    # Read all the data from edges.csv and construct the grpah
    with open(edgeFile, newline='') as f:
        data = csv.reader(f)
        temp = next(data)
        for line in data:
            s, e, dis, speed = line
            Graph[int(s)].append([int(e), float(dis)])

    # Do BFS (Using queue)
    while len(queue) > 0:
        current = queue.pop(0) # Get the first element in the queue and remove it
        for neighbor, dist in Graph[current]: # The nodes which connect with it
            if neighbor not in visited: # If the node we haven't visited before
                visited_node += 1 # Update the total nodes we visited
                From[neighbor] = [current, dist] # Record where it is from and the distance
                queue.append(neighbor) # Put it in the queue
                visited.add(neighbor) # Update since we have visited it

            if neighbor == end: # If we found our destination
                find = True
                break
        if find:
            break

    # Traverse all the nodes we have visited (start from the destination)
    now = end 
    while now != start:
        Previous, dis = From[now] # Get the node where we came from
        road.append(int(now))  # Put the node in the list
        now = Previous # Update the node
        path += dis # Update the distance we have gone through
    road.append(start)

    return road, path, visited_node
    # End your code (Part 1)


if __name__ == '__main__':
    path, dist, num_visited = bfs(2270143902, 1079387396)
    print(f'The number of path nodes: {len(path)}')
    print(f'Total distance of path: {dist}')
    print(f'The number of visited nodes: {num_visited}')
