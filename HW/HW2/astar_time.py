import csv
from collections import defaultdict
edgeFile = 'edges.csv'
heuristicFile = 'heuristic.csv'


def astar_time(start, end):
    # Begin your code (Part 6)
    # raise NotImplementedError("To be implemented")
    """
      This part is very similiar to part 4 and the three differences are the heuristic
      value, g(n) value and we want to get the maximum speed value. 
      At this part, the heuristics value is straight line distance between 
      strat node and end node / maximum speed, g(n) is the fastest time we need
      to go to this node which is equal to distance/speed limit 
    """
    Graph = defaultdict(list)  # Use a list to store the graph information
    visited_node = 0  # Total nodes we visited
    path = 0.0  # Total distance we have gone through
    road = []  # Store the nodes we have gone through
    visited = set()  # To record the nodes we have visited
    From = {}  # Store the node where we come from
    queue = []  # A priority queue for doning A* time search
    heuristic = {}  # Heuristic function
    Max_Speed = 0  # Record the maximum speed

    # Read all the data from edges.csv and construct the grpah
    with open(edgeFile, newline='') as f:
        data = csv.reader(f)
        temp = next(data)
        for line in data:
            s, e, dis, speed = line
            Max_Speed = max(Max_Speed, float(speed)/3.6)
            time = float(dis)/(float(speed)/3.6)
            Graph[int(s)].append([int(e), time])

    # Read all the data from heuristic.csv and use heuristic to record heuristic function
    with open(heuristicFile, newline='') as f:
        lines = csv.reader(f)
        temp = next(lines)
        for line in lines:
            for i in range(1, 4):
                if int(temp[i]) == end:
                    heuristic[int(line[0])] = float(line[i])/Max_Speed

    """
    A pirority queue, first element is the node, the second is its 
    heuristic value + time, the third is its parent, the fourth is time
    """
    queue.append([start, heuristic[start], start, 0])
    From[start] = [start, 0.0]
    while len(queue) > 0:
        weight = float('inf')
        cur = 0
        previous = 0
        time = 0.0

        """
        Get the smallest value of heuristic value + time in the priority 
        queue because we always want to get the lowest cost in A* search
        """
        for tmp, w, p, t in queue:
            if w < weight:
                weight = w
                cur = tmp
                previous = p
                time = t
        queue.remove([cur, weight, previous, time])  # Remove the chosen one

        if cur in visited:
            continue

        From[cur] = [previous, time]
        weight -= heuristic[cur]  # Because weight includes the heuristic value
        visited.add(cur)
        visited_node += 1

        if cur == end:  # If we have visited the node, then redo the loop to get another node
            break

        # Put the unvisited nodes of cur's neighbors to the priority queue
        for neighbor, t in Graph[cur]:
            queue.append(
                [neighbor, t+weight+heuristic[neighbor], cur, t])

    # Traverse all the nodes we have visited (start from the destination)
    now = end
    while now != start:
        Previous, time = From[now]  # Get the node where we came from
        road.append(int(now))  # Put the node in the list
        now = Previous  # Update the node
        path += time  # Update the distance we have gone through
    road.append(start)

    return road, path, visited_node

    # End your code (Part 6)


if __name__ == '__main__':
    path, time, num_visited = astar_time(2270143902, 1079387396)
    print(f'The number of path nodes: {len(path)}')
    print(f'Total second of path: {time}')
    print(f'The number of visited nodes: {num_visited}')
