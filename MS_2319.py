import random
from math import sqrt, exp
import matplotlib.pyplot as plt
from geopy.geocoders import Nominatim # type: ignore

# Calculate the Euclidean distance between two points (latitude, longitude)
def calculate_distance(point1, point2):
    return sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

# Compute the total distance of the current route
def compute_total_distance(cities, sequence):
    total_distance = sum(calculate_distance(cities[sequence[i]], cities[sequence[i+1]]) for i in range(len(sequence) - 1))
    total_distance += calculate_distance(cities[sequence[-1]], cities[sequence[0]])  # Closing the loop
    return total_distance

# Read city coordinates from a file using geopy to get latitude and longitude
def read_city_coordinates(filename):
    cities = []
    city_names = []
    geolocator = Nominatim(user_agent="TSPApp")
    
    with open(filename) as file:
        for line in file:
            city_name = line.strip()
            if city_name:  # Ensure line isn't empty
                location = geolocator.geocode(f"{city_name}, India", timeout=10)
                if location:  # Ensure location was found
                    city_names.append(city_name)
                    cities.append((round(location.latitude, 2), round(location.longitude, 2)))
                    print(f"{city_name}: ({location.latitude:.2f}, {location.longitude:.2f})")
    
    return cities, city_names

# Plot the current route on a graph
def plot_route(sequence, cities, total_distance, city_names):
    plt.figure()
    route = [cities[i] for i in sequence] + [cities[sequence[0]]]  # Complete the loop
    plt.plot([p[0] for p in route], [p[1] for p in route], marker='o')
    for i, city in enumerate(cities):
        plt.text(city[0], city[1], city_names[i])  # Label cities with their names
    plt.title(f"Total Distance: {total_distance:.2f}")
    plt.show()

# Try swapping two cities and decide whether to accept the new route
def swap_cities(cities, sequence, current_distance, city1, city2, temperature):
    new_sequence = sequence[:]
    new_sequence[city1], new_sequence[city2] = new_sequence[city2], new_sequence[city1]
    new_distance = compute_total_distance(cities, new_sequence)
    
    # Accept the new sequence if it is better, or with some probability if it is worse
    if new_distance < current_distance or random.random() < exp((current_distance - new_distance) / temperature):
        return new_sequence, new_distance
    return sequence, current_distance

# Main function to solve TSP using Simulated Annealing
def solve_tsp(cities, max_iterations, cooling_rate, initial_temp):
    n_cities = len(cities)
    sequence = list(range(n_cities))
    random.shuffle(sequence)  # Start with a random sequence
    current_distance = compute_total_distance(cities, sequence)
    temperature = initial_temp
    
    for i in range(max_iterations):
        city1, city2 = random.sample(range(n_cities), 2)
        sequence, current_distance = swap_cities(cities, sequence, current_distance, city1, city2, temperature)
        
        if (i + 1) % (max_iterations // 10) == 0:  # Plot route at intervals
            plot_route(sequence, cities, current_distance, city_names)
        
        temperature *= cooling_rate  # Decrease the temperature
    
    return sequence, current_distance

if __name__ == "__main__":
    # Read city data from file
    cities, city_names = read_city_coordinates("india_cities.txt")
    
    # Parameters for Simulated Annealing
    max_iterations = 1000
    cooling_rate = 0.95
    initial_temp = 1000.0
    
    # Solve TSP and plot the final route
    final_sequence, final_distance = solve_tsp(cities, max_iterations, cooling_rate, initial_temp)
    print(f"Optimal route: {final_sequence}")
    print(f"Optimal distance: {final_distance:.2f}")
    plot_route(final_sequence, cities, final_distance, city_names)
