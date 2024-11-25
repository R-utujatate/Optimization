import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
from geopy.geocoders import Nominatim
import time

# Load city names from the file
with open('India_cities.txt', 'r') as file:
    cities = [line.strip() for line in file.readlines()]

# Function to fetch latitude and longitude using the geopy library
def get_lat_lon(city_name):
    geolocator = Nominatim(user_agent="tsp_solver")
    try:
        location = geolocator.geocode(city_name + ", India")
        return location.latitude, location.longitude
    except:
        print(f"Could not find location for {city_name}")
        return None, None

# Fetch the coordinates for all cities
coordinates = []
for city in cities:
    lat, lon = get_lat_lon(city)
    if lat and lon:
        coordinates.append([lat, lon])
    else:
        coordinates.append([0, 0])  # Fallback if location not found
    time.sleep(1)  # Sleep to avoid overloading the API

coordinates = np.array(coordinates)

# Number of cities
n_cities = len(cities)

# Function to calculate the total distance of the tour
def calculate_distance(tour):
    distance = 0
    for i in range(len(tour)):
        start_city = coordinates[tour[i]]
        end_city = coordinates[tour[(i + 1) % len(tour)]]
        distance += np.linalg.norm(start_city - end_city)
    return distance

# Fitness function: smaller distance = better fitness
def fitness(tour):
    return 1 / calculate_distance(tour)

# Generate initial population of random tours
def create_population(pop_size):
    population = []
    for _ in range(pop_size):
        tour = random.sample(range(n_cities), n_cities)
        population.append(tour)
    return population

# Selection using tournament selection
def tournament_selection(population, fitnesses):
    tournament_size = 5
    selected = random.sample(list(zip(population, fitnesses)), tournament_size)
    selected = sorted(selected, key=lambda x: x[1], reverse=True)
    return selected[0][0]

# Crossover: ordered crossover
def crossover(parent1, parent2):
    size = len(parent1)
    child = [-1] * size
    start, end = sorted(random.sample(range(size), 2))
    
    child[start:end + 1] = parent1[start:end + 1]
    available_cities = [city for city in parent2 if city not in child]
    
    child = [city if city != -1 else available_cities.pop(0) for city in child]
    return child

# Mutation: swap mutation
def mutate(tour, mutation_rate):
    for i in range(len(tour)):
        if random.random() < mutation_rate:
            j = random.randint(0, len(tour) - 1)
            tour[i], tour[j] = tour[j], tour[i]
    return tour

# Function to evolve the population
def evolve_population(population, mutation_rate):
    fitnesses = [fitness(tour) for tour in population]
    new_population = []
    
    for _ in range(len(population)):
        parent1 = tournament_selection(population, fitnesses)
        parent2 = tournament_selection(population, fitnesses)
        
        child = crossover(parent1, parent2)
        child = mutate(child, mutation_rate)
        new_population.append(child)
    
    return new_population, max(fitnesses), np.mean(fitnesses)

# Plot the cities on a map with lines connecting them (path)
def plot_tour(tour, generation, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    tour_coords = np.array([coordinates[i] for i in tour + [tour[0]]])  # Close the loop
    ax.plot(tour_coords[:, 1], tour_coords[:, 0], 'bo-', label="Path", alpha=0.6)
    
    for i, city in enumerate(tour):
        ax.text(coordinates[city][1], coordinates[city][0], cities[city], fontsize=10, color='red')
    
    ax.set_title(f"Generation: {generation}, Distance: {calculate_distance(tour):.2f} km")
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()

# Genetic Algorithm
def genetic_algorithm(pop_size=100, generations=500, mutation_rate=0.01):
    population = create_population(pop_size)
    best_distances = []
    avg_fitnesses = []
    
    for generation in range(generations):
        population, best_fitness, avg_fitness = evolve_population(population, mutation_rate)
        
        best_tour = population[np.argmax([fitness(tour) for tour in population])]
        best_distances.append(1 / best_fitness)
        avg_fitnesses.append(avg_fitness)
        
        if generation % 50 == 0 or generation == generations - 1:
            print(f"Generation {generation}: Best Distance = {1/best_fitness:.2f}")
            plot_tour(best_tour, generation)
    
    # Plot the fitness over generations
    plt.figure(figsize=(10, 5))
    plt.plot(range(generations), best_distances, label="Best Distance")
    plt.plot(range(generations), avg_fitnesses, label="Average Fitness")
    plt.xlabel('Generation')
    plt.ylabel('Distance / Fitness')
    plt.title('Fitness Progression')
    plt.legend()
    plt.show()

# Run the genetic algorithm
genetic_algorithm(pop_size=200, generations=500, mutation_rate=0.02)
