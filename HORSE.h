#pragma once

#include <cmath>
#include <random>
#include <iostream>
#include <limits>
#include <tuple>
#include <fstream>
#include <sstream>
#include <string>
#include <ctime>
#include <chrono>
#include <unordered_set>
#include <unordered_map>
#include <iomanip>
#include <map>
#include <mpi.h>
#include <boost/mpi.hpp>
#include <boost/serialization/string.hpp>
#include <boost/serialization/vector.hpp>

struct Horizon {
    std::string section_name;
    double horizon_score;
    int section_number;
    int horizon_number;
    int horizon_height;
    std::vector<int> presence_absence_data;
};

struct Chromosome {
    std::vector<Horizon> horizons;
    double fitness;
    double local_operator_improvement;
    double global_operator_improvement;
    int local_operator_count;
    int global_operator_count;
};

namespace boost {
    namespace serialization {

        template<class Archive>
        void serialize(Archive& ar, Horizon& h, const unsigned int version) {
            ar& h.section_name;
            ar& h.horizon_score;
            ar& h.section_number;
            ar& h.horizon_number;
            ar& h.horizon_height;
            ar& h.presence_absence_data;
        }
        template<class Archive>
        void serialize(Archive& ar, Chromosome& c, const unsigned int version) {
            ar& c.horizons;
            ar& c.fitness;
            ar& c.local_operator_improvement;
            ar& c.global_operator_improvement;
            ar& c.local_operator_count;
            ar& c.global_operator_count;
        }
    } // namespace serialization
} // namespace boost

std::vector<Horizon> read_csv_data(const std::string& file_path, std::string& header_line);
void save_to_csv(const std::vector<Horizon>& horizons, const std::string& file_path, const std::string& header_line);
std::vector<Horizon> init_horizons(std::vector<Horizon>& horizons);
Chromosome init_chromosome(const std::vector<Horizon>& d3);
int calculate_penalty(const std::vector<Horizon>& horizons, int n_biostrat);
double calculate_fitness(const std::vector<Horizon>& horizons, int n_biostrat);
std::vector<Chromosome> selection(const std::vector<Chromosome>& population, int tournament_size);
std::tuple<std::vector<Horizon>, std::vector<Horizon>> crossover(const std::vector<Horizon>& parent1, const std::vector<Horizon>& parent2);
std::tuple<std::vector<Horizon>, std::vector<Horizon>> crossover_pmx(const std::vector<Horizon>& parent1, const std::vector<Horizon>& parent2);
std::vector<Horizon> mutation(const std::vector<Horizon>& individual, int n_biostrat);
void update_fitness_improvement(Chromosome& chromosome, double old_fitness);
void adaptive_local_mutation(Chromosome& chromosome, int mutation_range, int l, int history_rounds);
void apply_adaptive_local_mutation(std::vector<Chromosome>& population, int mutation_range, int l, int history_rounds);
void repair_chromosome(std::vector<Horizon>& chromosome);
std::string getCurrentTimeString();
std::string generateFilename(const std::string& prefix, const std::string& extension);
void update_population_with_best_solution(std::vector<Chromosome>& population, const Chromosome& received_best_solution);
void update_best_solution_library(std::vector<Chromosome>& best_solution_library, const Chromosome& received_solution, size_t LIBRARY_SIZE);