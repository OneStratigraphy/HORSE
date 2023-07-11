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

struct PenaltyParameters {
    int n_biostrat;
    std::vector<int> biostrat_columns;
    int n_pmag;
    int pmag;
    int n_dates;
    std::vector<std::vector<int>> dates;
    int n_ashes;
    std::vector<std::vector<int>> ashes;
    int n_continuous;
    std::vector<std::vector<int>> continuous;
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
PenaltyParameters initialize_penalty_parameters();
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

std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<> dis(0.0, 1.0);

int main(int argc, char* argv[]) {
    // MPI ��ʼ��
    MPI_Init(&argc, &argv);
    auto start_time = std::chrono::high_resolution_clock::now();
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int max_rank = size - 1;

    boost::mpi::environment env(argc, argv);
    boost::mpi::communicator world;

    // ��csv�ļ���ȡ����
    const std::string file_path = "./dataset/HA_Fan_et_al_2013.csv";
    std::string header_line;
    std::vector<Horizon> horizons = read_csv_data(file_path, header_line);

    srand(time(NULL) * rank); // �������������
    const int N_HORZIONS = horizons.size(); // ����ˮƽ������
    const int N_BIOSTRAT = horizons[0].presence_absence_data.size(); // ������������
    const int num_generations = 1000; // outer_num(����������)
    const int population_size = 1000; // inner_num
    const int exchange_interval = 5; // ���ý������
    const int library_size = 10; // �������Ž���С
    double crossover_rate = 0.99;
    double mutation_rate = 0.99;
    int mutation_range = 5;
    int l = 10;
    int history_rounds = 10;

    // ������Ϣ��ʶ
    const int MSG_REQUEST_BEST_SOLUTION = 13;
    const int MSG_SEND_BEST_SOLUTION = 14;

    std::vector<Chromosome> best_solutions(library_size); // �������Ž��
    if (rank == 0) { // �����̣�����ά�����Ž�⣩
        MPI_Status status;

        // ��ʼ�����
        for (int i = 0; i < library_size; ++i) {
            best_solutions[i] = init_chromosome(horizons);
        }

        int count = 0;
        while (count < 2 * num_generations) {
            count++;
            int msg_type;
            MPI_Recv(&msg_type, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            int sender_rank = status.MPI_SOURCE; // ��ȡ��Ӧ�ӽ������

            if (msg_type == MSG_REQUEST_BEST_SOLUTION) { // ���ͽ�
                // ���ӽ��̷���ȷ�Ͻ��ձ�ʶ
                MPI_Send(&msg_type, 1, MPI_INT, sender_rank, 0, MPI_COMM_WORLD);

                // �ӽ���������ѡһ�����Žⷢ�͸��ӽ���
                int target_rank = rand() % library_size;
                Chromosome best_solution = best_solutions[target_rank];
                world.send(sender_rank, 0, best_solution);
            }
            else if (msg_type == MSG_SEND_BEST_SOLUTION) { // ���ս�
                // ���ӽ��̷���ȷ�Ͻ��ձ�ʶ
                MPI_Send(&msg_type, 1, MPI_INT, sender_rank, 0, MPI_COMM_WORLD);

                // �����ӽ��̷��͵Ľ��Ž�
                Chromosome received_chromosome;
                world.recv(sender_rank, 0, received_chromosome);
                int generation = 0;
                world.recv(sender_rank, 0, generation);

                // ʹ�ý��յ��ĵ�ǰ���Ž����ȫ�����Ž��
                update_best_solution_library(best_solutions, received_chromosome, library_size);
                // �����������Ž�
                auto best_solution = std::min_element(best_solutions.begin(), best_solutions.end(), [](const Chromosome& a, const Chromosome& b) { return a.fitness < b.fitness; });
                std::cout << "Generation:" << generation << " | Rank:" << sender_rank << " | Current fitness:" << received_chromosome.fitness << " | Best fitness:" << best_solution->fitness << std::endl;
            }
        }
    }
    else { // �ӽ��̣�������������
        // ��ʼ����Ⱥ
        std::vector<Horizon> d3 = horizons;
        std::vector<Chromosome> population;
        for (int i = 0; i < population_size; ++i) {
            population.push_back(init_chromosome(d3));
        }
        // �������и�����Ӧ�Ȳ����������Ӧ��
        double best_fitness = std::numeric_limits<double>::infinity();
        for (Chromosome& chromosome : population) {
            chromosome.fitness = calculate_fitness(chromosome.horizons, N_BIOSTRAT);
            if (chromosome.fitness < best_fitness) {
                best_fitness = chromosome.fitness;
            }
        }
        std::cout << "rank:" << rank << " | Starting penalty: " << best_fitness << "" << std::endl;

        for (int generation = 0; generation < num_generations; ++generation) {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<> dis(0.0, 1.0);
            // ִ���Ŵ��㷨��һ�ε���
            // ѡ�񣨽��������ԣ�����population�������ѡ��mating_pool��size����
            std::vector<Chromosome> mating_pool = selection(population, int(2 * population_size / 10)); // 20%
            std::vector<Chromosome> new_population;

            //// ����Ӧ�ֲ�����
            //apply_adaptive_local_mutation(mating_pool, mutation_range, l, history_rounds);

            // ������ȺΪ��������ֽ��в���
            for (int j = 0; j < population_size; j += 2) {
                Chromosome parent1 = mating_pool[j];
                Chromosome parent2 = mating_pool[j + 1];
                // ���棺��ѡ
                if (dis(gen) < crossover_rate) {
                    std::tie(parent1.horizons, parent2.horizons) = crossover_pmx(parent1.horizons, parent2.horizons);
                }
                // ͻ��
                if (dis(gen) < mutation_rate) {
                    parent1.horizons = mutation(parent1.horizons, N_BIOSTRAT);
                }
                if (dis(gen) < mutation_rate) {
                    parent2.horizons = mutation(parent2.horizons, N_BIOSTRAT);
                }
                new_population.push_back(parent1);
                new_population.push_back(parent2);
            }

            for (Chromosome& chromosome : new_population) {
                // �޸�����������Լ������
                repair_chromosome(chromosome.horizons);
                std::sort(chromosome.horizons.begin(), chromosome.horizons.end(), [](const Horizon& a, const Horizon& b) { return a.horizon_score < b.horizon_score; });
            }
            // ���㲢������Ӧ��
            for (Chromosome& chromosome : new_population) {
                chromosome.fitness = calculate_fitness(chromosome.horizons, N_BIOSTRAT);
            }
            // ���㵱ǰ���������Ӧ��
            auto current_chromosome = std::min_element(new_population.begin(), new_population.end(), [](const Chromosome& a, const Chromosome& b) { return a.fitness < b.fitness; });
            // ��������Ⱥ
            population = new_population;
            best_fitness = std::min(best_fitness, current_chromosome->fitness);
            // ��һ��horizon_score
            for (Chromosome& chromosome : population) {
                for (int i = 0; i < chromosome.horizons.size(); ++i) {
                    chromosome.horizons[i].horizon_score = i / double(chromosome.horizons.size());
                }
            }
            // ���ӽ��������������̷���/�����
            if ((generation % max_rank + 1) == rank) {

                // �������̷��͵�ǰ�ӽ��̵����Ž�
                int msg_type = MSG_SEND_BEST_SOLUTION;
                MPI_Send(&msg_type, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);

                // ���������̵�ȷ�Ͻ��ձ�ʶ
                MPI_Recv(&msg_type, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                // ���͵�ǰ�ӽ��̵����Ž�
                Chromosome best_solution;
                best_solution.horizons = current_chromosome->horizons;
                best_solution.fitness = current_chromosome->fitness;
                world.send(0, 0, best_solution);
                // ���͵�ǰ��������
                world.send(0, 0, generation);

                // ���������̷���һ�����Ž�
                msg_type = MSG_REQUEST_BEST_SOLUTION;
                MPI_Send(&msg_type, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);

                // ���������̵�ȷ�Ͻ��ձ�ʶ 
                MPI_Recv(&msg_type, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                Chromosome received_chromosome;
                world.recv(0, 0, received_chromosome);

                // ʹ�ý��յ��Ľ��Ž�����ӽ��̵���Ⱥ
                update_population_with_best_solution(population, received_chromosome);
            }
        }
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();
    long long max_duration;
    MPI_Reduce(&duration, &max_duration, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        auto best_solution = std::min_element(best_solutions.begin(), best_solutions.end(), [](const Chromosome& a, const Chromosome& b) { return a.fitness < b.fitness; });
        std::string prefix = "GA_output";
        std::string extension = ".csv";
        std::string output_file_path = generateFilename(prefix, extension);
        save_to_csv(best_solution->horizons, output_file_path, header_line);
        std::cout << "Total time elapsed: " << max_duration << " seconds" << std::endl;
    }
    MPI_Finalize();
    return 0;
}

std::vector<Horizon> read_csv_data(const std::string& file_path, std::string& header_line) {
    std::vector<Horizon> horizons;
    std::ifstream file(file_path);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open the file." << std::endl;
        return horizons;
    }

    // Ignore the header line
    std::getline(file, header_line);

    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string token;
        Horizon horizon;

        std::getline(ss, token, ',');
        horizon.section_name = token;

        std::getline(ss, token, ',');
        horizon.horizon_score = std::stod(token);

        std::getline(ss, token, ',');
        horizon.section_number = std::stoi(token);

        std::getline(ss, token, ',');
        horizon.horizon_number = std::stoi(token);

        std::getline(ss, token, ',');
        horizon.horizon_height = std::stoi(token);

        while (std::getline(ss, token, ',')) {
            horizon.presence_absence_data.push_back(std::stoi(token));
        }

        horizons.push_back(horizon);
    }

    file.close();
    return horizons;
}

void save_to_csv(const std::vector<Horizon>& horizons, const std::string& file_path, const std::string& header_line) {
    std::ofstream output_file(file_path);

    if (!output_file.is_open()) {
        std::cerr << "Error: Unable to open the output file." << std::endl;
        return;
    }

    // Write header
    output_file << header_line << std::endl;

    // Write data
    for (const auto& horizon : horizons) {
        output_file << horizon.section_name << ","
            << horizon.horizon_score << ","
            << horizon.section_number << ","
            << horizon.horizon_number << ","
            << horizon.horizon_height;

        for (int data : horizon.presence_absence_data) {
            output_file << "," << data;
        }
        output_file << "\n";
    }
    output_file.close();
    std::cout << "Results saved to: " << file_path << std::endl;
}

PenaltyParameters initialize_penalty_parameters() {
    PenaltyParameters params;

    //params.n_biostrat = 62;
    params.n_biostrat = 146;
    for (int i = 1; i <= 62; ++i) {
        params.biostrat_columns.push_back(i);
    }

    params.n_pmag = 0;
    params.pmag = 63;

    params.n_dates = 0;
    params.dates = {
        {109, 2, 110, 1, 100},
        {111, 2, 112, 1, 100},
        {113, 2, 114, 1, 100}
    };

    params.n_ashes = 0;
    params.ashes = {
        {68, 100},
        {69, 100}
    };

    params.n_continuous = 0;
    params.continuous = {
        {70, 5},
        {71, 5}
    };

    return params;
}

std::vector<Horizon> init_horizons(std::vector<Horizon>& horizons) {
    return horizons;
}

Chromosome init_chromosome(const std::vector<Horizon>& d3) {
    Chromosome chromosome;
    chromosome.horizons = d3;
    // Ϊÿ��section�ڵ�horizon�����µ�score
    for (size_t i = 0; i < chromosome.horizons.size();) {
        size_t section_start = i;
        int current_section = chromosome.horizons[i].section_number;

        // �ҵ���ǰsection�Ľ���λ��
        while (i < chromosome.horizons.size() && chromosome.horizons[i].section_number == current_section) {
            ++i;
        }

        // �����µ�score
        double last_score = 0.0;
        for (size_t j = section_start; j < i; ++j) {
            double delta_score = dis(gen);
            chromosome.horizons[j].horizon_score = last_score + delta_score;
            last_score = chromosome.horizons[j].horizon_score;
        }
    }
    // ����horizon_score��������
    std::sort(chromosome.horizons.begin(), chromosome.horizons.end(), [](const Horizon& a, const Horizon& b) { return a.horizon_score < b.horizon_score; });
    // ����������Сhorizon_score
    double min_horizon_score = chromosome.horizons.front().horizon_score;
    double max_horizon_score = chromosome.horizons.back().horizon_score;
    // ��һ��horizon_score
    for (Horizon& horizon : chromosome.horizons) {
        horizon.horizon_score = (horizon.horizon_score - min_horizon_score) / (max_horizon_score - min_horizon_score);
    }
    for (int i = 0; i < chromosome.horizons.size(); i++) {
        chromosome.horizons[i].horizon_score = i / double(chromosome.horizons.size());
    }
    // ������º���penalty
    chromosome.fitness = calculate_fitness(chromosome.horizons, chromosome.horizons[0].presence_absence_data.size());
    return chromosome;
}

int calculate_penalty(const std::vector<Horizon>& horizons, int n_biostrat) {
    int penalty = 0;

    for (int column = 0; column < n_biostrat; ++column) {
        bool first_one_found = false;
        int first_one_index = 0;
        int last_one_index = 0;

        for (int row = 0; row < horizons.size(); ++row) {
            int value = horizons[row].presence_absence_data[column];
            if (value == 1) {
                if (!first_one_found) {
                    first_one_found = true;
                    first_one_index = row;
                }
                last_one_index = row;
            }
        }

        for (int row = first_one_index; row <= last_one_index; ++row) {
            if (horizons[row].presence_absence_data[column] == 0) {
                ++penalty;
            }
        }
    }

    return penalty;
}

double calculate_fitness(const std::vector<Horizon>& horizons, int n_biostrat) {
    return static_cast<double>(calculate_penalty(horizons, n_biostrat));
}

std::vector<Chromosome> selection(const std::vector<Chromosome>& population, int tournament_size) {
    std::vector<Chromosome> selected_population;
    size_t population_size = population.size();
    // ���ѡ����
    std::uniform_int_distribution<size_t> random_selector(0, population_size - 1);

    // ѡ������Ⱥ
    for (size_t i = 0; i < population_size; ++i) {
        Chromosome best_candidate;
        double best_fitness = std::numeric_limits<double>::infinity();

        // ���ѡ�� tournament_size ��Ⱦɫ�岢�ҵ����������Ӧ�ȵ�Ⱦɫ��
        for (int j = 0; j < tournament_size; ++j) {
            size_t candidate_index = random_selector(gen);
            const Chromosome& candidate = population[candidate_index];

            if (candidate.fitness < best_fitness) {
                best_fitness = candidate.fitness;
                best_candidate = candidate;
            }
        }

        // �����Ⱦɫ�����ӵ�����Ⱥ��
        selected_population.push_back(best_candidate);
    }

    return selected_population;
}

std::tuple<std::vector<Horizon>, std::vector<Horizon>> crossover(const std::vector<Horizon>& parent1, const std::vector<Horizon>& parent2) {
    std::vector<Horizon> offspring1 = parent1;
    std::vector<Horizon> offspring2 = parent2;

    std::uniform_int_distribution<> dis_cross(0, int(parent1.size() - 1));
    std::uniform_real_distribution<> dis_move(-1.0 / parent1.size(), 1.0 / parent1.size());

    std::unordered_set<int> selected_indices;
    int crossover_size = int(1 * parent1.size() / 2);

    while (selected_indices.size() < crossover_size) {
        int index = dis_cross(gen);
        if (selected_indices.find(index) == selected_indices.end()) {
            selected_indices.insert(index);

            int section_number = offspring1[index].section_number;
            int horizon_number = offspring1[index].horizon_number;

            // Find the corresponding horizon in offspring2
            auto it = std::find_if(offspring2.begin(), offspring2.end(), [section_number, horizon_number](const Horizon& horizon) { return (horizon.section_number == section_number) & (horizon.horizon_number == horizon_number); });

            // Swap horizon_scores
            if (it != offspring2.end()) {
                double temp = offspring1[index].horizon_score;
                offspring1[index].horizon_score = it->horizon_score + dis_move(gen);
                it->horizon_score = temp + dis_move(gen);
            }
        }
    }

    return std::make_pair(offspring1, offspring2);
}

using Key = std::pair<int, int>;
// �Զ����ϣ����
struct KeyHash {
    std::size_t operator()(const Key& key) const {
        return std::hash<int>{}(key.first) ^ (std::hash<int>{}(key.second) << 1);
    }
};
using HorizonMap = std::unordered_map<Key, Horizon, KeyHash>;
std::tuple<std::vector<Horizon>, std::vector<Horizon>> crossover_pmx(const std::vector<Horizon>& parent1, const std::vector<Horizon>& parent2) {
    //for (std::size_t i = 0; i < parent1.size(); ++i) {
    //    if (parent1[i].horizon_score != parent2[i].horizon_score) {
    //        std::cout << "child1_score:" << parent1[i].horizon_score << "child2_score:" << parent2[i].horizon_score << std::endl;
    //    }
    //}
    size_t parent_size = parent1.size();
    std::vector<Horizon> child1(parent_size), child2(parent_size);
    HorizonMap mapping;
    // ������ɽ���Ƭ�ε���ʼ�ͽ���λ��
    size_t cut1 = rand() % parent_size;
    size_t cut2 = cut1 + rand() % (parent_size - cut1);
    for (std::size_t i = 0; i < parent1.size(); ++i) {
        mapping.insert({ {parent1[i].section_number, parent1[i].horizon_number}, parent1[i] });
    }
    // ����Ƭ��
    for (std::size_t i = cut1; i <= cut2; ++i) {
        child1[i] = parent2[i];
        child2[i] = parent1[i];
    }
    // �޸�ӳ���ϵ
    for (std::size_t i = 0; i < parent1.size(); ++i) {
        if (i < cut1 || i > cut2) {
            // �޸�child1ӳ��
            Key key_out1 = { parent1[i].section_number, parent1[i].horizon_number };
            size_t j = cut1;
            while (j <= cut2) {
                Key key_in = { child1[j].section_number, child1[j].horizon_number };
                if (key_out1 == key_in) {
                    Key key_new = { child2[j].section_number, child2[j].horizon_number };
                    auto it = mapping.find(key_new);
                    if (it != mapping.end()) {
                        child1[i] = it->second;
                        key_out1 = key_new;
                        j = cut1;
                    }
                }
                else {
                    j += 1;
                }
            }
            // �޸�child2ӳ��
            Key key_out2 = { parent2[i].section_number, parent2[i].horizon_number };
            size_t k = cut1;
            while (k <= cut2) {
                Key key_in = { child2[k].section_number, child2[k].horizon_number };
                if (key_out2 == key_in) {
                    Key key_new = { child1[k].section_number, child1[k].horizon_number };
                    auto it = mapping.find(key_new);
                    if (it != mapping.end()) {
                        child2[i] = it->second;
                        key_out2 = key_new;
                        k = cut1;
                    }
                }
                else {
                    k += 1;
                }
            }
        }
    }
    for (std::size_t i = 0; i < parent1.size(); ++i) {
        if (child1[i].section_name == "") {
            child1[i] = parent1[i];
        }
        if (child2[i].section_name == "") {
            child2[i] = parent2[i];
        }
        child1[i].horizon_score = parent1[i].horizon_score;
        child2[i].horizon_score = parent2[i].horizon_score;
    }
    return std::make_tuple(child1, child2);
}

std::vector<Horizon> mutation(const std::vector<Horizon>& individual, int n_biostrat) {
    std::vector<Horizon> mutated_individual = individual;
    std::uniform_int_distribution<int> index_dist(0, int(mutated_individual.size() - 1));
    std::uniform_int_distribution<int> biostrat_dist(0, n_biostrat - 1);
    std::uniform_real_distribution<> dis_move(-1.0 / individual.size(), 1.0 / individual.size());

    // �Ŷ�
    int num_swaps = 1 + index_dist(gen) % 1; // �� 1 �� 3 ֮�����ѡ���Ŷ�����
    for (int i = 0; i < num_swaps; ++i) {
        int index1 = index_dist(gen);
        int index2 = index_dist(gen);
        while (index1 == index2) {
            index2 = index_dist(gen);
        }
        std::swap(mutated_individual[index1].horizon_score, mutated_individual[index2].horizon_score);
        mutated_individual[index1].horizon_score += dis_move(gen);
        mutated_individual[index2].horizon_score += dis_move(gen);
    }

    return mutated_individual;
}

void update_fitness_improvement(Chromosome& chromosome, double old_fitness) {
    double improvement = old_fitness - chromosome.fitness;
    if (improvement > 0) {
        if (chromosome.local_operator_count > chromosome.global_operator_count) {
            chromosome.local_operator_improvement += improvement;
        }
        else {
            chromosome.global_operator_improvement += improvement;
        }
    }
}

void adaptive_local_mutation(Chromosome& chromosome, int mutation_range, int l, int history_rounds) {
    std::uniform_int_distribution<> mutation_position(0, chromosome.horizons.size() - 1);
    std::uniform_real_distribution<> mutation_dis(0.0, 1.0);

    double P_l = chromosome.local_operator_improvement / (chromosome.local_operator_improvement + chromosome.global_operator_improvement);
    double P_g = 1.0 - P_l;

    double old_fitness = chromosome.fitness;

    for (int i = 0; i < chromosome.horizons.size(); ++i) {
        double mutation_choice = mutation_dis(gen);
        if (mutation_choice < P_l) {
            // Apply local mutation operator
            int mutation_index = mutation_position(gen);
            int local_range = std::max(0, mutation_index - mutation_range);
            int global_range_left = std::max(0, mutation_index - l);
            int global_range_right = std::min(static_cast<int>(chromosome.horizons.size()) - 1, mutation_index + l);

            chromosome.horizons[mutation_index].horizon_score = chromosome.horizons[local_range].horizon_score +
                mutation_dis(gen) * (chromosome.horizons[mutation_index].horizon_score - chromosome.horizons[local_range].horizon_score);
            chromosome.local_operator_count++;
        }
        else {
            // Apply global mutation operator
            int mutation_index = mutation_position(gen);
            int global_range_left = std::max(0, mutation_index - l);
            int global_range_right = std::min(static_cast<int>(chromosome.horizons.size()) - 1, mutation_index + l);

            chromosome.horizons[mutation_index].horizon_score = mutation_dis(gen) * (chromosome.horizons[global_range_right].horizon_score - chromosome.horizons[global_range_left].horizon_score) + chromosome.horizons[global_range_left].horizon_score;
            chromosome.global_operator_count++;
        }
    }

    // Calculate the new fitness
    chromosome.fitness = calculate_fitness(chromosome.horizons, chromosome.horizons[0].presence_absence_data.size());

    // Update fitness improvement
    update_fitness_improvement(chromosome, old_fitness);

    // Update adaptive probabilities after a certain number of history rounds
    if (chromosome.local_operator_count + chromosome.global_operator_count >= history_rounds) {
        chromosome.local_operator_improvement /= chromosome.local_operator_count;
        chromosome.global_operator_improvement /= chromosome.global_operator_count;

        chromosome.local_operator_count = 0;
        chromosome.global_operator_count = 0;
    }
}

void apply_adaptive_local_mutation(std::vector<Chromosome>& population, int mutation_range, int l, int history_rounds) {
    for (Chromosome& chromosome : population) {
        adaptive_local_mutation(chromosome, mutation_range, l, history_rounds);
    }
}

void repair_chromosome(std::vector<Horizon>& chromosome) {
    // �� section_number �� chromosome ���з���
    std::map<int, std::vector<Horizon*>> sections;
    for (auto& horizon : chromosome) {
        sections[horizon.section_number].push_back(&horizon);
    }

    // ����ÿ�� section������ horizon_number �����������
    for (auto& section : sections) {
        std::sort(section.second.begin(), section.second.end(),
            [](const Horizon* a, const Horizon* b) {
                return a->horizon_number < b->horizon_number;
            });
        std::vector<double> sortedHorizonScores;
        for (const auto& horizon : section.second) {
            sortedHorizonScores.push_back(horizon->horizon_score);
        }
        std::sort(sortedHorizonScores.begin(), sortedHorizonScores.end());

        // ����horizon_scoreֵ
        for (size_t i = 1; i < section.second.size(); ++i) {
            //if (section.second[i]->horizon_score < section.second[i - 1]->horizon_score) {
            //    section.second[i]->horizon_score = section.second[i - 1]->horizon_score + 0.001;
            //}
            section.second[i]->horizon_score = sortedHorizonScores[i];
        }
    }
}

std::string getCurrentTimeString() {
    auto now = std::chrono::system_clock::now();
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);

    std::tm local_tm;
#ifdef _WIN32
    localtime_s(&local_tm, &now_time);
#else
    localtime_r(&now_time, &local_tm);
#endif

    std::ostringstream oss;
    oss << std::put_time(&local_tm, "%Y%m%d_%H%M%S");
    return oss.str();
}

std::string generateFilename(const std::string& prefix, const std::string& extension) {
    std::ostringstream oss;
    oss << prefix << "_" << getCurrentTimeString() << extension;
    return oss.str();
}

void update_population_with_best_solution(std::vector<Chromosome>& population, const Chromosome& received_best_solution) {
    // �ҵ���Ⱥ�������������
    int worst_solution_index = 0;
    double worst_fitness = population[0].fitness;

    for (size_t i = 1; i < population.size(); ++i) {
        if (population[i].fitness > worst_fitness) {
            worst_fitness = population[i].fitness;
            worst_solution_index = i;
        }
    }

    // �ý��յ��Ľ��Ž��滻������
    population[worst_solution_index] = received_best_solution;
}

void update_best_solution_library(std::vector<Chromosome>& best_solution_library, const Chromosome& received_solution, size_t LIBRARY_SIZE) {
    // ������յ��Ľ����Ӧ��
    double received_fitness = received_solution.fitness;

    // Ѱ�ҿ�����Ӧ�Ƚϲ�Ľ������
    int worst_solution_index = -1;
    double worst_fitness = -1;

    // �������еĽ⣬Ѱ����Ӧ�Ƚϲ�Ľ�
    for (size_t i = 0; i < best_solution_library.size(); ++i) {
        if (worst_solution_index == -1 || best_solution_library[i].fitness > worst_fitness) {
            worst_solution_index = i;
            worst_fitness = best_solution_library[i].fitness;
        }
    }

    // ������յ��Ľ����Ӧ�Ⱥ��ڿ����������Ӧ�ȣ��ý��յ��Ľ��滻��������
    if (received_fitness < worst_fitness) {
        best_solution_library[worst_solution_index] = received_solution;
    }
    // �����δ������ֱ�ӽ����յ��Ľ����ӵ�����
    else if (best_solution_library.size() < LIBRARY_SIZE) {
        best_solution_library.push_back(received_solution);
    }
}