#include "HORSE.h"

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
    //const std::string file_path = "18988_GA_output_20230905_142250.csv";
    std::string header_line;
    std::vector<Horizon> horizons = read_csv_data(file_path, header_line);

    srand(time(NULL) * rank); // �������������
    const int N_HORZIONS = horizons.size(); // ����ˮƽ������
    const int N_BIOSTRAT = horizons[0].presence_absence_data.size(); // ������������
    const int num_generations = 1000; // outer_num(����������)
    const int population_size = 100; // inner_num
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
            std::vector<Chromosome> mating_pool = selection(population, int(5 * population_size / 10)); // 20%
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
