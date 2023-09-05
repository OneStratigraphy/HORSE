#include "HORSE.h"

int main(int argc, char* argv[]) {
    // MPI 初始化
    MPI_Init(&argc, &argv);
    auto start_time = std::chrono::high_resolution_clock::now();
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int max_rank = size - 1;

    boost::mpi::environment env(argc, argv);
    boost::mpi::communicator world;

    // 从csv文件读取数据
    const std::string file_path = "./dataset/HA_Fan_et_al_2013.csv";
    //const std::string file_path = "18988_GA_output_20230905_142250.csv";
    std::string header_line;
    std::vector<Horizon> horizons = read_csv_data(file_path, header_line);

    srand(time(NULL) * rank); // 设置随机数种子
    const int N_HORZIONS = horizons.size(); // 设置水平层数量
    const int N_BIOSTRAT = horizons[0].presence_absence_data.size(); // 设置生物数量
    const int num_generations = 1000; // outer_num(最大迭代次数)
    const int population_size = 100; // inner_num
    const int exchange_interval = 5; // 设置交换间隔
    const int library_size = 10; // 设置最优解库大小
    double crossover_rate = 0.99;
    double mutation_rate = 0.99;
    int mutation_range = 5;
    int l = 10;
    int history_rounds = 10;

    // 设置消息标识
    const int MSG_REQUEST_BEST_SOLUTION = 13;
    const int MSG_SEND_BEST_SOLUTION = 14;
    std::vector<Chromosome> best_solutions(library_size); // 创建最优解库
    if (rank == 0) { // 主进程（负责维护最优解库）
        MPI_Status status;

        // 初始化解库
        for (int i = 0; i < library_size; ++i) {
            best_solutions[i] = init_chromosome(horizons);
        }

        int count = 0;
        while (count < 2 * num_generations) {
            count++;
            int msg_type;
            MPI_Recv(&msg_type, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            int sender_rank = status.MPI_SOURCE; // 获取对应子进程序号

            if (msg_type == MSG_REQUEST_BEST_SOLUTION) { // 发送解
                // 向子进程发送确认接收标识
                MPI_Send(&msg_type, 1, MPI_INT, sender_rank, 0, MPI_COMM_WORLD);

                // 从解库中随机挑选一个较优解发送给子进程
                int target_rank = rand() % library_size;
                Chromosome best_solution = best_solutions[target_rank];
                world.send(sender_rank, 0, best_solution);
            }
            else if (msg_type == MSG_SEND_BEST_SOLUTION) { // 接收解
                // 向子进程发送确认接收标识
                MPI_Send(&msg_type, 1, MPI_INT, sender_rank, 0, MPI_COMM_WORLD);

                // 接收子进程发送的较优解
                Chromosome received_chromosome;
                world.recv(sender_rank, 0, received_chromosome);
                int generation = 0;
                world.recv(sender_rank, 0, generation);

                // 使用接收到的当前最优解更新全局最优解库
                update_best_solution_library(best_solutions, received_chromosome, library_size);
                // 计算解库中最优解
                auto best_solution = std::min_element(best_solutions.begin(), best_solutions.end(), [](const Chromosome& a, const Chromosome& b) { return a.fitness < b.fitness; });
                std::cout << "Generation:" << generation << " | Rank:" << sender_rank << " | Current fitness:" << received_chromosome.fitness << " | Best fitness:" << best_solution->fitness << std::endl;
            }
        }
    }
    else { // 子进程（负责解的搜索）
        // 初始化种群
        std::vector<Horizon> d3 = horizons;
        std::vector<Chromosome> population;
        for (int i = 0; i < population_size; ++i) {
            population.push_back(init_chromosome(d3));
        }
        // 更新所有个体适应度并计算最佳适应度
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
            // 执行遗传算法的一次迭代
            // 选择（锦标赛策略）：从population中随机挑选至mating_pool，size不变
            std::vector<Chromosome> mating_pool = selection(population, int(5 * population_size / 10)); // 20%
            std::vector<Chromosome> new_population;

            //// 自适应局部变异
            //apply_adaptive_local_mutation(mating_pool, mutation_range, l, history_rounds);

            // 划分种群为相等两部分进行操作
            for (int j = 0; j < population_size; j += 2) {
                Chromosome parent1 = mating_pool[j];
                Chromosome parent2 = mating_pool[j + 1];
                // 交叉：任选
                if (dis(gen) < crossover_rate) {
                    std::tie(parent1.horizons, parent2.horizons) = crossover_pmx(parent1.horizons, parent2.horizons);
                }
                // 突变
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
                // 修复个体以满足约束条件
                repair_chromosome(chromosome.horizons);
                std::sort(chromosome.horizons.begin(), chromosome.horizons.end(), [](const Horizon& a, const Horizon& b) { return a.horizon_score < b.horizon_score; });
            }
            // 计算并更新适应度
            for (Chromosome& chromosome : new_population) {
                chromosome.fitness = calculate_fitness(chromosome.horizons, N_BIOSTRAT);
            }
            // 计算当前代的最佳适应度
            auto current_chromosome = std::min_element(new_population.begin(), new_population.end(), [](const Chromosome& a, const Chromosome& b) { return a.fitness < b.fitness; });
            // 更新子种群
            population = new_population;
            best_fitness = std::min(best_fitness, current_chromosome->fitness);
            // 归一化horizon_score
            for (Chromosome& chromosome : population) {
                for (int i = 0; i < chromosome.horizons.size(); ++i) {
                    chromosome.horizons[i].horizon_score = i / double(chromosome.horizons.size());
                }
            }
            // 各子进程轮流向主进程发送/请求解
            if ((generation % max_rank + 1) == rank) {

                // 向主进程发送当前子进程的最优解
                int msg_type = MSG_SEND_BEST_SOLUTION;
                MPI_Send(&msg_type, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);

                // 接收主进程的确认接收标识
                MPI_Recv(&msg_type, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                // 发送当前子进程的最优解
                Chromosome best_solution;
                best_solution.horizons = current_chromosome->horizons;
                best_solution.fitness = current_chromosome->fitness;
                world.send(0, 0, best_solution);
                // 发送当前迭代次数
                world.send(0, 0, generation);

                // 请求主进程发送一个较优解
                msg_type = MSG_REQUEST_BEST_SOLUTION;
                MPI_Send(&msg_type, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);

                // 接收主进程的确认接收标识 
                MPI_Recv(&msg_type, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                Chromosome received_chromosome;
                world.recv(0, 0, received_chromosome);

                // 使用接收到的较优解更新子进程的种群
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
