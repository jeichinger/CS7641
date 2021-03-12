from mlrose_hiive.runners import RHCRunner
from mlrose_hiive.runners import SARunner
from mlrose_hiive.runners import GARunner
from mlrose_hiive.runners import MIMICRunner
from CustomKnapSackGenerator import CustomKnapSackGenerator
from mlrose_hiive.algorithms.crossovers import one_point_crossover
from mlrose_hiive.algorithms.mutators import change_one_mutator
import os
import sys
import time
import numpy as np
import mlrose_hiive
import matplotlib.pyplot as plt

DATA_PATH = os.getcwd() + "/Data"
LOG_PATH = os.getcwd() + "/Logs"
OUTPUT_PATH = os.getcwd() + "/Output"
TRAVELING_SALES_PERSON_PATH = OUTPUT_PATH + "/CPP"
KNAPSACK_PATH = OUTPUT_PATH + "/KNAPSACK"
K_COLOR_PATH = OUTPUT_PATH + "/KCOLOR"

RHC_PATH = "/RHC"
SA_PATH = "/SA"
GA_PATH = "/GA"
MIMIC_PATH = "/MIMIC"

def run_knapsack_experiment(run_rhc, run_sa, run_ga, run_mimic, find_params):

    title_dic = {'fontsize': 7, 'fontweight': 'bold'}
    axis_label_size = 7
    tuning_figure_size = (3.5,2)

    np.random.seed(0)

    problem = CustomKnapSackGenerator.generate(1,
                                               number_of_items_types=100,
                                               max_item_count=50,
                                               max_weight_per_item=100,
                                               max_value_per_item=50,
                                               max_weight_pct=0.6,
                                               multiply_by_max_item_count=True,
                                               crossover=one_point_crossover.OnePointCrossOver,
                                               mutator=change_one_mutator.ChangeOneMutator)

    rhc_run_iterations = None
    rhc_run_average_restarts = None
    rhc_run_average_time = None

    sa_run_iterations = None
    sa_run_average = None
    sa_run_average_time = None

    ga_run_iterations = None
    ga_run_average = None
    ga_run_average_time = None

    mimic_run_iterations = None
    mimic_run_average = None
    mimic_run_average_time = None

    if run_rhc:

        out_dir = KNAPSACK_PATH + RHC_PATH

        if find_params:

            fig, (ax1), = plt.subplots(1, 1, figsize=tuning_figure_size)
            ax1.set_title("Randomized Hill Climbing with Various Restarts \n Continous Peaks Problem", title_dic)
            ax1.set_ylabel("Average Fitness Score", title_dic)
            ax1.tick_params(axis="x", labelsize=axis_label_size)
            ax1.tick_params(axis="y", labelsize=axis_label_size)
            ax1.set_xlabel("Iteration", title_dic)

            num_restarts = [3, 5, 15, 25, 50]
            restart_params = {3:[], 5:[], 15:[], 25:[], 50:[]}

            for param in num_restarts:

                restarts = []
                iterations = []

                for i in range(param):
                    seed = np.random.randint(100)

                    # run RHC runner with param restarts.
                    rhc = RHCRunner(problem=problem,
                                    experiment_name="Grid Search Restart " + str(i) + " of max restarts " + str(param),
                                    output_directory=out_dir,
                                    seed=seed,
                                    iteration_list=np.arange(1, 1000, 100),
                                    max_attempts=500,
                                    restart_list=[0])

                    rhc_run_stats, _ = rhc.run()
                    iterations = rhc_run_stats['Iteration'].tolist()

                    fitness = np.asarray(rhc_run_stats['Fitness'].tolist())

                    restarts.append(fitness)

                average_restarts = np.mean(np.asarray(restarts), axis=0)
                ax1.plot(iterations, average_restarts, label=str(param) + " Restarts")

            # Put a legend to the right of the current axis
            box = ax1.get_position()
            ax1.set_position([box.x0, box.y0, box.width * 0.99, box.height])
            ax1.legend(loc='best', fontsize=6)
            ax1.grid()
            plt.tight_layout()
            filename = "various restarts CPP problem.png"
            filename = os.path.join(out_dir, filename)
            plt.savefig(filename)

        # Reproduce results from 3 restart run.
        restarts = []
        iterations = []
        times = []
        for i in range(15):

            # run RHC runner with param restarts.
            rhc = RHCRunner(problem=problem,
                            experiment_name="RHC QUEENS" + str(i),
                            output_directory=out_dir,
                            seed=np.random.randint(100),
                            iteration_list=np.arange(1, 1000, 100),
                            max_attempts=500,
                            restart_list=[0])

            rhc_run_stats, _ = rhc.run()
            rhc_run_iterations = rhc_run_stats['Iteration'].tolist()
            fitness = np.asarray(rhc_run_stats['Fitness'].tolist())
            restarts.append(fitness)
            times.append(rhc_run_stats["Time"].iloc[-1] * 1000)

        rhc_run_average_restarts = np.mean(np.asarray(restarts), axis=0)
        rhc_run_average_time = np.mean(times)

    if run_sa:

        out_dir = KNAPSACK_PATH + SA_PATH

        if find_params:

            #decay_list = [mlrose_hiive.GeomDecay, mlrose_hiive.ExpDecay]

            # Round two: Exp Decay was better
            decay_list = [mlrose_hiive.ExpDecay]

            decay_names = {mlrose_hiive.GeomDecay: ('Geometric', 'dotted'),
                           mlrose_hiive.ExpDecay: ('Exponential', 'solid'),
                           mlrose_hiive.ArithDecay: ('Arithmetic', 'dashed')}

            temperature_list = [500,1000, 2500,3500,4500,5000]

            parameters = []
            for i in decay_list:
                for j in temperature_list:
                    parameters.append((i, j))

            fig, (ax1), = plt.subplots(1, 1, figsize=tuning_figure_size)
            ax1.set_title("Simulated Annealing with Exponential Decay \n Schedule and Initial Temperature - Knapsack", title_dic)
            ax1.set_ylabel("Average Fitness Score", title_dic)
            ax1.tick_params(axis="x", labelsize=axis_label_size)
            ax1.tick_params(axis="y", labelsize=axis_label_size)
            ax1.set_xlabel("Iteration", title_dic)

            for entry in parameters:

                runs = []
                for i in range(5):
                    sa = SARunner(problem=problem,
                                  experiment_name="KnapSack Simulated Annealing with Decay Schedule" + str(
                                      entry[0]) + "and Initial Temperature " + str(entry[1]) + "Run " + str(i),
                                  output_directory=out_dir,
                                  seed=np.random.randint(100),
                                  iteration_list=np.arange(1, 2000, 100),
                                  max_attempts=1000,
                                  decay_list=[entry[0]],
                                  temperature_list=[entry[1]])

                    sa_run_stats, _ = sa.run()

                    sa_run_iterations = sa_run_stats['Iteration'].tolist()
                    fitness = np.asarray(sa_run_stats['Fitness'].tolist())
                    runs.append(fitness)

                sa_run_average = np.mean(np.asarray(runs), axis=0)

                ax1.plot(sa_run_iterations, sa_run_average, linestyle=decay_names[entry[0]][1],
                         label="Init Temp:" + str(entry[1]))

            ax1.legend(loc='best', fontsize=6)
            ax1.grid()
            plt.tight_layout()
            filename = "decay_and_initial_temp.png"
            filename = os.path.join(out_dir, filename)
            plt.savefig(filename)

        # Run 5 times and average
        runs = []
        times = []
        for i in range(5):

            sa = SARunner(problem=problem,
                          experiment_name="Knapsack Simulated Annealing after Param Tuning Decay Schedule " + str(i),
                          output_directory=out_dir,
                          seed=np.random.randint(100),
                          iteration_list=np.arange(1, 1500, 100),
                          max_attempts=1000,
                          decay_list=[mlrose_hiive.ExpDecay],
                          temperature_list=[3500])

            sa_run_stats, _ = sa.run()

            sa_run_iterations = sa_run_stats['Iteration'].tolist()

            times.append(sa_run_stats["Time"].iloc[-1] * 1000)
            fitness = np.asarray(sa_run_stats['Fitness'].tolist())
            runs.append(fitness)

        sa_run_average = np.mean(np.asarray(runs), axis=0)
        sa_run_average_time = np.mean(times)

    if run_ga:

        out_dir = KNAPSACK_PATH + GA_PATH

        if find_params:

            population_sizes = [25, 50, 75, 100]

            # first iteration revealed a pop size of 75 was sufficient.
            population_sizes = [75]

            pop_keys = {
                100: ('dashed'),
                50: ('dashed'),
                75: ('solid')
            }

            mutation_rates = [0.1, 0.3, 0.6, 0.9,]

            parameters = []
            for i in population_sizes:
                for j in mutation_rates:
                    parameters.append((i, j))

            fig, (ax1), = plt.subplots(1, 1, figsize=tuning_figure_size)
            ax1.set_title("Genetic Algorithms with Population Size 75 & \n Various Mutation Rates - Knapsack", title_dic)
            ax1.set_ylabel("Average Fitness Score", title_dic)
            ax1.tick_params(axis="x", labelsize=axis_label_size)
            ax1.tick_params(axis="y", labelsize=axis_label_size)
            ax1.set_xlabel("Iteration", title_dic)

            for entry in parameters:

                runs = []
                for i in range(5):
                    ga = GARunner(problem=problem,
                                  experiment_name="Knapsack GA with Population" + str(
                                      entry[0]) + "and Mutation Split " + str(entry[1]),
                                  output_directory=out_dir,
                                  seed=np.random.randint(100),
                                  iteration_list=np.arange(1, 1500, 25),
                                  max_attempts=150,
                                  population_sizes=[entry[0]],
                                  mutation_rates=[entry[1]])

                    ga_run_stats, _ = ga.run()

                    fitness = np.asarray(ga_run_stats['Fitness'].tolist())
                    ga_run_iterations = ga_run_stats['Iteration'].tolist()

                    runs.append(fitness)

                ga_run_average = np.mean(np.asarray(runs), axis=0)

                ax1.plot(ga_run_iterations, ga_run_average, linestyle=pop_keys[entry[0]],
                         label="MutRate:" + str(entry[1]))

            ax1.legend(loc='best', fontsize=6, ncol=2)
            ax1.grid()
            plt.tight_layout()
            filename = "population_and_mutation_rates.png"
            filename = os.path.join(out_dir, filename)
            plt.savefig(filename)

        runs = []
        times = []
        for i in range(5):

            ga = GARunner(problem=problem,
                          experiment_name="CPP GA After Parameter Tuning" + "Run " + str(i),
                          output_directory=out_dir,
                          seed=np.random.randint(100),
                          iteration_list=np.arange(1, 1500, 25),
                          max_attempts=150,
                          population_sizes=[75],
                          mutation_rates=[0.9])

            ga_run_stats, _ = ga.run()
            ga_run_iterations = ga_run_stats['Iteration'].tolist()
            times.append(ga_run_stats["Time"].iloc[-1] * 1000)
            fitness = np.asarray(ga_run_stats['Fitness'].tolist())
            runs.append(fitness)

        ga_run_average = np.mean(np.asarray(runs), axis=0)
        ga_run_average_time = np.mean(times)

    if run_mimic:

        out_dir = KNAPSACK_PATH + MIMIC_PATH

        if find_params:

            population_sizes = [25, 50, 100, 150, 200]

            # Pop size of 100 was best tradeoff with time.
            population_sizes = [100]
            keep_percent = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,]

            pop_keys = {
                100: ('solid')
            }

            parameters = []
            for i in population_sizes:
                for j in keep_percent:
                    parameters.append((i, j))

            fig, (ax1), = plt.subplots(1, 1, figsize=tuning_figure_size)
            ax1.set_title("Mimic with 100 Population Size and Various \n Keep Percentages - Knapsack", title_dic)
            ax1.set_ylabel("Average Fitness Score", title_dic)
            ax1.tick_params(axis="x", labelsize=axis_label_size)
            ax1.tick_params(axis="y", labelsize=axis_label_size)
            ax1.set_xlabel("Iteration", title_dic)

            for entry in parameters:

                runs = []
                for i in range(5):

                    mimic = MIMICRunner(problem=problem,
                                     experiment_name="Knapsack Mimic with Population" + str(
                                         entry[0]) + "and Keep Percent " + str(entry[1]) + "run " + str(i),
                                     output_directory=out_dir,
                                     seed=np.random.randint(100),
                                     iteration_list=np.arange(1, 50, 1),
                                     population_sizes=[entry[0]],
                                     keep_percent_list=[entry[1]],
                                     max_attempts=5, generate_curves=True, use_fast_mimic=True)

                    df_run_stats, _ = mimic.run()

                    mimic_run_iterations = df_run_stats['Iteration'].tolist()
                    fitness = np.asarray(df_run_stats['Fitness'].tolist())
                    runs.append(fitness)

                mimic_run_average = np.mean(np.asarray(runs), axis=0)

                ax1.plot(mimic_run_iterations, mimic_run_average, linestyle=pop_keys[entry[0]],
                         label="Keep %:" + str(entry[1]))

            # Put a legend to the right of the current axis
            ax1.legend(loc='best', fontsize=6)
            ax1.grid()
            plt.tight_layout()
            filename = "population_and_keep_percent_rates.png"
            filename = os.path.join(out_dir, filename)
            plt.savefig(filename)

        runs = []
        times = []
        for i in range(5):

            mimic = MIMICRunner(problem=problem,
                                 experiment_name="Knapsack Mimic" + "Run " + str(i),
                                 output_directory=out_dir,
                                 seed=np.random.randint(100),
                                 iteration_list=np.arange(1, 150, 5),
                                 population_sizes=[100],
                                 keep_percent_list=[0.5],
                                 max_attempts=5, generate_curves=True, use_fast_mimic=True)

            mimic_run_stats, _ = mimic.run()
            mimic_run_iterations = mimic_run_stats['Iteration'].tolist()
            times.append(mimic_run_stats["Time"].iloc[-1] * 1000)
            fitness = np.asarray(mimic_run_stats['Fitness'].tolist())
            runs.append(fitness)

        mimic_run_average = np.mean(np.asarray(runs), axis=0)
        mimic_run_average_time = np.mean(times)

    fig, (fitness_cp, time_cp) = plt.subplots(1, 2, figsize=(6, 2))
    fitness_cp.set_title(
        "Fitness vs Iterations\n Knapsack N = 100", title_dic)
    fitness_cp.set_ylabel("Average Fitness Score", title_dic)
    fitness_cp.tick_params(axis="x", labelsize=axis_label_size)
    fitness_cp.tick_params(axis="y", labelsize=axis_label_size)
    fitness_cp.set_xlabel("Iteration", title_dic)

    fitness_cp.plot(sa_run_iterations, sa_run_average, label="SA")
    fitness_cp.plot(ga_run_iterations, ga_run_average, label="GA")
    fitness_cp.plot(rhc_run_iterations, rhc_run_average_restarts, label="RHC")
    fitness_cp.plot(mimic_run_iterations, mimic_run_average, label="MIMIC")

    fitness_cp.legend(loc='best', fontsize=6)

    time_cp.set_xlabel("Log Time (ms)", title_dic)
    time_cp.set_title("Runtime \n Knapsack N = 100", title_dic)
    time_cp.set_ylabel("Algorithm", title_dic)
    time_cp.tick_params(axis="x", labelsize=axis_label_size)
    time_cp.tick_params(axis="y", labelsize=axis_label_size)

    x = ['RHC', 'SA', 'GA', 'MIMIC']
    x_pos = [i for i, _ in enumerate(x)]
    time_cp.set_yticks(x_pos)
    time_cp.set_yticklabels(x)
    time_cp.set_xscale('log')
    time_cp.barh(x_pos, [rhc_run_average_time, sa_run_average_time, ga_run_average_time, mimic_run_average_time])

    time_cp.grid()
    fitness_cp.grid()
    plt.tight_layout()

    path = KNAPSACK_PATH
    filename = "fitness_vs_iteration.png"
    filename = os.path.join(path, filename)
    plt.savefig(filename)

    fig, (fitness_cp, time_cp) = plt.subplots(1, 2, figsize=(6, 2))
    fitness_cp.set_title(
        "Fitness vs Problem Size\n Knapsack Problem", title_dic)
    fitness_cp.set_ylabel("Average Fitness Score", title_dic)
    fitness_cp.tick_params(axis="x", labelsize=axis_label_size)
    fitness_cp.tick_params(axis="y", labelsize=axis_label_size)
    fitness_cp.set_xlabel("Problem Size(N)", title_dic)

    # Running on different problem set sizes
    sizes = [50, 75, 100, 125, 150]

    rhc_average_size = []
    sa_average_size = []
    ga_average_size = []
    mimic_average_size = []

    rhc_average_time = []
    sa_average_time = []
    ga_average_time = []
    mimic_average_time = []

    out_dir = KNAPSACK_PATH

    for size in sizes:

        problem = CustomKnapSackGenerator.generate(1,
                                                   number_of_items_types=size,
                                                   max_item_count=50,
                                                   max_weight_per_item=100,
                                                   max_value_per_item=50,
                                                   max_weight_pct=0.6,
                                                   multiply_by_max_item_count=True,
                                                   crossover=one_point_crossover.OnePointCrossOver,
                                                   mutator=change_one_mutator.ChangeOneMutator)

        restarts = []
        iterations = []
        times = []
        for i in range(15):
            # run RHC runner with param restarts.
            rhc = RHCRunner(problem=problem,
                            experiment_name="RHC QUEENS" + str(i),
                            output_directory=out_dir,
                            seed=np.random.randint(100),
                            iteration_list=np.arange(1, 1000, 100),
                            max_attempts=500,
                            restart_list=[0])

            rhc_run_stats, _ = rhc.run()
            rhc_run_iterations = rhc_run_stats['Iteration'].tolist()
            restarts.append(rhc_run_stats['Fitness'].tolist())
            times.append(rhc_run_stats["Time"].iloc[-1] * 1000)

        rhc_run_average_restarts = np.mean(np.asarray(restarts), axis=0)
        rhc_run_average_time = np.mean(times)

        rhc_average_size.append(rhc_run_average_restarts[-1])
        rhc_average_time.append(rhc_run_average_time)

        runs = []
        times = []
        for i in range(5):

            sa = SARunner(problem=problem,
                          experiment_name="Knapsack Simulated Annealing after Param Tuning Decay Schedule " + str(i),
                          output_directory=out_dir,
                          seed=np.random.randint(100),
                          iteration_list=np.arange(1, 1500, 100),
                          max_attempts=1000,
                          decay_list=[mlrose_hiive.ExpDecay],
                          temperature_list=[3500])

            sa_run_stats, _ = sa.run()

            sa_run_iterations = sa_run_stats['Iteration'].tolist()

            times.append(sa_run_stats["Time"].iloc[-1] * 1000)
            runs.append(sa_run_stats['Fitness'].tolist())

        sa_run_average = np.mean(np.asarray(runs), axis=0)
        sa_run_average_time = np.mean(times)

        sa_average_size.append(sa_run_average[-1])
        sa_average_time.append(sa_run_average_time)

        runs = []
        times = []
        for i in range(5):

            ga = GARunner(problem=problem,
                          experiment_name="CPP GA After Parameter Tuning" + "Run " + str(i),
                          output_directory=out_dir,
                          seed=np.random.randint(100),
                          iteration_list=np.arange(1, 1500, 25),
                          max_attempts=150,
                          population_sizes=[75],
                          mutation_rates=[0.9])

            ga_run_stats, _ = ga.run()
            ga_run_iterations = ga_run_stats['Iteration'].tolist()
            times.append(ga_run_stats["Time"].iloc[-1] * 1000)
            runs.append(ga_run_stats['Fitness'].tolist())

        ga_run_average = np.mean(np.asarray(runs), axis=0)
        ga_run_average_time = np.mean(times)

        ga_average_size.append(ga_run_average[-1])
        ga_average_time.append(ga_run_average_time)

        runs = []
        times = []
        for i in range(5):

            mimic = MIMICRunner(problem=problem,
                                 experiment_name="Knapsack Mimic" + "Run " + str(i),
                                 output_directory=out_dir,
                                 seed=np.random.randint(100),
                                 iteration_list=np.arange(1, 150, 5),
                                 population_sizes=[100],
                                 keep_percent_list=[0.5],
                                 max_attempts=5, generate_curves=True, use_fast_mimic=True)

            mimic_run_stats, _ = mimic.run()
            mimic_run_iterations = mimic_run_stats['Iteration'].tolist()
            times.append(mimic_run_stats["Time"].iloc[-1] * 1000)
            runs.append(mimic_run_stats['Fitness'].tolist())

        mimic_run_average = np.mean(np.asarray(runs), axis=0)
        mimic_run_average_time = np.mean(times)

        mimic_average_size.append(mimic_run_average[-1])
        mimic_average_time.append(mimic_run_average_time)

    fitness_cp.plot(sizes, sa_average_size, label="SA")
    fitness_cp.plot(sizes, ga_average_size, label="GA")
    fitness_cp.plot(sizes, rhc_average_size, label="RHC")
    fitness_cp.plot(sizes, mimic_average_size, label="MIMIC")

    fitness_cp.legend(loc='best', fontsize=6)

    time_cp.set_ylabel("Log Time (ms)", title_dic)
    time_cp.set_title("Runtime vs Problem Size \n Knapsack Problem", title_dic)
    time_cp.set_xlabel("Problem Size(N)", title_dic)
    time_cp.tick_params(axis="x", labelsize=axis_label_size)
    time_cp.tick_params(axis="y", labelsize=axis_label_size)
    time_cp.set_yscale('log')

    time_cp.plot(sizes, sa_average_time, label="SA")
    time_cp.plot(sizes, ga_average_time, label="GA")
    time_cp.plot(sizes, rhc_average_time, label="RHC")
    time_cp.plot(sizes, mimic_average_time, label="MIMIC")
    time_cp.legend(loc='best', fontsize=6)

    time_cp.grid()
    fitness_cp.grid()
    plt.tight_layout()

    path = KNAPSACK_PATH
    filename = "fitness_vs_iteration_various_problem_sizes.png"
    filename = os.path.join(path, filename)
    plt.savefig(filename)