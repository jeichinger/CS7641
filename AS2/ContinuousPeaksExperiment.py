from mlrose_hiive.runners import RHCRunner
from mlrose_hiive.runners import SARunner
from mlrose_hiive.runners import GARunner
from mlrose_hiive.runners import MIMICRunner
from CustomContinuousPeaksGenerator import CustomContinuousPeaksGenerator
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
QUEENS_PATH = OUTPUT_PATH + "/QUEENS"
K_COLOR_PATH = OUTPUT_PATH + "/KCOLOR"

RHC_PATH = "/RHC"
SA_PATH = "/SA"
GA_PATH = "/GA"
MIMIC_PATH = "/MIMIC"

def run_continuous_peaks(run_rhc, run_sa, run_ga, run_mimic, find_params):

    np.random.seed(0)

    continuous_peaks_problem = CustomContinuousPeaksGenerator.generate(1, 80)

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

        out_dir = TRAVELING_SALES_PERSON_PATH + RHC_PATH

        if find_params:

            title_dic = {'fontsize': 6, 'fontweight': 'bold'}
            fig, (ax1), = plt.subplots(1, 1, figsize=(3, 2))
            ax1.set_title("Randomized Hill Climbing with Various Restarts \n Continous Peaks Problem", title_dic)
            ax1.set_ylabel("Average Fitness Score", title_dic)
            ax1.tick_params(axis="x", labelsize=6)
            ax1.tick_params(axis="y", labelsize=6)
            ax1.set_xlabel("Iteration", title_dic)

            num_restarts = [3, 5, 15, 25, 50]
            restart_params = {3:[], 5:[], 15:[], 25:[], 50:[]}

            for param in num_restarts:

                restarts = []
                iterations = []

                for i in range(param):
                    seed = np.random.randint(100)

                    # run RHC runner with param restarts.
                    rhc = RHCRunner(problem=continuous_peaks_problem,
                                    experiment_name="Grid Search Restart " + str(i) + " of max restarts " + str(param),
                                    output_directory=out_dir,
                                    seed=seed,
                                    iteration_list=np.arange(1, 2200, 100),
                                    max_attempts=500,
                                    restart_list=[0])

                    rhc_run_stats, _ = rhc.run()
                    iterations = rhc_run_stats['Iteration'].tolist()
                    restarts.append(rhc_run_stats['Fitness'].tolist())

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
            rhc = RHCRunner(problem=continuous_peaks_problem,
                            experiment_name="RHC CPP",
                            output_directory=out_dir,
                            seed=np.random.randint(100),
                            iteration_list=np.arange(1, 3500, 100),
                            max_attempts=500,
                            restart_list=[0])

            rhc_run_stats, _ = rhc.run()
            rhc_run_iterations = rhc_run_stats['Iteration'].tolist()
            restarts.append(rhc_run_stats['Fitness'].tolist())
            times.append(rhc_run_stats["Time"].iloc[-1] * 1000)

        rhc_run_average_restarts = np.mean(np.asarray(restarts), axis=0)
        rhc_run_average_time = np.mean(times)

    if run_sa:

        out_dir = TRAVELING_SALES_PERSON_PATH + SA_PATH

        if find_params:

            decay_list = [mlrose_hiive.GeomDecay, mlrose_hiive.ExpDecay]

            decay_names = {mlrose_hiive.GeomDecay: ('Geometric', 'solid'),
                           mlrose_hiive.ExpDecay: ('Exponential', 'dotted'),
                           mlrose_hiive.ArithDecay: ('Arithmetic', 'dashed')}

            # Arithmetic decay was way too slow. No progress was getting made.
            decay_names = {mlrose_hiive.GeomDecay: ('Geometric', 'solid'),
                           mlrose_hiive.ExpDecay: ('Exponential', 'dotted')}
                           #mlrose_hiive.ArithDecay: ('Arithmetic', 'dashed')}

            temperature_list = [1, 10, 50, 100, 250, 500, 1000]

            parameters = []
            for i in decay_list:
                for j in temperature_list:
                    parameters.append((i, j))

            title_dic = {'fontsize': 6, 'fontweight': 'bold'}
            fig, (ax1), = plt.subplots(1, 1, figsize=(5, 2.5))
            ax1.set_title("Simulated Annealing with Various Decay Schedules \n and Initial Temperature", title_dic)
            ax1.set_ylabel("Average Fitness Score", title_dic)
            ax1.tick_params(axis="x", labelsize=6)
            ax1.tick_params(axis="y", labelsize=6)
            ax1.set_xlabel("Iteration", title_dic)

            for entry in parameters:

                runs = []
                for i in range(5):
                    sa = SARunner(problem=continuous_peaks_problem,
                                  experiment_name="TSP Simulated Annealing with Decay Schedule" + str(
                                      entry[0]) + "and Initial Temperature " + str(entry[1]) + "Run " + str(i),
                                  output_directory=out_dir,
                                  seed=np.random.randint(100),
                                  iteration_list=np.arange(1, 5000, 100),
                                  max_attempts=200,
                                  decay_list=[entry[0]],
                                  temperature_list=[entry[1]])

                    sa_run_stats, _ = sa.run()

                    sa_run_iterations = sa_run_stats['Iteration'].tolist()
                    runs.append(sa_run_stats['Fitness'].tolist())

                sa_run_average = np.mean(np.asarray(runs), axis=0)

                ax1.plot(sa_run_iterations, sa_run_average, linestyle=decay_names[entry[0]][1],
                         label=decay_names[entry[0]][0] + " Init Temp:" + str(entry[1]))

            # Put a legend to the right of the current axis
            box = ax1.get_position()
            ax1.set_position([box.x0, box.y0, box.width * 0.99, box.height])
            ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=6)
            ax1.grid()
            plt.tight_layout()
            filename = "decay_and_initial_temp.png"
            filename = os.path.join(out_dir, filename)
            plt.savefig(filename)

        runs = []
        times = []
        for i in range(5):

            sa = SARunner(problem=continuous_peaks_problem,
                          experiment_name="TSP Simulated Annealing after Param Tuning Decay Schedule",
                          output_directory=out_dir,
                          seed=np.random.randint(100),
                          iteration_list=np.arange(1, 5000, 100),
                          max_attempts=1000,
                          decay_list=[mlrose_hiive.GeomDecay],
                          temperature_list=[50])

            sa_run_stats, _ = sa.run()

            sa_run_iterations = sa_run_stats['Iteration'].tolist()

            times.append(sa_run_stats["Time"].iloc[-1] * 1000)
            runs.append(sa_run_stats['Fitness'].tolist())

        sa_run_average = np.mean(np.asarray(runs), axis=0)
        sa_run_average_time = np.mean(times)

    if run_ga:

        out_dir = TRAVELING_SALES_PERSON_PATH + GA_PATH

        if find_params:

            population_sizes = [150, 200, 250]

            pop_keys = {
                250: ('solid')
            }

            # first iteration revealed a pop size of 150 was sufficient.
            population_sizes = [250]
            mutation_rates = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
            #pop_keys = {
            #    150: ('solid')
            #}

            parameters = []
            for i in population_sizes:
                for j in mutation_rates:
                    parameters.append((i, j))

            title_dic = {'fontsize': 6, 'fontweight': 'bold'}
            fig, (ax1), = plt.subplots(1, 1, figsize=(3, 2))
            ax1.set_title("Genetic Algorithms with Various Mutation Rates", title_dic)
            ax1.set_ylabel("Average Fitness Score", title_dic)
            ax1.tick_params(axis="x", labelsize=6)
            ax1.tick_params(axis="y", labelsize=6)
            ax1.set_xlabel("Iteration", title_dic)

            for entry in parameters:

                runs = []
                for i in range(5):
                    ga = GARunner(problem=continuous_peaks_problem,
                                  experiment_name="TSP GA with Population" + str(
                                      entry[0]) + "and Mutation Split " + str(entry[1]),
                                  output_directory=out_dir,
                                  seed=np.random.randint(100),
                                  iteration_list=np.arange(1, 500, 25),
                                  max_attempts=150,
                                  population_sizes=[entry[0]],
                                  mutation_rates=[entry[1]])

                    ga_run_stats, _ = ga.run()

                    ga_run_iterations = ga_run_stats['Iteration'].tolist()
                    runs.append(ga_run_stats['Fitness'].tolist())

                ga_run_average = np.mean(np.asarray(runs), axis=0)

                ax1.plot(ga_run_iterations, ga_run_average, linestyle=pop_keys[entry[0]],
                         label="MutRate:" + str(entry[1]))

            # Put a legend to the right of the current axis
            box = ax1.get_position()
            ax1.legend(loc='best', fontsize=6)
            ax1.grid()
            plt.tight_layout()
            filename = "population_and_mutation_rates.png"
            filename = os.path.join(out_dir, filename)
            plt.savefig(filename)

        runs = []
        times = []
        for i in range(5):

            ga = GARunner(problem=continuous_peaks_problem,
                          experiment_name="CPP GA After Parameter Tuning" + "Run " + str(i),
                          output_directory=out_dir,
                          seed=np.random.randint(100),
                          iteration_list=np.arange(1, 600, 25),
                          max_attempts=150,
                          population_sizes=[250],
                          mutation_rates=[0.3])

            ga_run_stats, _ = ga.run()
            ga_run_iterations = ga_run_stats['Iteration'].tolist()
            times.append(ga_run_stats["Time"].iloc[-1] * 1000)
            runs.append(ga_run_stats['Fitness'].tolist())

        ga_run_average = np.mean(np.asarray(runs), axis=0)
        ga_run_average_time = np.mean(times)

    if run_mimic:

        out_dir = TRAVELING_SALES_PERSON_PATH + MIMIC_PATH

        if find_params:

            population_sizes = [200, 250, 300]

            # Second search, 200 provided the overall best.
            population_sizes = [250]
            keep_percent = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

            pop_keys = {
                250: ('solid'),
            }

            parameters = []
            for i in population_sizes:
                for j in keep_percent:
                    parameters.append((i, j))

            title_dic = {'fontsize': 6, 'fontweight': 'bold'}
            fig, (ax1), = plt.subplots(1, 1, figsize=(3, 2))
            ax1.set_title("Mimic with 250 Population Size and Various \n Keep Percentages", title_dic)
            ax1.set_ylabel("Fitness Score", title_dic)
            ax1.tick_params(axis="x", labelsize=6)
            ax1.tick_params(axis="y", labelsize=6)
            ax1.set_xlabel("Iteration", title_dic)

            for entry in parameters:

                runs = []
                for i in range(5):

                    mimic = MIMICRunner(problem=continuous_peaks_problem,
                                     experiment_name="TSP Mimic with Population" + str(
                                         entry[0]) + "and Keep Percent " + str(entry[1]),
                                     output_directory=out_dir,
                                     seed=np.random.randint(100),
                                     iteration_list=np.arange(1, 250, 25),
                                     population_sizes=[entry[0]],
                                     keep_percent_list=[entry[1]],
                                     max_attempts=5, generate_curves=True, use_fast_mimic=True)

                    df_run_stats, _ = mimic.run()

                    mimic_run_iterations = df_run_stats['Iteration'].tolist()
                    runs.append(df_run_stats['Fitness'].tolist())

                mimic_run_average = np.mean(np.asarray(runs), axis=0)

                ax1.plot(mimic_run_iterations, mimic_run_average, linestyle=pop_keys[entry[0]],
                         label="Keep %:" + str(entry[1]))

            # Put a legend to the right of the current axis
            box = ax1.get_position()
            ax1.set_position([box.x0, box.y0, box.width * 0.99, box.height])
            ax1.legend(loc='best', fontsize=6)
            ax1.grid()
            plt.tight_layout()
            filename = "population_and_keep_percent_rates.png"
            filename = os.path.join(out_dir, filename)
            plt.savefig(filename)

        runs = []
        times = []
        for i in range(5):

            mimic = MIMICRunner(problem=continuous_peaks_problem,
                                 experiment_name="CPP Mimic" + "Run " + str(i),
                                 output_directory=out_dir,
                                 seed=np.random.randint(100),
                                 iteration_list=np.arange(1, 150, 5),
                                 population_sizes=[250],
                                 keep_percent_list=[0.5],
                                 max_attempts=50, generate_curves=True, use_fast_mimic=True)

            mimic_run_stats, _ = mimic.run()
            mimic_run_iterations = mimic_run_stats['Iteration'].tolist()
            times.append(mimic_run_stats["Time"].iloc[-1] * 1000)
            runs.append(mimic_run_stats['Fitness'].tolist())

        mimic_run_average = np.mean(np.asarray(runs), axis=0)
        mimic_run_average_time = np.mean(times)

    title_dic = {'fontsize': 6, 'fontweight': 'bold'}
    fig, (fitness_cp, time_cp) = plt.subplots(1, 2, figsize=(6, 2))
    fitness_cp.set_title(
        "Fitness vs Iterations\n Continuous Peak Problem N = 80", title_dic)
    fitness_cp.set_ylabel("Average Fitness Score", title_dic)
    fitness_cp.tick_params(axis="x", labelsize=6)
    fitness_cp.tick_params(axis="y", labelsize=6)
    fitness_cp.set_xlabel("Iteration", title_dic)

    fitness_cp.plot(sa_run_iterations, sa_run_average, label="SA")
    fitness_cp.plot(ga_run_iterations, ga_run_average, label="GA")
    fitness_cp.plot(rhc_run_iterations, rhc_run_average_restarts, label="RHC")
    fitness_cp.plot(mimic_run_iterations, mimic_run_average, label="MIMIC")
    fitness_cp.plot(sa_run_iterations, np.ones((len(sa_run_iterations),1)) * 151, label="Global\nOptimum", linestyle='--', color='k')

    fitness_cp.legend(loc='best', fontsize=6)

    time_cp.set_xlabel("Time (ms)", title_dic)
    time_cp.set_title("Runtime \n Continuous Peaks Problem N = 80", title_dic)
    time_cp.set_ylabel("Algorithm", title_dic)
    time_cp.tick_params(axis="x", labelsize=6)
    time_cp.tick_params(axis="y", labelsize=6)

    x = ['RHC', 'SA', 'GA', 'MIMIC']
    x_pos = [i for i, _ in enumerate(x)]
    time_cp.set_yticks(x_pos)
    time_cp.set_yticklabels(x)
    time_cp.barh(x_pos, [rhc_run_average_time, sa_run_average_time, ga_run_average_time, mimic_run_average_time])

    time_cp.grid()
    fitness_cp.grid()
    plt.tight_layout()

    path = TRAVELING_SALES_PERSON_PATH
    filename = "fitness_vs_iteration.png"
    filename = os.path.join(path, filename)
    plt.savefig(filename)

    title_dic = {'fontsize': 6, 'fontweight': 'bold'}
    fig, (fitness_cp, time_cp) = plt.subplots(1, 2, figsize=(6, 2))
    fitness_cp.set_title(
        "Fitness vs Problem Size\n Continuous Peak Problem", title_dic)
    fitness_cp.set_ylabel("Average Fitness Score", title_dic)
    fitness_cp.tick_params(axis="x", labelsize=6)
    fitness_cp.tick_params(axis="y", labelsize=6)
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

    for size in sizes:

        continuous_peaks_problem = CustomContinuousPeaksGenerator.generate(1, size)

        restarts = []
        iterations = []
        seeds = [44,43,29]
        times = []
        for i in range(3):
            seed = seeds[i]

            # run RHC runner with param restarts.
            rhc = RHCRunner(problem=continuous_peaks_problem,
                            experiment_name="RHC CPP",
                            output_directory=out_dir,
                            seed=seed,
                            iteration_list=np.arange(1, 2200, 100),
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

            sa = SARunner(problem=continuous_peaks_problem,
                          experiment_name="TSP Simulated Annealing after Param Tuning Decay Schedule",
                          output_directory=out_dir,
                          seed=np.random.randint(100),
                          iteration_list=np.arange(1, 5000, 100),
                          max_attempts=1000,
                          decay_list=[mlrose_hiive.GeomDecay],
                          temperature_list=[1000])

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

            ga = GARunner(problem=continuous_peaks_problem,
                          experiment_name="CPP GA After Parameter Tuning",
                          output_directory=out_dir,
                          seed=np.random.randint(100),
                          iteration_list=np.arange(1, 500, 25),
                          max_attempts=150,
                          population_sizes=[150],
                          mutation_rates=[0.3])

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

            mimic = MIMICRunner(problem=continuous_peaks_problem,
                                 experiment_name="CPP Mimic",
                                 output_directory=out_dir,
                                 seed=np.random.randint(100),
                                 iteration_list=np.arange(1, 150, 5),
                                 population_sizes=[250],
                                 keep_percent_list=[0.5],
                                 max_attempts=50, generate_curves=True, use_fast_mimic=True)

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

    time_cp.set_ylabel("Time (ms)", title_dic)
    time_cp.set_title("Runtime vs Problem Size \n Continuous Peaks Problem", title_dic)
    time_cp.set_xlabel("Problem Size(N)", title_dic)
    time_cp.tick_params(axis="x", labelsize=6)
    time_cp.tick_params(axis="y", labelsize=6)

    time_cp.plot(sizes, sa_average_time, label="SA")
    time_cp.plot(sizes, ga_average_time, label="GA")
    time_cp.plot(sizes, rhc_average_time, label="RHC")
    time_cp.plot(sizes, mimic_average_time, label="MIMIC")
    time_cp.legend(loc='best', fontsize=6)

    time_cp.grid()
    fitness_cp.grid()
    plt.tight_layout()

    path = TRAVELING_SALES_PERSON_PATH
    filename = "fitness_vs_iteration_various_problem_sizes.png"
    filename = os.path.join(path, filename)
    plt.savefig(filename)