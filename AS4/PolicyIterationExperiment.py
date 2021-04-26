from hiive.mdptoolbox.mdp import ValueIteration
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import FormatStrFormatter
import pickle
import numpy as np

class ValueIterationExperiment():

    P_INDEX = 0
    R_INDEX = 1
    HYPER_PARAMS_INDEX = 2
    PROBLEM_SIZE_INDEX = 3

    def __init__(self, problems, output_dir, problem_name):

        self.problems = problems
        self.output_dir = output_dir
        self.policy = None
        self.problem_name = problem_name
        self.hyper_params = {}
        self.policies = {}

    def tune_hyper_parameter(self, hyper_param_name, values):

        for problem in self.problems:

            rewards_list = []
            hyper_params = problem[self.HYPER_PARAMS_INDEX]

            title = "Tuning " + hyper_param_name + " for Value Iteration \n" + self.problem_name + " Size " + problem[self.PROBLEM_SIZE_INDEX]
            title_dic = {'fontsize': 7, 'fontweight': 'bold'}
            fig, (ax1) = plt.subplots(1, 1, figsize=(3, 2))
            ax1.set_xlabel("Iterations (Log Scale)", title_dic)
            ax1.set_xscale('log')
            ax1.set_title(title , title_dic)
            ax1.set_ylabel("Max Utility", title_dic)
            ax1.tick_params(axis="x", labelsize=7)
            ax1.tick_params(axis="y", labelsize=7)
            ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            ax1.grid()
            plt.tight_layout()
            plt.grid()

            if "max_iter" in hyper_params:
                iters = hyper_params["max_iter"]
            else:
                iters = 10000

            if "epsilon" in hyper_params:
                epsilon = hyper_params["epsilon"]
            else:
                epsilon = 0.0000001

            if 'gamma' in hyper_params:
                gamma = hyper_params["gamma"]
            else:
                gamma = 0.99

            for value in values:

                if hyper_param_name == 'gamma':
                    vi = ValueIteration(problem[self.P_INDEX], problem[self.R_INDEX], gamma=value, max_iter=iters, epsilon=epsilon)
                elif hyper_param_name == 'max_iter':
                    vi = ValueIteration(problem[self.P_INDEX], problem[self.R_INDEX], gamma=gamma, max_iter=value, epsilon=epsilon)
                elif hyper_param_name == 'epsilon':
                    vi = ValueIteration(problem[self.P_INDEX], problem[self.R_INDEX], gamma=gamma, max_iter=iters, epsilon=value)

                stats = vi.run()

                rewards = [dict["Max V"] for dict in stats]
                print("Max Reward with " + hyper_param_name + " value " + str(value) +  " " + str(rewards[-1]))

                ax1.plot(rewards, label=str(value))

                #rewards_list.append(rewards)

            ax1.legend(loc='lower right', fontsize=6, ncol=2)
            path = os.path.join(self.output_dir)
            filename = title + ".png"
            filename = os.path.join(path, filename)
            plt.savefig(filename)
            plt.close()

    def plot_convergence(self):

        title = "Convergence for Value Iteration \n" + self.problem_name
        title_dic = {'fontsize': 7, 'fontweight': 'bold'}
        fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(5, 2))
        ax1.set_xlabel("Iterations", title_dic)
        ax1.set_title(title, title_dic)
        ax1.set_ylabel("Delta Utility(Log Scale)", title_dic)
        ax1.tick_params(axis="x", labelsize=7)
        ax1.tick_params(axis="y", labelsize=7)
        ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax1.grid()
        ax1.set_yscale('log')

        title1 = "Cumulative Time (ms) vs Iterations \n" + self.problem_name
        ax2.set_xlabel("Iterations", title_dic)
        ax2.set_title(title1, title_dic)
        ax2.set_ylabel("Time(ms)", title_dic)
        ax2.tick_params(axis="x", labelsize=7)
        ax2.tick_params(axis="y", labelsize=7)
        ax2.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        ax2.grid()
        plt.tight_layout()
        plt.grid()


        deltaTime = None
        for problem in self.problems:

            hyper_params = problem[self.HYPER_PARAMS_INDEX]
            vi = ValueIteration(problem[self.P_INDEX], problem[self.R_INDEX], gamma=hyper_params["gamma"], max_iter=hyper_params["max_iter"], epsilon=hyper_params["epsilon"])

            stats = vi.run()
            time = [dict["Time"] for dict in stats]
            time = np.array([x - time[i - 1] for i, x in enumerate(time)][1:])
            time = time * 1000

            self.policies[problem[self.PROBLEM_SIZE_INDEX]] = list(vi.policy)

            path = os.path.join(self.output_dir)
            filename = "valueiterationrunstats"
            filename = os.path.join(path, filename)

            outfile = open(filename, 'wb')
            pickle.dump(stats, outfile)
            outfile.close()

            delta = [dict["Error"] for dict in stats]
            ax1.plot(delta, label="Size:" + str(problem[self.PROBLEM_SIZE_INDEX]))
            ax2.plot(time, label="Size:" + str(problem[self.PROBLEM_SIZE_INDEX]))

                #rewards_list.append(rewards)

        ax1.legend(loc='best', fontsize=6)
        ax2.legend(loc='best', fontsize=6)
        path = os.path.join(self.output_dir)
        filename = title + ".png"
        filename = os.path.join(path, filename)
        plt.savefig(filename)
        plt.close()

    def plot_time_to_solution_for_varying_sizes(self):
        return



