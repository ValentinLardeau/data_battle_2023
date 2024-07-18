from deep_learning_power_measure.power_measure import experiment, parsers, rapl, rapl_power
import os

output_folder = "conso/"


def project():
    os.system("python read-pdf.py  -f NO_Quad_15/15_9-9/15_9-9__WELL__15-09-09_PB-706-0109.pdf -a False ")


def find_legend():
    os.system("python find_legend.py  -i output1/NO_Quad_15/15_9-9/15_9-9__WELL__15-09-09_PB-706-0109/images/195.jpg ")


def find_lithological_column():
    os.system(
        "python find_lithological_column.py  -i 236.jpg ")


def IA():
    os.system(
        "python Read_colonne.py")


def enregistrementConso(output_folder, f, *args, **kwargs):
    driver = parsers.JsonParser(output_folder)
    exp = experiment.Experiment(driver)
    p, q = exp.measure_yourself(period=2)
    f(*args, **kwargs)
    q.put(experiment.STOP_MESSAGE)


def affichageConso(output_folder):
    driver = parsers.JsonParser(output_folder)
    exp_result = experiment.ExpResults(driver)
    exp_result.print()


enregistrementConso(output_folder, find_legend)
affichageConso(output_folder)

# find_legend

# ============================================ EXPERIMENT SUMMARY ============================================
# Experiment duration:  108.84731578826904 seconds
# ENERGY CONSUMPTION:
# on the cpu

# RAM consumption not available. Your usage was  3.0GiB with an overhead of 2.8GiB
# Total CPU consumption: 1186.525950108312 joules, your experiment consumption:  749.5914946041186 joules
# total intel power:  1639.9504618482033 joules
# total psys power:  2511.0190110082663 joules

enregistrementConso(output_folder, find_lithological_column)
affichageConso(output_folder)

# find_lithological_column

# ============================================ EXPERIMENT SUMMARY ============================================
# Experiment duration:  52.35184407234192 seconds
# ENERGY CONSUMPTION:
# on the cpu

# RAM consumption not available. Your usage was  2.7GiB with an overhead of 2.4GiB
# Total CPU consumption: 558.7283924348047 joules, your experiment consumption:  319.07518427451237 joules
# total intel power:  771.6018092227271 joules
# total psys power:  1189.5968983078976 joules


enregistrementConso(output_folder, IA)
affichageConso(output_folder)

# Read_colonne

# ============================================ EXPERIMENT SUMMARY ============================================
# Experiment duration:  4.108924865722656 seconds
# ENERGY CONSUMPTION:
# on the cpu

# RAM consumption not available. Your usage was  427.5MiB with an overhead of 207.8MiB
# Total CPU consumption: 50.36353731894181 joules, your experiment consumption:  16.466691483715113 joules
# total intel power:  67.13192795166589 joules
# total psys power:  100.73777433494747 joules

enregistrementConso(output_folder, project)
affichageConso(output_folder)


# The entire project consumption:

# ============================================ EXPERIMENT SUMMARY ============================================
# Experiment duration:  228.37450098991394 seconds
# ENERGY CONSUMPTION:
# on the cpu

# RAM consumption not available. Your usage was  3.1GiB with an overhead of 2.9GiB
# Total CPU consumption: 2261.410923167332 joules, your experiment consumption:  1249.8909609026912 joules
# total intel power:  3391.2704331540763 joules
# total psys power:  5348.205584792256 joules
