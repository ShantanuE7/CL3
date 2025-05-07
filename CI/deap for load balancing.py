import random
from deap import base, creator, tools
import numpy as np

# ---- Problem Definition ----
jobs = {
    0: [(0, 3), (1, 2), (2, 2)],
    1: [(0, 2), (2, 1), (1, 4)],
    2: [(1, 4), (2, 3)]
}

num_jobs = len(jobs)
num_machines = 3
num_operations = sum(len(ops) for ops in jobs.values())

# Flatten job operations for easier indexing
job_operations = []
for job_id, ops in jobs.items():
    for op in ops:
        job_operations.append((job_id, op))

# Create individual by encoding job IDs (sequence matters)
job_sequence = []
for job_id, ops in jobs.items():
    job_sequence += [job_id] * len(ops)

# ---- DEAP Setup ----
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # Minimize makespan
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

# ✅ Add this function ABOVE the toolbox.register line
def create_individual():
    seq = job_sequence.copy()
    random.shuffle(seq)
    return seq

# Register the fixed individual generator
toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def decode_schedule(individual):
    job_counters = {j: 0 for j in jobs}
    machine_available = [0] * num_machines
    job_end_time = [0] * num_jobs

    schedule = []
    for job_id in individual:
        op_idx = job_counters[job_id]

        # SAFETY CHECK: skip if job has no more operations
        if op_idx >= len(jobs[job_id]):
            continue  # skip invalid extra job occurrence

        machine, duration = jobs[job_id][op_idx]

        start = max(machine_available[machine], job_end_time[job_id])
        end = start + duration

        machine_available[machine] = end
        job_end_time[job_id] = end

        job_counters[job_id] += 1
        schedule.append((job_id, op_idx, machine, start, end))

    makespan = max([end for _, _, _, _, end in schedule])
    return makespan, schedule

def eval_schedule(individual):
    makespan, _ = decode_schedule(individual)
    return (makespan,)

toolbox.register("evaluate", eval_schedule)
toolbox.register("mate", tools.cxPartialyMatched)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

# ---- GA Parameters ----
POP_SIZE = 50
N_GEN = 100
CXPB = 0.8
MUTPB = 0.2

population = toolbox.population(n=POP_SIZE)

# ---- Evolution ----
for gen in range(N_GEN):
    offspring = toolbox.select(population, len(population))
    offspring = list(map(toolbox.clone, offspring))

    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < CXPB:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

    for mutant in offspring:
        if random.random() < MUTPB:
            toolbox.mutate(mutant)
            del mutant.fitness.values

    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    population[:] = offspring

    best = tools.selBest(population, 1)[0]
    print(f"Gen {gen+1}, Best Makespan: {best.fitness.values[0]}")

# ---- Final Schedule ----
best = tools.selBest(population, 1)[0]
makespan, final_schedule = decode_schedule(best)

print("\n Final Best Schedule:")
for job_id, op_idx, machine, start, end in final_schedule:
    print(f"Job {job_id}, Op {op_idx}, Machine {machine}, Start {start}, End {end}")

print(f"\n Best Makespan: {makespan}")
'''
Gen 100, Best Makespan: 7.0

 Final Best Schedule:
Job 2, Op 0, Machine 1, Start 0, End 4
Job 2, Op 1, Machine 2, Start 4, End 7
Job 0, Op 0, Machine 0, Start 0, End 3
Job 1, Op 0, Machine 0, Start 3, End 5

 Best Makespan: 7
'''