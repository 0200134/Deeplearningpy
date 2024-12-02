using System;
using System.Linq;
using System.Collections.Generic;

namespace GeneticAlgorithmExample
{
    class Program
    {
        private static Random random = new Random();
        private const int PopulationSize = 100;
        private const int ChromosomeLength = 10;
        private const int MaxGenerations = 1000;
        private const double CrossoverRate = 0.8;
        private const double MutationRate = 0.01;

        static void Main(string[] args)
        {
            List<int[]> population = InitializePopulation();

            for (int generation = 0; generation < MaxGenerations; generation++)
            {
                List<int[]> newPopulation = new List<int[]>();

                for (int i = 0; i < PopulationSize / 2; i++)
                {
                    int[] parent1 = SelectParent(population);
                    int[] parent2 = SelectParent(population);

                    int[] child1, child2;
                    Crossover(parent1, parent2, out child1, out child2);

                    Mutate(child1);
                    Mutate(child2);

                    newPopulation.Add(child1);
                    newPopulation.Add(child2);
                }

                population = newPopulation;
                int[] bestSolution = population.OrderByDescending(Fitness).First();
                Console.WriteLine($"Generation {generation}: Best Fitness = {Fitness(bestSolution)}");
            }
        }

        private static List<int[]> InitializePopulation()
        {
            var population = new List<int[]>();
            for (int i = 0; i < PopulationSize; i++)
            {
                var chromosome = new int[ChromosomeLength];
                for (int j = 0; j < ChromosomeLength; j++)
                {
                    chromosome[j] = random.Next(0, 2);
                }
                population.Add(chromosome);
            }
            return population;
        }

        private static int[] SelectParent(List<int[]> population)
        {
            return population[random.Next(0, PopulationSize)];
        }

        private static void Crossover(int[] parent1, int[] parent2, out int[] child1, out int[] child2)
        {
            child1 = new int[ChromosomeLength];
            child2 = new int[ChromosomeLength];
            if (random.NextDouble() < CrossoverRate)
            {
                int crossoverPoint = random.Next(1, ChromosomeLength - 1);
                for (int i = 0; i < crossoverPoint; i++)
                {
                    child1[i] = parent1[i];
                    child2[i] = parent1[i];
                }
                for (int i = crossoverPoint; i < ChromosomeLength; i++)
                {
                    child1[i] = parent2[i];
                    child2[i] = parent1[i];
                }
            }
            else
            {
                Array.Copy(parent1, child1, ChromosomeLength);
                Array.Copy(parent2, child2, ChromosomeLength);
            }
        }

        private static void Mutate(int[] chromosome)
        {
            for (int i = 0; i < ChromosomeLength; i++)
            {
                if (random.NextDouble() < MutationRate)
                {
                    chromosome[i] = 1 - chromosome[i];
                }
            }
        }

        private static int Fitness(int[] chromosome)
        {
            return chromosome.Sum();
        }
    }
}
