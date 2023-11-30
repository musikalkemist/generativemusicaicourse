import random
from dataclasses import dataclass

import music21


@dataclass(frozen=True)
class MelodyData:
    """
    A data class representing the data of a melody.

    This class encapsulates the details of a melody including its notes, total
    duration, and the number of bars. The notes are represented as a list of
    tuples, with each tuple containing a pitch and its duration. The total
    duration and the number of bars are computed based on the notes provided.

    Attributes:
        notes (list of tuples): List of tuples representing the melody's notes.
            Each tuple is in the format (pitch, duration).
        duration (int): Total duration of the melody, computed from notes.
        number_of_bars (int): Total number of bars in the melody, computed from
            the duration assuming a 4/4 time signature.

    Methods:
        __post_init__: A method called after the data class initialization to
            calculate and set the duration and number of bars based on the
            provided notes.
    """

    notes: list
    duration: int = None  # Computed attribute
    number_of_bars: int = None  # Computed attribute

    def __post_init__(self):
        object.__setattr__(
            self, "duration", sum(duration for _, duration in self.notes)
        )
        object.__setattr__(self, "number_of_bars", self.duration // 4)


class GeneticMelodyHarmonizer:
    """
    Generates chord accompaniments for a given melody using a genetic algorithm.
    It evolves a population of chord sequences to find one that best fits the
    melody based on a fitness function.

    Attributes:
        melody_data (MusicData): Data containing melody information.
        chords (list): Available chords for generating sequences.
        population_size (int): Size of the chord sequence population.
        mutation_rate (float): Probability of mutation in the genetic algorithm.
        fitness_evaluator (FitnessEvaluator): Instance used to assess fitness.
    """

    def __init__(
        self,
        melody_data,
        chords,
        population_size,
        mutation_rate,
        fitness_evaluator,
    ):
        """
        Initializes the generator with melody data, chords, population size,
        mutation rate, and a fitness evaluator.

        Parameters:
            melody_data (MusicData): Melody information.
            chords (list): Available chords.
            population_size (int): Size of population in the algorithm.
            mutation_rate (float): Mutation probability per chord.
            fitness_evaluator (FitnessEvaluator): Evaluator for chord fitness.
        """
        self.melody_data = melody_data
        self.chords = chords
        self.mutation_rate = mutation_rate
        self.population_size = population_size
        self.fitness_evaluator = fitness_evaluator
        self._population = []

    def generate(self, generations=1000):
        """
        Generates a chord sequence that harmonizes a melody using a genetic
        algorithm.

        Parameters:
            generations (int): Number of generations for evolution.

        Returns:
            best_chord_sequence (list): Harmonization with the highest fitness
                found in the last generation.
        """
        self._population = self._initialise_population()
        for _ in range(generations):
            parents = self._select_parents()
            new_population = self._create_new_population(parents)
            self._population = new_population
        best_chord_sequence = (
            self.fitness_evaluator.get_chord_sequence_with_highest_fitness(
                self._population
            )
        )
        return best_chord_sequence

    def _initialise_population(self):
        """
        Initializes population with random chord sequences.

        Returns:
            list: List of randomly generated chord sequences.
        """
        return [
            self._generate_random_chord_sequence()
            for _ in range(self.population_size)
        ]

    def _generate_random_chord_sequence(self):
        """
        Generate a random chord sequence with as many chords as the numbers
        of bars in the melody.

        Returns:
            list: List of randomly generated chords.
        """
        return [
            random.choice(self.chords)
            for _ in range(self.melody_data.number_of_bars)
        ]

    def _select_parents(self):
        """
        Selects parent sequences for breeding based on fitness.

        Returns:
            list: Selected parent chord sequences.
        """
        fitness_values = [
            self.fitness_evaluator.evaluate(seq) for seq in self._population
        ]
        return random.choices(
            self._population, weights=fitness_values, k=self.population_size
        )

    def _create_new_population(self, parents):
        """
        Generates a new population of chord sequences from the provided parents.

        This method creates a new generation of chord sequences using crossover
        and mutation operations. For each pair of parent chord sequences,
        it generates two children. Each child is the result of a crossover
        operation between the pair of parents, followed by a potential
        mutation. The new population is formed by collecting all these
        children.

        The method ensures that the new population size is equal to the
        predefined population size of the generator. It processes parents in
        pairs, and for each pair, two children are generated.

        Parameters:
            parents (list): A list of parent chord sequences from which to
                generate the new population.

        Returns:
            list: A new population of chord sequences, generated from the
                parents.

        Note:
            This method assumes an even population size and that the number of
            parents is equal to the predefined population size.
        """
        new_population = []
        for i in range(0, self.population_size, 2):
            child1, child2 = self._crossover(
                parents[i], parents[i + 1]
            ), self._crossover(parents[i + 1], parents[i])
            child1 = self._mutate(child1)
            child2 = self._mutate(child2)
            new_population.extend([child1, child2])
        return new_population

    def _crossover(self, parent1, parent2):
        """
        Combines two parent sequences into a new child sequence using one-point
        crossover.

        Parameters:
            parent1 (list): First parent chord sequence.
            parent2 (list): Second parent chord sequence.

        Returns:
            list: Resulting child chord sequence.
        """
        cut_index = random.randint(1, len(parent1) - 1)
        return parent1[:cut_index] + parent2[cut_index:]

    def _mutate(self, chord_sequence):
        """
        Mutates a chord in the sequence based on mutation rate.

        Parameters:
            chord_sequence (list): Chord sequence to mutate.

        Returns:
            list: Mutated chord sequence.
        """
        if random.random() < self.mutation_rate:
            mutation_index = random.randint(0, len(chord_sequence) - 1)
            chord_sequence[mutation_index] = random.choice(self.chords)
        return chord_sequence


class FitnessEvaluator:
    """
    Evaluates the fitness of a chord sequence based on various musical criteria.

    Attributes:
        melody (list): List of tuples representing notes as (pitch, duration).
        chords (dict): Dictionary of chords with their corresponding notes.
        weights (dict): Weights for different fitness evaluation functions.
        preferred_transitions (dict): Preferred chord transitions.
    """

    def __init__(
        self, melody_data, chord_mappings, weights, preferred_transitions
    ):
        """
        Initialize the FitnessEvaluator with melody, chords, weights, and
        preferred transitions.

        Parameters:
            melody_data (MelodyData): Melody information.
            chord_mappings (dict): Available chords mapped to their notes.
            weights (dict): Weights for each fitness evaluation function.
            preferred_transitions (dict): Preferred chord transitions.
        """
        self.melody_data = melody_data
        self.chord_mappings = chord_mappings
        self.weights = weights
        self.preferred_transitions = preferred_transitions

    def get_chord_sequence_with_highest_fitness(self, chord_sequences):
        """
        Returns the chord sequence with the highest fitness score.

        Parameters:
            chord_sequences (list): List of chord sequences to evaluate.

        Returns:
            list: Chord sequence with the highest fitness score.
        """
        return max(chord_sequences, key=self.evaluate)

    def evaluate(self, chord_sequence):
        """
        Evaluate the fitness of a given chord sequence.

        Parameters:
            chord_sequence (list): The chord sequence to evaluate.

        Returns:
            float: The overall fitness score of the chord sequence.
        """
        return sum(
            self.weights[func] * getattr(self, f"_{func}")(chord_sequence)
            for func in self.weights
        )

    def _chord_melody_congruence(self, chord_sequence):
        """
        Calculates the congruence between the chord sequence and the melody.
        This function assesses how well each chord in the sequence aligns
        with the corresponding segment of the melody. The alignment is
        measured by checking if the notes in the melody are present in the
        chords being played at the same time, rewarding sequences where the
        melody notes fit well with the chords.

        Parameters:
            chord_sequence (list): A list of chords to be evaluated against the
                melody.

        Returns:
            float: A score representing the degree of congruence between the
                chord sequence and the melody, normalized by the melody's
                duration.
        """
        score, melody_index = 0, 0
        for chord in chord_sequence:
            bar_duration = 0
            while bar_duration < 4 and melody_index < len(
                self.melody_data.notes
            ):
                pitch, duration = self.melody_data.notes[melody_index]
                if pitch[0] in self.chord_mappings[chord]:
                    score += duration
                bar_duration += duration
                melody_index += 1
        return score / self.melody_data.duration

    def _chord_variety(self, chord_sequence):
        """
        Evaluates the diversity of chords used in the sequence. This function
        calculates a score based on the number of unique chords present in the
        sequence compared to the total available chords. Higher variety in the
        chord sequence results in a higher score, promoting musical
        complexity and interest.

        Parameters:
            chord_sequence (list): The chord sequence to evaluate.

        Returns:
            float: A normalized score representing the variety of chords in the
                sequence relative to the total number of available chords.
        """
        unique_chords = len(set(chord_sequence))
        total_chords = len(self.chord_mappings)
        return unique_chords / total_chords

    def _harmonic_flow(self, chord_sequence):
        """
        Assesses the harmonic flow of the chord sequence by examining the
        transitions between successive chords. This function scores the
        sequence based on how frequently the chord transitions align with
        predefined preferred transitions. Smooth and musically pleasant
        transitions result in a higher score.

        Parameters:
            chord_sequence (list): The chord sequence to evaluate.

        Returns:
            float: A normalized score based on the frequency of preferred chord
                transitions in the sequence.
        """
        score = 0
        for i in range(len(chord_sequence) - 1):
            next_chord = chord_sequence[i + 1]
            if next_chord in self.preferred_transitions[chord_sequence[i]]:
                score += 1
        return score / (len(chord_sequence) - 1)

    def _functional_harmony(self, chord_sequence):
        """
        Evaluates the chord sequence based on principles of functional harmony.
        This function checks for the presence of key harmonic functions such as
        the tonic at the beginning and end of the sequence and the presence of
        subdominant and dominant chords. Adherence to these harmonic
        conventions is rewarded with a higher score.

        Parameters:
            chord_sequence (list): The chord sequence to evaluate.

        Returns:
            float: A score representing the extent to which the sequence
                adheres to traditional functional harmony, normalized by
                the number of checks performed.
        """
        score = 0
        if chord_sequence[0] in ["C", "Am"]:
            score += 1
        if chord_sequence[-1] in ["C"]:
            score += 1
        if "F" in chord_sequence and "G" in chord_sequence:
            score += 1
        return score / 3


def create_score(melody, chord_sequence, chord_mappings):
    """
    Create a music21 score with a given melody and chord sequence.

    Args:
        melody (list): A list of tuples representing notes in the format
            (note_name, duration).
        chord_sequence (list): A list of chord names.

    Returns:
        music21.stream.Score: A music score containing the melody and chord
            sequence.
    """
    # Create a Score object
    score = music21.stream.Score()

    # Create the melody part and add notes to it
    melody_part = music21.stream.Part()
    for note_name, duration in melody:
        melody_note = music21.note.Note(note_name, quarterLength=duration)
        melody_part.append(melody_note)

    # Create the chord part and add chords to it
    chord_part = music21.stream.Part()
    current_duration = 0  # Track the duration for chord placement

    for chord_name in chord_sequence:
        # Translate chord names to note lists
        chord_notes_list = chord_mappings.get(chord_name, [])
        # Create a music21 chord
        chord_notes = music21.chord.Chord(
            chord_notes_list, quarterLength=4
        )  # Assuming 4/4 time signature
        chord_notes.offset = current_duration
        chord_part.append(chord_notes)
        current_duration += 4  # Increase by 4 beats

    # Append parts to the score
    score.append(melody_part)
    score.append(chord_part)

    return score


def main():

    twinkle_twinkle_melody = [
        ("C5", 1),
        ("C5", 1),
        ("G5", 1),
        ("G5", 1),
        ("A5", 1),
        ("A5", 1),
        ("G5", 2),  # Twinkle, twinkle, little star,
        ("F5", 1),
        ("F5", 1),
        ("E5", 1),
        ("E5", 1),
        ("D5", 1),
        ("D5", 1),
        ("C5", 2),  # How I wonder what you are!
        ("G5", 1),
        ("G5", 1),
        ("F5", 1),
        ("F5", 1),
        ("E5", 1),
        ("E5", 1),
        ("D5", 2),  # Up above the world so high,
        ("G5", 1),
        ("G5", 1),
        ("F5", 1),
        ("F5", 1),
        ("E5", 1),
        ("E5", 1),
        ("D5", 2),  # Like a diamond in the sky.
        ("C5", 1),
        ("C5", 1),
        ("G5", 1),
        ("G5", 1),
        ("A5", 1),
        ("A5", 1),
        ("G5", 2),  # Twinkle, twinkle, little star,
        ("F5", 1),
        ("F5", 1),
        ("E5", 1),
        ("E5", 1),
        ("D5", 1),
        ("D5", 1),
        ("C5", 2)  # How I wonder what you are!
    ]
    weights = {
        "chord_melody_congruence": 0.4,
        "chord_variety": 0.1,
        "harmonic_flow": 0.3,
        "functional_harmony": 0.2
    }
    chord_mappings = {
        "C": ["C", "E", "G"],
        "Dm": ["D", "F", "A"],
        "Em": ["E", "G", "B"],
        "F": ["F", "A", "C"],
        "G": ["G", "B", "D"],
        "Am": ["A", "C", "E"],
        "Bdim": ["B", "D", "F"]
    }
    preferred_transitions = {
        "C": ["G", "Am", "F"],
        "Dm": ["G", "Am"],
        "Em": ["Am", "F", "C"],
        "F": ["C", "G"],
        "G": ["Am", "C"],
        "Am": ["Dm", "Em", "F", "C"],
        "Bdim": ["F", "Am"]
    }

    # Instantiate objects for generating harmonization
    melody_data = MelodyData(twinkle_twinkle_melody)
    fitness_evaluator = FitnessEvaluator(
        melody_data=melody_data,
        weights=weights,
        chord_mappings=chord_mappings,
        preferred_transitions=preferred_transitions,
    )
    harmonizer = GeneticMelodyHarmonizer(
        melody_data=melody_data,
        chords=list(chord_mappings.keys()),
        population_size=100,
        mutation_rate=0.05,
        fitness_evaluator=fitness_evaluator,
    )

    # Generate chords with genetic algorithm
    generated_chords = harmonizer.generate(generations=1000)

    # Render to music21 score and show it
    music21_score = create_score(
        twinkle_twinkle_melody, generated_chords, chord_mappings
    )
    music21_score.show()


if __name__ == "__main__":
    main()
