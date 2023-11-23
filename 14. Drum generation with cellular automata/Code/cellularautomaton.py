import random
from enum import Enum

import numpy as np
from music21 import instrument, metadata, note, stream


class DrumInstruments(Enum):
    KICK = 0
    SNARE = 1
    HIHAT = 2


class DrumStates(Enum):
    OFF = 0
    ON = 1


class CellularAutomatonDrumGenerator:
    """
    Generates drum patterns using a cellular automaton.

    This class simulates a 2D cellular automaton where each cell represents a
    drum sound (kick, snare, or hi-hat) at a given time step. The state of
    each cell evolves based on predefined rules, resulting in a rhythmic
    drum pattern.

    Attributes:
        pattern_length (int): Length of the drum pattern in beats.
        state (np.ndarray): Current state of the drum pattern.
    """

    HIHAT_ON_PROBABILITY = 0.7
    MUTATION_PROBABILITY = 0.1

    def __init__(self, pattern_length):
        """
        Initializes the CellularAutomatonDrumGenerator with a specified pattern
        length.

        Parameters:
            pattern_length (int): The length of the drum pattern in beats.
        """
        self.pattern_length = pattern_length
        self.state = self._initialize_state(pattern_length)
        self._rules = {
            "syncopation_resolution": self._apply_syncopation_resolution_rule,
            "filling_gaps": self._apply_filling_gaps_rule,
            "accenting": self._apply_accenting_rule,
            "mutation": self._apply_mutation_rule,
        }

    def step(self):
        """
        Advances the drum pattern by one time step by applying the defined
        rules.
        """
        new_state = self.state.copy()
        for position in range(self.pattern_length):
            new_state = self._apply_rules(position, new_state)
        self.state = new_state

    def _initialize_state(self, pattern_length):
        """
        Randomly initializes the state of the drum pattern.

        Parameters:
            pattern_length (int): The length of the drum pattern in beats.

        Returns:
            np.ndarray: The initial state array.
        """
        number_of_instruments = len(DrumInstruments)
        return np.random.choice(
            [DrumStates.OFF.value, DrumStates.ON.value],
            size=(number_of_instruments, pattern_length),
        )

    def _apply_rules(self, position, new_state):
        """
        Applies the set of rules to the drum pattern at a given position.

        Parameters:
            position (int): The current position in the drum pattern.
            new_state (np.ndarray): The state array being modified.

        Returns:
            np.ndarray: The updated state array after applying the rules.
        """
        for rule in self._rules.values():
            new_state = rule(position, new_state)
        return new_state

    def _apply_syncopation_resolution_rule(self, position, new_state):
        """
        Applies the syncopation resolution rule at a given position.

        Rule: If a hi-hat is ON and the kick is OFF at a position,
        turn ON the kick in the next position.

        Parameters:
            position (int): The current position in the drum pattern.
            new_state (np.ndarray): The state array being modified.

        Returns:
            np.ndarray: The updated state array.
        """
        next_position = self._get_next_position(position)

        if (
            self.state[DrumInstruments.HIHAT.value][position]
            == DrumStates.ON.value
            and self.state[DrumInstruments.KICK.value][position]
            == DrumStates.OFF.value
        ):
            new_state[DrumInstruments.KICK.value][
                next_position
            ] = DrumStates.ON.value

        return new_state

    def _get_next_position(self, position):
        """
        Determines the next position in a circular sequence.

        Given the current position within a circular sequence (such as a drum
        pattern), this method calculates the index of the next position. If
        the current position is the last in the sequence, it wraps around to
        the first position.

        Parameters:
            position (int): The current position in the sequence.

        Returns:
            int: The index of the next position in the sequence.
        """
        return (position + 1) % self.pattern_length

    def _apply_filling_gaps_rule(self, position, new_state):
        """
        Applies the filling gaps rule at a given position.

        Rule: If a kick is OFF in two consecutive positions, turn ON the snare
        in the next position. Similarly, if a snare is OFF in two
        consecutive positions, turn ON the kick in the next position.

        Parameters:
            position (int): The current position in the drum pattern.
            new_state (np.ndarray): The state array being modified.

        Returns:
            np.ndarray: The updated state array.
        """
        previous_position = self._get_previous_position(position)
        next_position = self._get_next_position(position)

        if (
            self.state[DrumInstruments.KICK.value][previous_position]
            == DrumStates.OFF.value
            and self.state[DrumInstruments.KICK.value][position]
            == DrumStates.OFF.value
        ):
            new_state[DrumInstruments.SNARE.value][
                next_position
            ] = DrumStates.ON.value

        if (
            self.state[DrumInstruments.SNARE.value][previous_position]
            == DrumStates.OFF.value
            and self.state[DrumInstruments.SNARE.value][position]
            == DrumStates.OFF.value
        ):
            new_state[DrumInstruments.KICK.value][
                next_position
            ] = DrumStates.ON.value

        return new_state

    def _get_previous_position(self, position):
        """
        Determines the previous position in a circular sequence.

        Given the current position within a circular sequence (like a drum
        pattern), this method calculates the index of the previous position.
        If the current position is the first in the sequence (position 0),
        it wraps around to the last position.

        Parameters:
            position (int): The current position in the sequence.

        Returns:
            int: The index of the previous position in the sequence.
        """
        if position > 0:
            return position - 1
        return self.pattern_length - 1

    def _apply_accenting_rule(self, position, new_state):
        """
        Applies the accenting rule at a given position.

        Rule: If both kick and snare are ON at a position, turn ON the hi-hat
        at the same position with a probability determined by
        HIHAT_ON_PROBABILITY.

        Parameters:
            position (int): The current position in the drum pattern.
            new_state (np.ndarray): The state array being modified.

        Returns:
            np.ndarray: The updated state array.
        """
        if (
            self.state[DrumInstruments.KICK.value][position]
            == DrumStates.ON.value
            and self.state[DrumInstruments.SNARE.value][position]
            == DrumStates.ON.value
        ):
            if random.random() < self.HIHAT_ON_PROBABILITY:
                new_hihat_state = DrumStates.ON.value
            else:
                new_hihat_state = DrumStates.OFF.value
            new_state[DrumInstruments.HIHAT.value][position] = new_hihat_state

        return new_state

    def _apply_mutation_rule(self, position, new_state):
        """
        Applies a mutation to a random instrument at a given position.

        Rule: At each position, with a small probability (MUTATION_PROBABILITY),
        randomly toggle the state (ON/OFF) of one of the instruments (kick,
        snare, hi-hat).

        Parameters:
            position (int): The current position in the drum pattern.
            new_state (np.ndarray): The state array being modified.

        Returns:
            np.ndarray: The updated state array.
        """
        if random.random() < self.MUTATION_PROBABILITY:
            instrument_choice = random.choice(
                [
                    DrumInstruments.KICK,
                    DrumInstruments.SNARE,
                    DrumInstruments.HIHAT,
                ]
            )
            new_instrument_state = random.choice(
                [DrumStates.ON.value, DrumStates.OFF.value]
            )
            new_state[instrument_choice.value][position] = new_instrument_state

        return new_state


class DrumPatternMusic21Converter:
    """
    Converts drum patterns into music21 scores.

    This class takes a drum pattern state array and converts it into a score
    using the music21 library.
    """

    # Mapping of instrument indices to MIDI pitches
    MIDI_PITCHES = {
        DrumInstruments.KICK: 36,
        DrumInstruments.SNARE: 38,
        DrumInstruments.HIHAT: 42,
    }

    def to_music21_score(self, state):
        """
        Converts a drum pattern state to a music21 stream for musical
        representation.

        Parameters:
            state (np.ndarray): The state array of the drum pattern.

        Returns:
            music21.stream.Score: The music21 score representation of the drum
                pattern.
        """
        score = stream.Score()
        score.metadata = metadata.Metadata(
            title="Drum Pattern generated by Cellular Automaton"
        )

        pattern_length = len(state[0])
        for drum_instrument in DrumInstruments:
            part = self._instrument_to_music21_part(
                drum_instrument, state, pattern_length
            )
            score.append(part)

        return score

    def _instrument_to_music21_part(
        self, drum_instrument, state, pattern_length
    ):
        """
        Converts a specific instrument's pattern to a music21 part.

        Parameters:
            drum_instrument (DrumInstruments): The drum instrument (KICK, SNARE,
                HIHAT).
            state (np.ndarray): The state array of the drum pattern.
            pattern_length (int): The length of the drum pattern.

        Returns:
            music21.stream.Part: A music21 part representation of the
                instrument's pattern.
        """
        part = stream.Part()
        part.insert(0, instrument.HiHatCymbal())

        for position in range(pattern_length):
            if (
                state[drum_instrument.value][position] == 1
            ):  # If the instrument is ON at this position
                note_pitch = self._get_midi_pitch_for_instrument(
                    drum_instrument
                )
                drum_note = note.Note()
                drum_note.pitch.midi = note_pitch
                drum_note.duration.quarterLength = 1
                part.append(drum_note)
            else:
                part.append(note.Rest(quarterLength=1))

        return part

    def _get_midi_pitch_for_instrument(self, drum_instrument):
        """
        Retrieves the MIDI pitch corresponding to a drum instrument.

        Parameters:
            drum_instrument (DrumInstruments): The drum instrument (KICK, SNARE,
                HIHAT).

        Returns:
            int: The MIDI pitch number corresponding to the instrument.
        """
        return self.MIDI_PITCHES.get(
            drum_instrument, 0
        )  # Default to 0 if not found


def main():
    """
    Generates a drum pattern using cellular automaton and converts it to a
    music21 score.

    This function creates a drum pattern of a specified length, evolves it
    through multiple steps, and then uses a music converter to transform the
    pattern into a music21 score, which is then displayed.
    """

    drum_generator = CellularAutomatonDrumGenerator(pattern_length=16)
    music_converter = DrumPatternMusic21Converter()

    for _ in range(8):
        drum_generator.step()
    score = music_converter.to_music21_score(drum_generator.state)

    score.show()


if __name__ == "__main__":
    main()
