import pretty_midi
import numpy as np
import joblib
import glob
import os
from math import isclose
import json

def beat_accuracy(pm, eps=0.001):
    # Use notes in first instrument to measure how well beats line up with note onsets
    
    if len(pm.instruments[0]) == 0:
        return -1, -1

    note_onsets = []

    for nt in pm.instruments[0].notes:
        note_onsets.append(nt.start)

    dists = []
    onbeats = []

    for beat in pm.get_beats():
        region_start = beat - eps
        region_end = beat + eps

        # get all note onsets between region_start and region_end
        region_note_onsets = [on for on in note_onsets if on >= region_start and on <= region_end]

        dist = []
        onbeat = []

        for on in region_note_onsets:
            dist.append(abs(beat - on))
            onbeat.append(isclose(beat, on))

        if len(dist) != 0:
            dists.append(sum(dist)/len(dist))
            onbeats.append(sum(onbeat)/len(onbeat))

    if (len(dists)) != 0:
        avg_dist = sum(dists)/len(dists)
        avg_onbeat = sum(onbeats)/len(onbeats)
    else:
        avg_dist = -1
        avg_onbeat = -1

    return avg_dist, avg_onbeat

def compute_statistics(midi_file):
    """
    Given a path to a MIDI file, compute a dictionary of statistics about it
    
    Parameters
    ----------
    midi_file : str
        Path to a MIDI file.
    
    Returns
    -------
    statistics : dict
        Dictionary reporting the values for different events in the file.
    """
    # Some MIDI files will raise Exceptions on loading, if they are invalid.
    # We just skip those.
    try:
        pm = pretty_midi.PrettyMIDI(midi_file)
        # Extract informative events from the MIDI file
        return {'n_instruments': len(pm.instruments),
                'notes': [n.pitch for i in pm.instruments for n in i.notes],
                'velocities': [n.velocity for i in pm.instruments for n in i.notes],
                'program_numbers': [i.program for i in pm.instruments if not i.is_drum],
                'key_numbers': [k.key_number for k in pm.key_signature_changes],
                'tempos': list(pm.get_tempo_changes()[1]),
                'time_signature_changes': pm.time_signature_changes,
                'end_time': pm.get_end_time(),
                'lyrics': [l.text for l in pm.lyrics],
                'beat_accuracy': beat_accuracy(pm)}
    # Silently ignore exceptions for a clean presentation (sorry Python!)
    except Exception as e:
        pass

# Compute statistics about every file in our collection in parallel using joblib
# We do things in parallel because there are tons so it would otherwise take too long!
statistics = joblib.Parallel(n_jobs=10, verbose=0)(
    joblib.delayed(compute_statistics)(midi_file)
    for midi_file in glob.glob(os.path.join('data', 'lmd_full', '*', '*.mid')))
# When an error occurred, None will be returned; filter those out.
statistics = [s for s in statistics if s is not None]

# Convert statistics to JSON string
statistics_json = json.dumps(statistics)

output_file = 'statistics.json'

# Write the JSON string to the file
with open(output_file, 'w') as file:
    file.write(statistics_json)
